# api/ticktrader_ws.py
import asyncio
import websockets
import json
import hmac
import hashlib
import time
from typing import Dict, List, Optional, Callable, Any # Adicionado Any
from datetime import datetime, timezone # Adicionado timezone
import ssl
from collections import deque

from config.settings import CONFIG
from utils.logger import setup_logger

logger = setup_logger("ticktrader_ws")

class TickData:
    """Estrutura de dados para tick"""
    def __init__(self, data: Dict[str, Any]): # Usar Any para maior flexibilidade nos dados de entrada
        self.symbol: Optional[str] = data.get('Symbol')
        # Garantir que o timestamp seja sempre UTC
        timestamp_ms = data.get('Timestamp', time.time() * 1000) # Default para tempo atual se ausente
        self.timestamp: datetime = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)

        best_bid = data.get('BestBid', {})
        self.bid: float = float(best_bid.get('Price', 0.0))
        self.bid_volume: float = float(best_bid.get('Volume', 0.0))

        best_ask = data.get('BestAsk', {})
        self.ask: float = float(best_ask.get('Price', 0.0))
        self.ask_volume: float = float(best_ask.get('Volume', 0.0))

        if self.bid > 0 and self.ask > 0: # Evitar mid/spread incorreto se bid/ask for 0
            self.mid: float = (self.bid + self.ask) / 2.0
            self.spread: float = self.ask - self.bid
        else: # Caso um dos preços seja inválido
            self.mid: float = 0.0
            self.spread: float = 0.0


    def to_dict(self) -> Dict[str, Any]: # Usar Any
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat(), # Melhor para serialização JSON
            'bid': self.bid,
            'ask': self.ask,
            'bid_volume': self.bid_volume,
            'ask_volume': self.ask_volume,
            'mid': self.mid,
            'spread': self.spread
        }

    @classmethod
    def from_dict(cls, data_dict: Dict[str, Any]): # Adicionado para consistência
        # Converte timestamp de string ISO para milissegundos se necessário
        if isinstance(data_dict.get('timestamp'), str):
            dt_obj = datetime.fromisoformat(data_dict['timestamp'].replace('Z', '+00:00'))
            timestamp_ms = int(dt_obj.timestamp() * 1000)
        else:
            timestamp_ms = data_dict.get('timestamp', int(time.time() * 1000))

        return cls({
            'Symbol': data_dict.get('symbol'),
            'Timestamp': timestamp_ms,
            'BestBid': {'Price': data_dict.get('bid'), 'Volume': data_dict.get('bid_volume')},
            'BestAsk': {'Price': data_dict.get('ask'), 'Volume': data_dict.get('ask_volume')}
        })

class DOMSnapshot:
    """Estrutura para snapshot do livro de ordens"""
    def __init__(self, data: Dict[str, Any]): # Usar Any
        self.symbol: Optional[str] = data.get('Symbol')
        timestamp_ms = data.get('Timestamp', time.time() * 1000)
        self.timestamp: datetime = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)

        self.bids: List[Tuple[float, float]] = []
        for level in data.get('Bids', []):
            try:
                self.bids.append((float(level['Price']), float(level['Volume'])))
            except (TypeError, ValueError) as e:
                logger.warning(f"Formato inválido para nível de DOM (bid): {level}. Erro: {e}")


        self.asks: List[Tuple[float, float]] = []
        for level in data.get('Asks', []):
            try:
                self.asks.append((float(level['Price']), float(level['Volume'])))
            except (TypeError, ValueError) as e:
                logger.warning(f"Formato inválido para nível de DOM (ask): {level}. Erro: {e}")


    def get_depth(self, levels: int = 10) -> Dict[str, Any]: # Usar Any
        """Retorna profundidade até N níveis"""
        return {
            'bids': self.bids[:levels],
            'asks': self.asks[:levels],
            'bid_volume': sum(vol for _, vol in self.bids[:levels]),
            'ask_volume': sum(vol for _, vol in self.asks[:levels])
        }

class TickTraderFeed:
    """Cliente WebSocket para feed de dados"""

    def __init__(self):
        self.ws: Optional[websockets.client.WebSocketClientProtocol] = None # Adicionada tipagem
        self.connected: bool = False
        self.subscriptions: set[str] = set() # Adicionada tipagem
        self.tick_buffer: deque[TickData] = deque(maxlen=10000) # Adicionada tipagem
        self.dom_buffer: Dict[str, DOMSnapshot] = {} # Adicionada tipagem
        self.callbacks: Dict[str, List[Callable]] = {} # Adicionada tipagem
        self.latency_ms: float = 0.0 # Inicializar como float
        self.message_id: int = 0
        self._receive_task: Optional[asyncio.Task] = None # Para gerenciar a tarefa de recebimento
        self._keep_alive_task: Optional[asyncio.Task] = None # Para gerenciar a tarefa de keep-alive

    async def connect(self):
        """Conecta ao WebSocket feed"""
        if self.connected:
            logger.info("Feed WebSocket já está conectado.")
            return

        try:
            logger.info(f"Conectando ao feed WebSocket: {CONFIG.WS_FEED_URL}")

            # Configurar SSL de forma mais segura para produção
            ssl_context = ssl.create_default_context()
            if "demowebapi.fxopen" in CONFIG.WS_FEED_URL: # Exemplo: relaxar para demo específico se necessário
                 ssl_context.check_hostname = False
                 ssl_context.verify_mode = ssl.CERT_NONE
            # Para produção real, não desabilitar check_hostname e verify_mode sem um bom motivo.
            # Se o servidor usar um certificado auto-assinado ou de uma CA privada,
            # você precisaria adicionar o certificado da CA usando ssl_context.load_verify_locations().

            self.ws = await websockets.connect(
                CONFIG.WS_FEED_URL,
                ssl=ssl_context,
                ping_interval=20, # Envia pings a cada 20s
                ping_timeout=10,  # Espera 10s por um pong
                # open_timeout=10 # Timeout para estabelecer a conexão (opcional)
            )

            self.connected = True
            logger.info("Feed WebSocket conectado. Autenticando...")

            # Autenticar
            await self._authenticate() # A autenticação deve ocorrer antes de iniciar _process_messages
                                      # para garantir que a sessão esteja pronta para receber dados de subscrição.


            # Iniciar processamento de mensagens e keep-alive após autenticação bem-sucedida
            if self.connected: # Verificar novamente após autenticação (poderia falhar)
                if self._receive_task is None or self._receive_task.done():
                    self._receive_task = asyncio.create_task(self._process_messages())
                # if self._keep_alive_task is None or self._keep_alive_task.done(): # Opcional, websockets lida com pings
                #     self._keep_alive_task = asyncio.create_task(self._send_keep_alive())

                logger.info("Autenticação do Feed WebSocket bem-sucedida e processador de mensagens iniciado.")


        except websockets.exceptions.InvalidStatusCode as e:
            logger.error(f"Falha ao conectar feed WebSocket: Status Code {e.status_code}. Resposta: {e.headers}")
            self.connected = False # Garantir que está False
            raise
        except Exception as e:
            logger.exception(f"Erro ao conectar feed WebSocket:") # Usar logger.exception
            self.connected = False # Garantir que está False
            raise

    async def _authenticate(self):
        """Autentica no WebSocket"""
        timestamp = str(int(time.time() * 1000))
        auth_message = {
            "Id": self._get_next_id(),
            "Request": "Login",
            "Params": {
                "AuthType": CONFIG.WEB_API_AUTH_TYPE, # Usar de CONFIG
                "WebApiId": CONFIG.WEB_API_TOKEN_ID,
                "WebApiKey": CONFIG.WEB_API_TOKEN_KEY,
                "Timestamp": timestamp,
                "Signature": self._generate_signature(timestamp)
            }
        }
        await self._send_message(auth_message)
        # A confirmação de login geralmente vem como uma mensagem do servidor.
        # Pode ser necessário esperar por uma resposta específica aqui ou em _process_messages.
        # Ex:
        # response = await self._wait_for_response(auth_message["Id"])
        # if not response or response.get("Response") != "Ok":
        #     self.connected = False
        #     raise ConnectionRefusedError(f"Autenticação do Feed falhou: {response}")
        # logger.info("Feed WebSocket autenticado")
        # Por enquanto, assumimos que _send_message é suficiente e o servidor confirma.
        # Vamos simular uma pequena espera para a resposta de login (idealmente, seria orientado a eventos)
        await asyncio.sleep(1) # Pequena pausa para processar a resposta do login


    def _generate_signature(self, timestamp: str) -> str:
        """Gera assinatura HMAC para autenticação"""
        message = f"{timestamp}{CONFIG.WEB_API_TOKEN_ID}{CONFIG.WEB_API_TOKEN_KEY}"
        signature = hmac.new(
            CONFIG.WEB_API_TOKEN_SECRET.encode('utf-8'), # Especificar encoding
            message.encode('utf-8'), # Especificar encoding
            hashlib.sha256
        ).hexdigest()
        return signature

    async def subscribe_symbol(self, symbol: str):
        """Inscreve para receber ticks de um símbolo"""
        if not self.connected:
            logger.warning(f"Não conectado. Não é possível inscrever em {symbol}.")
            return
        if symbol in self.subscriptions:
            logger.info(f"Já inscrito em ticks para {symbol}")
            return

        message = {
            "Id": self._get_next_id(),
            "Request": "SubscribeToSpots",
            "Params": {
                "Symbols": [symbol],
                "SubscribeToTicks": True,
                # "RequestType": "Subscribe" # Alguns APIs podem requerer isso
            }
        }
        await self._send_message(message)
        # A confirmação da subscrição virá como uma mensagem. O ideal é esperar por ela.
        # Por ora, adicionamos à lista e logamos.
        self.subscriptions.add(symbol)
        logger.info(f"Solicitação de inscrição em ticks enviada para {symbol}")


    async def subscribe_dom(self, symbol: str, depth: int = 10):
        """Inscreve para receber DOM (profundidade de mercado)"""
        if not self.connected:
            logger.warning(f"Não conectado. Não é possível inscrever no DOM de {symbol}.")
            return

        message = {
            "Id": self._get_next_id(),
            "Request": "SubscribeToDepth",
            "Params": {
                "Symbol": symbol,
                "Depth": depth
            }
        }
        await self._send_message(message)
        logger.info(f"Solicitação de inscrição em DOM enviada para {symbol} (profundidade: {depth})")

    async def _process_messages(self):
        """Processa mensagens recebidas"""
        if not self.ws:
            logger.error("Tentativa de processar mensagens sem conexão WebSocket (ws is None).")
            self.connected = False
            return

        while self.connected and self.ws and not self.ws.closed: # Adicionado check de ws.closed
            try:
                message_str = await asyncio.wait_for(self.ws.recv(), timeout=CONFIG.WS_TRADE_URL.count("fxopen") and 30.0 or 15.0) # Adicionado timeout
                data = json.loads(message_str) # Renomeado message para message_str

                server_time_ms = data.get('Timestamp')
                if server_time_ms:
                    local_time_ms = int(time.time() * 1000)
                    self.latency_ms = abs(local_time_ms - server_time_ms)

                event_type = data.get('Response') # TickTrader usa 'Response' para tipo de dado
                if not event_type and 'Event' in data: # Alguns usam 'Event'
                    event_type = data.get('Event')


                if event_type == 'Tick':
                    await self._handle_tick(data)
                elif event_type == 'Depth':
                    await self._handle_depth(data)
                elif event_type == 'Error':
                    logger.error(f"Erro do servidor Feed: {data.get('Result', {}).get('Message', data)}")
                elif event_type == 'Ok':
                    logger.info(f"Resposta Ok do servidor Feed: Id {data.get('Id')}, Result: {data.get('Result')}")
                elif event_type == 'Login': # Resposta ao Login
                    if data.get("Result", {}).get("Success", False):
                        logger.info(f"Login no Feed WebSocket bem-sucedido: {data.get('Id')}")
                    else:
                        logger.error(f"Falha no Login do Feed WebSocket: {data}")
                        await self.disconnect(reason="Falha no Login") # Desconectar se login falhar
                        return # Parar de processar mensagens
                # Adicionar outros tipos de resposta/evento conforme necessário

            except asyncio.TimeoutError:
                logger.debug("Nenhuma mensagem recebida do feed WebSocket, enviando ping (se necessário).")
                # A biblioteca websockets lida com pings automaticamente com ping_interval.
                # Se precisar de um ping customizado:
                # try:
                #     if self.ws and not self.ws.closed:
                #         pong_waiter = await self.ws.ping()
                #         await asyncio.wait_for(pong_waiter, timeout=5.0)
                # except asyncio.TimeoutError:
                #     logger.warning("Timeout no PING do feed WebSocket, conexão pode estar instável.")
                #     await self._reconnect() # Forçar reconexão
                #     return # Sair do loop para reconectar
                continue # Continuar o loop para a próxima tentativa de recv
            except websockets.exceptions.ConnectionClosedOK:
                logger.info("Conexão Feed WebSocket fechada normalmente.")
                self.connected = False
                break # Sair do loop de processamento
            except websockets.exceptions.ConnectionClosedError as e:
                logger.warning(f"Conexão Feed WebSocket fechada com erro: Code {e.code}, Reason: {e.reason}")
                self.connected = False
                break # Sair do loop e tentar reconectar
            except Exception as e:
                logger.exception(f"Erro ao processar mensagem do feed:") # Usar logger.exception
                # Considerar uma pequena pausa antes de continuar para evitar loops rápidos de erro
                await asyncio.sleep(1)


        logger.info("Loop de processamento de mensagens do Feed encerrado.")
        if self.connected: # Se saiu do loop mas ainda está "conectado", tentar reconectar
            await self._reconnect()


    async def _handle_tick(self, data: Dict[str, Any]): # Usar Any
        """Processa tick recebido"""
        try:
            tick_result = data.get('Result')
            if not tick_result:
                logger.warning(f"Tick recebido sem campo 'Result': {data}")
                return
            tick = TickData(tick_result)
        except Exception as e:
            logger.error(f"Erro ao parsear TickData: {e}. Dados: {data.get('Result')}")
            return


        self.tick_buffer.append(tick)

        if 'tick' in self.callbacks:
            for callback in self.callbacks['tick']:
                try:
                    await callback(tick)
                except Exception as e:
                    logger.error(f"Erro ao executar callback de tick: {e}")


    async def _handle_depth(self, data: Dict[str, Any]): # Usar Any
        """Processa snapshot de profundidade"""
        try:
            dom_result = data.get('Result')
            if not dom_result:
                logger.warning(f"Depth (DOM) recebido sem campo 'Result': {data}")
                return
            dom = DOMSnapshot(dom_result)
        except Exception as e:
            logger.error(f"Erro ao parsear DOMSnapshot: {e}. Dados: {data.get('Result')}")
            return

        if dom.symbol: # Checar se símbolo não é None
            self.dom_buffer[dom.symbol] = dom
        else:
            logger.warning(f"Snapshot DOM recebido sem símbolo: {dom_result}")


        if 'dom' in self.callbacks:
            for callback in self.callbacks['dom']:
                try:
                    await callback(dom)
                except Exception as e:
                    logger.error(f"Erro ao executar callback de DOM: {e}")

    async def get_tick(self) -> Optional[TickData]:
        """Obtém próximo tick do buffer"""
        if self.tick_buffer:
            return self.tick_buffer.popleft()
        return None

    async def get_dom_snapshot(self, symbol: str) -> Optional[DOMSnapshot]:
        """Obtém último snapshot do DOM"""
        return self.dom_buffer.get(symbol)

    def register_callback(self, event_type: str, callback: Callable):
        """Registra callback para eventos"""
        if event_type not in self.callbacks:
            self.callbacks[event_type] = []
        self.callbacks[event_type].append(callback)

    async def _send_message(self, message: Dict[str, Any]): # Usar Any
        """Envia mensagem ao servidor"""
        if self.ws and self.connected and not self.ws.closed: # Adicionado check de ws.closed
            try:
                await self.ws.send(json.dumps(message))
            except websockets.exceptions.ConnectionClosed:
                logger.warning("Tentativa de enviar mensagem em conexão WS fechada (Feed).")
                self.connected = False # Marcar como não conectado
                # A reconexão será tratada pelo loop _process_messages ou por uma chamada externa.
            except Exception as e:
                logger.error(f"Erro ao enviar mensagem WS (Feed): {e}")
        else:
            logger.warning("Feed WebSocket não conectado ou fechado, mensagem não enviada.")


    def _get_next_id(self) -> str:
        """Gera próximo ID de mensagem"""
        self.message_id += 1
        # Usar um prefixo para diferenciar de outros IDs que possam vir da API
        return f"ttfeed_req_{self.message_id}"


    async def _reconnect(self, max_retries=5, delay_seconds=5):
        """Reconecta ao WebSocket com backoff exponencial."""
        if hasattr(self, '_reconnect_lock') and self._reconnect_lock.locked():
            logger.info("Tentativa de reconexão já em progresso.")
            return
        self._reconnect_lock = asyncio.Lock()

        async with self._reconnect_lock:
            logger.info("Tentando reconectar Feed WebSocket...")
            for attempt in range(max_retries):
                if self.connected:
                    logger.info("Reconectado com sucesso ao Feed WebSocket durante a tentativa.")
                    return
                try:
                    # Limpar tarefas antigas se existirem e não estiverem concluídas
                    if self._receive_task and not self._receive_task.done():
                        self._receive_task.cancel()
                    if self._keep_alive_task and not self._keep_alive_task.done():
                        self._keep_alive_task.cancel()

                    if self.ws and not self.ws.closed: # Fechar conexão antiga se existir
                        await self.ws.close()

                    await self.connect() # connect() agora inclui autenticação e inicia _process_messages

                    if self.connected:
                        # Re-inscrever
                        current_subscriptions = list(self.subscriptions) # Copiar para evitar modificação durante iteração
                        self.subscriptions.clear() # Limpar para que subscribe_symbol reinscreva
                        for symbol in current_subscriptions:
                            await self.subscribe_symbol(symbol)
                            # Adicionar subscrição ao DOM também se aplicável
                            # Ex: await self.subscribe_dom(symbol, CONFIG.DOM_LEVELS)
                        logger.info("Reconexão e reinscrição no Feed WebSocket bem-sucedidas.")
                        return # Sair se conectado
                except Exception as e:
                    logger.error(f"Falha na tentativa {attempt + 1} de reconexão do Feed: {e}")
                    if attempt < max_retries - 1:
                        current_delay = delay_seconds * (2 ** attempt)
                        logger.info(f"Próxima tentativa de reconexão do Feed em {current_delay} segundos.")
                        await asyncio.sleep(current_delay)
                    else:
                        logger.error("Máximo de tentativas de reconexão do Feed atingido. Desistindo.")
                        # Aqui você pode decidir tomar uma ação mais drástica, como notificar um admin.
                        return # Desistir após max_retries

    def is_connected(self) -> bool:
        """Verifica se está conectado"""
        # Consideramos conectado se o websocket existe, está aberto e `self.connected` é True.
        return self.connected and self.ws is not None and not self.ws.closed


    async def get_latency(self) -> float:
        """Retorna latência em ms"""
        return self.latency_ms

    async def disconnect(self, reason: str = "Desconexão solicitada"): # Adicionado parâmetro reason
        """Desconecta do WebSocket"""
        logger.info(f"Desconectando Feed WebSocket: {reason}")
        self.connected = False # Definir primeiro para parar loops
        if self._receive_task and not self._receive_task.done():
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                logger.info("Tarefa de recebimento de mensagens do Feed cancelada.")
        if self._keep_alive_task and not self._keep_alive_task.done():
            self._keep_alive_task.cancel()
            try:
                await self._keep_alive_task
            except asyncio.CancelledError:
                logger.info("Tarefa de keep-alive do Feed cancelada.")

        if self.ws:
            try:
                await self.ws.close(code=1000, reason=reason)
            except Exception as e:
                logger.error(f"Erro ao fechar conexão Feed WebSocket: {e}")
            self.ws = None
        logger.info("Feed WebSocket desconectado.")


class TickTraderTrade:
    """Cliente WebSocket para execução de ordens"""

    def __init__(self):
        self.ws: Optional[websockets.client.WebSocketClientProtocol] = None # Adicionada tipagem
        self.connected: bool = False
        self.authenticated: bool = False
        self.account_info: Dict[str, Any] = {} # Adicionada tipagem
        self.positions: Dict[str, Dict[str, Any]] = {} # Adicionada tipagem e Any
        self.orders: Dict[str, Dict[str, Any]] = {} # Adicionada tipagem e Any
        self.callbacks: Dict[str, List[Callable]] = {} # Adicionada tipagem
        self.message_id: int = 0
        self.pending_responses: Dict[str, Optional[Dict[str, Any]]] = {} # Adicionada tipagem e Any
        self._receive_task: Optional[asyncio.Task] = None # Para gerenciar a tarefa de recebimento

    async def connect(self):
        """Conecta ao WebSocket de trading"""
        if self.connected:
            logger.info("Trade WebSocket já está conectado.")
            return

        try:
            logger.info(f"Conectando ao trade WebSocket: {CONFIG.WS_TRADE_URL}")

            ssl_context = ssl.create_default_context()
            if "demowebapi.fxopen" in CONFIG.WS_TRADE_URL: # Exemplo
                 ssl_context.check_hostname = False
                 ssl_context.verify_mode = ssl.CERT_NONE

            self.ws = await websockets.connect(
                CONFIG.WS_TRADE_URL,
                ssl=ssl_context,
                ping_interval=20,
                ping_timeout=10
            )
            self.connected = True # Marcar como conectado aqui, autenticação depois

            # Iniciar processamento de mensagens ANTES de autenticar,
            # para que a resposta do Login seja capturada.
            if self._receive_task is None or self._receive_task.done():
                self._receive_task = asyncio.create_task(self._process_messages())

            logger.info("Trade WebSocket conectado, autenticando...")
            await self.authenticate() # Autenticar após iniciar o processador de mensagens

        except websockets.exceptions.InvalidStatusCode as e:
            logger.error(f"Falha ao conectar trade WebSocket: Status Code {e.status_code}. Resposta: {e.headers}")
            self.connected = False
            raise
        except Exception as e:
            logger.exception(f"Erro ao conectar trade WebSocket:")
            self.connected = False
            raise

    async def authenticate(self):
        """Autentica conta de trading"""
        if not self.connected:
            logger.error("Não é possível autenticar, Trade WebSocket não conectado.")
            return

        timestamp = str(int(time.time() * 1000))
        auth_message = {
            "Id": self._get_next_id(),
            "Request": "Login",
            "Params": {
                "AuthType": CONFIG.WEB_API_AUTH_TYPE, # Usar de CONFIG
                "WebApiId": CONFIG.WEB_API_TOKEN_ID,
                "WebApiKey": CONFIG.WEB_API_TOKEN_KEY,
                "Timestamp": timestamp,
                "Signature": self._generate_signature(timestamp)
            }
        }

        response = await self._send_and_wait(auth_message)

        if response and response.get('Result', {}).get('Success', False): # Checagem mais robusta
            self.authenticated = True
            await self._load_account_info()
            logger.info("Autenticado com sucesso no trade WebSocket")
        else:
            self.authenticated = False # Garantir que está False
            error_msg = response.get('Result', {}).get('Message', 'Resposta de autenticação inválida ou falha.') if response else "Sem resposta para autenticação."
            logger.error(f"Falha na autenticação do Trade WebSocket: {error_msg}")
            # Considerar desconectar ou tentar novamente após um tempo
            await self.disconnect(reason=f"Falha na autenticação: {error_msg}")
            raise ConnectionRefusedError(f"Falha na autenticação do Trade: {error_msg}")


    def _generate_signature(self, timestamp: str) -> str:
        """Gera assinatura HMAC"""
        message = f"{timestamp}{CONFIG.WEB_API_TOKEN_ID}{CONFIG.WEB_API_TOKEN_KEY}"
        signature = hmac.new(
            CONFIG.WEB_API_TOKEN_SECRET.encode('utf-8'), # Especificar encoding
            message.encode('utf-8'), # Especificar encoding
            hashlib.sha256
        ).hexdigest()
        return signature

    async def _load_account_info(self):
        """Carrega informações da conta"""
        if not self.authenticated:
            logger.warning("Não autenticado, não é possível carregar informações da conta.")
            return

        account_msg = {
            "Id": self._get_next_id(),
            "Request": "GetAccount"
        }

        response = await self._send_and_wait(account_msg)
        if response and response.get('Result') and not response.get('Error'):
            self.account_info = response['Result']
            logger.info(f"Informações da conta carregadas: Login {self.account_info.get('Login')}, "
                       f"Balanço: {self.account_info.get('Balance')} {self.account_info.get('Currency')}")
        else:
            logger.error(f"Falha ao carregar informações da conta: {response}")


    async def create_market_order(self, symbol: str, side: str, volume: float, strategy_comment: Optional[str]=None) -> Optional[Dict[str, Any]]:
        """Cria ordem a mercado"""
        if not self.is_connected():
            logger.error("Não conectado ao Trade WebSocket. Ordem a mercado não enviada.")
            return None

        comment = strategy_comment if strategy_comment else f"Bot_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

        order_msg = {
            "Id": self._get_next_id(),
            "Request": "CreateOrder",
            "Params": {
                "Symbol": symbol,
                "Type": "Market", # Ou o tipo exato esperado pela API, ex: 0 para Market
                "Side": side.capitalize(), # API pode esperar "Buy" ou "Sell"
                "Volume": volume,
                "Comment": comment
                # "Slippage": CONFIG.MAX_SLIPPAGE_PIPS # Se a API suportar slippage na ordem
            }
        }

        response = await self._send_and_wait(order_msg)

        if response and response.get('Result') and not response.get('Error'):
            order = response['Result']
            if 'Id' in order: # Verificar se o ID da ordem está presente
                 self.orders[order['Id']] = order
                 logger.info(f"Ordem a mercado criada: {order['Id']} | {side} {volume} {symbol}")
                 return order
            else:
                logger.error(f"Resposta de criação de ordem a mercado sem ID: {response}")
                return None
        else:
            error_details = response.get('Error', response.get('Result', {})) if response else "Sem resposta"
            logger.error(f"Falha ao criar ordem a mercado: {error_details}")
            return None


    async def create_limit_order(self, symbol: str, side: str, volume: float,
                                price: float, stop_loss: Optional[float] = None,
                                take_profit: Optional[float] = None, strategy_comment: Optional[str]=None) -> Optional[Dict[str, Any]]:
        """Cria ordem limite"""
        if not self.is_connected():
            logger.error("Não conectado ao Trade WebSocket. Ordem limite não enviada.")
            return None

        comment = strategy_comment if strategy_comment else f"Bot_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        params: Dict[str, Any] = { # Adicionada tipagem
            "Symbol": symbol,
            "Type": "Limit", # Ou o tipo exato esperado pela API, ex: 1 para Limit
            "Side": side.capitalize(),
            "Volume": volume,
            "Price": price,
            "Comment": comment
            # "ExpirationTime": int((datetime.now(timezone.utc) + timedelta(hours=1)).timestamp() * 1000) # Exemplo: expira em 1 hora
        }

        if stop_loss is not None:
            params["StopLoss"] = stop_loss
        if take_profit is not None:
            params["TakeProfit"] = take_profit

        order_msg = {
            "Id": self._get_next_id(),
            "Request": "CreateOrder",
            "Params": params
        }

        response = await self._send_and_wait(order_msg)

        if response and response.get('Result') and not response.get('Error'):
            order = response['Result']
            if 'Id' in order:
                self.orders[order['Id']] = order
                logger.info(f"Ordem limite criada: {order['Id']} | {side} {volume} {symbol} @ {price}")
                return order
            else:
                logger.error(f"Resposta de criação de ordem limite sem ID: {response}")
                return None
        else:
            error_details = response.get('Error', response.get('Result', {})) if response else "Sem resposta"
            logger.error(f"Falha ao criar ordem limite: {error_details}")
            return None


    async def modify_order(self, order_id: str,
                          new_price: Optional[float] = None, # Adicionado new_price para ordens limite
                          stop_loss: Optional[float] = None,
                          take_profit: Optional[float] = None,
                          new_volume: Optional[float] = None) -> bool: # Adicionado new_volume
        """Modifica ordem existente"""
        if not self.is_connected():
            logger.error("Não conectado. Modificação de ordem falhou.")
            return False

        params: Dict[str, Any] = {"OrderId": order_id} # Adicionada tipagem

        if new_price is not None: # Para ordens limite
            params["Price"] = new_price
        if stop_loss is not None:
            params["StopLoss"] = stop_loss
        if take_profit is not None:
            params["TakeProfit"] = take_profit
        if new_volume is not None:
            params["Volume"] = new_volume


        # Não modificar se não houver nada para mudar além do OrderId
        if len(params) <= 1:
            logger.info(f"Nenhuma modificação solicitada para a ordem {order_id}.")
            return True # Considerar como sucesso se nada precisou ser mudado


        modify_msg = {
            "Id": self._get_next_id(),
            "Request": "ModifyOrder",
            "Params": params
        }

        response = await self._send_and_wait(modify_msg)
        if response and response.get('Result') and not response.get('Error'):
            logger.info(f"Ordem {order_id} modificada com sucesso.")
            # Atualizar a ordem local se a modificação for bem-sucedida
            if order_id in self.orders:
                if new_price is not None: self.orders[order_id]['Price'] = new_price
                if stop_loss is not None: self.orders[order_id]['StopLoss'] = stop_loss
                if take_profit is not None: self.orders[order_id]['TakeProfit'] = take_profit
                if new_volume is not None: self.orders[order_id]['Volume'] = new_volume
            return True
        else:
            error_details = response.get('Error', response.get('Result', {})) if response else "Sem resposta"
            logger.error(f"Falha ao modificar ordem {order_id}: {error_details}")
            return False


    async def cancel_order(self, order_id: str) -> bool:
        """Cancela ordem"""
        if not self.is_connected():
            logger.error("Não conectado. Cancelamento de ordem falhou.")
            return False

        cancel_msg = {
            "Id": self._get_next_id(),
            "Request": "CancelOrder",
            "Params": {
                "OrderId": order_id
            }
        }

        response = await self._send_and_wait(cancel_msg)

        if response and response.get('Result') and not response.get('Error'): # Checagem mais específica do sucesso
            self.orders.pop(order_id, None)
            logger.info(f"Ordem cancelada: {order_id}")
            return True
        else:
            error_details = response.get('Error', response.get('Result', {})) if response else "Sem resposta"
            logger.error(f"Falha ao cancelar ordem {order_id}: {error_details}")
            return False


    async def close_position(self, position_id: str, volume: Optional[float] = None) -> bool: # Adicionado volume para fechamento parcial
        """Fecha posição"""
        if not self.is_connected():
            logger.error("Não conectado. Fechamento de posição falhou.")
            return False

        params: Dict[str, Any] = {"PositionId": position_id} # Adicionada tipagem
        if volume is not None: # Para fechamento parcial
            params["Volume"] = volume


        close_msg = {
            "Id": self._get_next_id(),
            "Request": "ClosePosition",
            "Params": params
        }

        response = await self._send_and_wait(close_msg)

        if response and response.get('Result') and not response.get('Error'): # Checagem mais específica
            # Se fechamento parcial, a posição pode não ser removida completamente
            if volume is None or (position_id in self.positions and volume >= self.positions[position_id].get('Volume', float('inf'))):
                self.positions.pop(position_id, None)
            logger.info(f"Posição {position_id} (ou parte dela) fechada.")
            return True
        else:
            error_details = response.get('Error', response.get('Result', {})) if response else "Sem resposta"
            logger.error(f"Falha ao fechar posição {position_id}: {error_details}")
            return False


    async def get_positions(self) -> List[Dict[str, Any]]:
        """Obtém posições abertas"""
        if not self.is_connected():
            logger.error("Não conectado. Não é possível obter posições.")
            return []

        positions_msg = {
            "Id": self._get_next_id(),
            "Request": "GetPositions"
        }
        response = await self._send_and_wait(positions_msg)

        if response and response.get('Result') and isinstance(response['Result'], list) and not response.get('Error'):
            self.positions = {p['Id']: p for p in response['Result'] if 'Id' in p}
            return response['Result']
        else:
            error_details = response.get('Error', response.get('Result', {})) if response else "Sem resposta"
            logger.error(f"Falha ao obter posições: {error_details}")
            return []


    async def get_account_balance(self) -> float:
        """Obtém balanço da conta"""
        if not self.is_connected(): # Verificar conexão antes de carregar
            logger.error("Não conectado. Não é possível obter balanço.")
            return 0.0
        await self._load_account_info() # Recarregar informações da conta
        return float(self.account_info.get('Balance', 0.0)) # Default para float


    async def _process_messages(self):
        """Processa mensagens do servidor"""
        if not self.ws:
            logger.error("Tentativa de processar mensagens sem conexão WebSocket (ws is None) em Trade.")
            self.connected = False
            return

        while self.connected and self.ws and not self.ws.closed: # Adicionado check de ws.closed
            try:
                message_str = await asyncio.wait_for(self.ws.recv(), timeout=CONFIG.WS_TRADE_URL.count("fxopen") and 30.0 or 15.0) # Adicionado timeout
                data = json.loads(message_str) # Renomeado message para message_str

                message_id_resp = data.get('Id') # Renomeado para evitar conflito
                if message_id_resp and message_id_resp in self.pending_responses:
                    # Colocar na fila de respostas pendentes
                    queue = self.pending_responses[message_id_resp]
                    if isinstance(queue, asyncio.Queue): # Checar se é uma Queue
                        await queue.put(data)
                    else: # Fallback se não for Queue (lógica original)
                         self.pending_responses[message_id_resp] = data


                if data.get('Event'):
                    await self._handle_event(data)
                # Algumas APIs de trading podem enviar 'ExecutionReport' ou similar como evento
                elif data.get('Response') == 'ExecutionReport': # Exemplo, verificar nome exato
                     await self._handle_event({'Event': 'ExecutionReport', 'Data': data.get('Result')})


            except asyncio.TimeoutError:
                logger.debug("Nenhuma mensagem recebida do trade WebSocket, enviando ping (se necessário).")
                # Lógica de ping similar ao Feed, se aplicável e não tratada pela lib.
                continue
            except websockets.exceptions.ConnectionClosedOK:
                logger.info("Conexão Trade WebSocket fechada normally.")
                self.connected = False
                self.authenticated = False
                break
            except websockets.exceptions.ConnectionClosedError as e:
                logger.warning(f"Conexão Trade WebSocket fechada com erro: Code {e.code}, Reason: {e.reason}")
                self.connected = False
                self.authenticated = False
                break # Sair e tentar reconectar
            except Exception as e:
                logger.exception(f"Erro ao processar mensagem trade:")
                await asyncio.sleep(1)

        logger.info("Loop de processamento de mensagens do Trade encerrado.")
        if self.connected: # Se saiu do loop mas ainda está "conectado", tentar reconectar
            await self._reconnect()


    async def _handle_event(self, data: Dict[str, Any]): # Usar Any
        """Processa eventos do servidor"""
        event_type = data.get('Event')
        event_data = data.get('Data', data.get('Result', data)) # Pegar dados do evento

        logger.debug(f"Evento Trade recebido: {event_type}, Data: {event_data}")


        if event_type == 'OrderUpdate' or event_type == 'ExecutionReport': # Unificar
            order = event_data.get('Order', event_data) # 'ExecutionReport' pode ter os dados da ordem diretamente
            if order and 'Id' in order:
                self.orders[order['Id']] = order
                if 'OrderUpdate' in self.callbacks: # Chamar callbacks de OrderUpdate
                    for callback in self.callbacks['OrderUpdate']:
                        await callback(order) # Passar o objeto da ordem


        elif event_type == 'PositionUpdate':
            position = event_data.get('Position', event_data)
            if position and 'Id' in position:
                self.positions[position['Id']] = position
                if 'PositionUpdate' in self.callbacks:
                    for callback in self.callbacks['PositionUpdate']:
                        await callback(position)

        elif event_type == 'TradeUpdate': # Execução de um trade (fill)
            trade_fill_data = event_data.get('Trade', event_data)
            if 'TradeUpdate' in self.callbacks:
                 for callback in self.callbacks['TradeUpdate']:
                    await callback(trade_fill_data)


        # Executar callbacks genéricos para o tipo de evento se existirem
        if event_type in self.callbacks:
            for callback in self.callbacks[event_type]:
                try:
                    await callback(event_data) # Passar os dados do evento
                except Exception as e:
                    logger.error(f"Erro ao executar callback de evento trade '{event_type}': {e}")

    async def _send_and_wait(self, message: Dict[str, Any], timeout: float = 10.0) -> Optional[Dict[str, Any]]: # Usar Any
        """Envia mensagem e aguarda resposta usando asyncio.Queue para segurança de concorrência."""
        if not self.is_connected():
            logger.error("Não conectado, _send_and_wait falhou.")
            return None

        message_id_req = message['Id'] # Renomeado
        # Usar uma asyncio.Queue para cada resposta esperada
        response_queue: asyncio.Queue[Optional[Dict[str, Any]]] = asyncio.Queue(maxsize=1)
        self.pending_responses[message_id_req] = response_queue


        await self._send_message(message)

        try:
            # Aguardar resposta da fila
            response = await asyncio.wait_for(response_queue.get(), timeout=timeout)
            return response
        except asyncio.TimeoutError:
            logger.warning(f"Timeout aguardando resposta para {message_id_req}")
            return None
        except Exception as e:
            logger.error(f"Erro inesperado em _send_and_wait para {message_id_req}: {e}")
            return None
        finally:
            # Remover a fila de pending_responses após uso ou timeout
            if message_id_req in self.pending_responses:
                del self.pending_responses[message_id_req]


    async def _send_message(self, message: Dict[str, Any]): # Usar Any
        """Envia mensagem ao servidor"""
        if self.ws and self.connected and not self.ws.closed: # Adicionado check de ws.closed
            try:
                await self.ws.send(json.dumps(message))
            except websockets.exceptions.ConnectionClosed:
                logger.warning("Tentativa de enviar mensagem em conexão WS fechada (Trade).")
                self.connected = False
                self.authenticated = False
            except Exception as e:
                logger.error(f"Erro ao enviar mensagem WS (Trade): {e}")
        else:
            logger.warning("Trade WebSocket não conectado ou fechado, mensagem não enviada.")


    def _get_next_id(self) -> str:
        """Gera próximo ID de mensagem"""
        self.message_id += 1
        return f"ttrade_req_{self.message_id}" # Prefixo diferente para Trade


    async def _reconnect(self, max_retries=5, delay_seconds=5):
        """Reconecta ao WebSocket Trade com backoff."""
        if hasattr(self, '_reconnect_lock_trade') and self._reconnect_lock_trade.locked():
            logger.info("Tentativa de reconexão (Trade) já em progresso.")
            return
        self._reconnect_lock_trade = asyncio.Lock()

        async with self._reconnect_lock_trade:
            logger.info("Tentando reconectar Trade WebSocket...")
            self.authenticated = False # Resetar estado de autenticação
            for attempt in range(max_retries):
                if self.is_connected(): # is_connected() checa self.connected e self.authenticated
                    logger.info("Reconectado com sucesso ao Trade WebSocket durante a tentativa.")
                    return
                try:
                    if self._receive_task and not self._receive_task.done():
                        self._receive_task.cancel()
                    if self.ws and not self.ws.closed:
                        await self.ws.close()

                    await self.connect() # connect() agora tenta autenticar e inicia _process_messages

                    if self.is_connected():
                        logger.info("Reconexão ao Trade WebSocket bem-sucedida.")
                        # Não precisa re-inscrever em ordens/posições, o servidor envia atualizações.
                        return # Sair se conectado
                except Exception as e:
                    logger.error(f"Falha na tentativa {attempt + 1} de reconexão do Trade: {e}")
                    if attempt < max_retries - 1:
                        current_delay = delay_seconds * (2 ** attempt)
                        logger.info(f"Próxima tentativa de reconexão do Trade em {current_delay} segundos.")
                        await asyncio.sleep(current_delay)
                    else:
                        logger.error("Máximo de tentativas de reconexão do Trade atingido. Desistindo.")
                        return


    def is_connected(self) -> bool:
        """Verifica conexão e autenticação"""
        return self.connected and self.authenticated and self.ws is not None and not self.ws.closed


    async def disconnect(self, reason: str = "Desconexão solicitada"): # Adicionado parâmetro reason
        """Desconecta"""
        logger.info(f"Desconectando Trade WebSocket: {reason}")
        self.connected = False
        self.authenticated = False
        if self._receive_task and not self._receive_task.done():
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                logger.info("Tarefa de recebimento de mensagens do Trade cancelada.")

        if self.ws:
            try:
                await self.ws.close(code=1000, reason=reason)
            except Exception as e:
                logger.error(f"Erro ao fechar conexão Trade WebSocket: {e}")
            self.ws = None
        logger.info("Trade WebSocket desconectado.")

    # Adicionar método para registrar callbacks
    def register_callback(self, event_type: str, callback: Callable):
        """Registra callback para eventos de trade (OrderUpdate, PositionUpdate, etc.)"""
        if event_type not in self.callbacks:
            self.callbacks[event_type] = []
        self.callbacks[event_type].append(callback)