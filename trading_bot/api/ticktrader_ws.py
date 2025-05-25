# api/ticktrader_ws.py
import asyncio
import websockets
import json
import hmac
import hashlib
import time
from typing import Dict, List, Optional, Callable
from datetime import datetime
import ssl
from collections import deque

from config.settings import CONFIG
from utils.logger import setup_logger

logger = setup_logger("ticktrader_ws")

class TickData:
    """Estrutura de dados para tick"""
    def __init__(self, data: Dict):
        self.symbol = data.get('Symbol')
        self.timestamp = datetime.fromtimestamp(data.get('Timestamp', 0) / 1000)
        self.bid = float(data.get('BestBid', {}).get('Price', 0))
        self.ask = float(data.get('BestAsk', {}).get('Price', 0))
        self.bid_volume = float(data.get('BestBid', {}).get('Volume', 0))
        self.ask_volume = float(data.get('BestAsk', {}).get('Volume', 0))
        self.mid = (self.bid + self.ask) / 2
        self.spread = self.ask - self.bid
        
    def to_dict(self) -> Dict:
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp,
            'bid': self.bid,
            'ask': self.ask,
            'bid_volume': self.bid_volume,
            'ask_volume': self.ask_volume,
            'mid': self.mid,
            'spread': self.spread
        }

class DOMSnapshot:
    """Estrutura para snapshot do livro de ordens"""
    def __init__(self, data: Dict):
        self.symbol = data.get('Symbol')
        self.timestamp = datetime.fromtimestamp(data.get('Timestamp', 0) / 1000)
        self.bids = [(float(level['Price']), float(level['Volume'])) 
                     for level in data.get('Bids', [])]
        self.asks = [(float(level['Price']), float(level['Volume'])) 
                     for level in data.get('Asks', [])]
        
    def get_depth(self, levels: int = 10) -> Dict:
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
        self.ws = None
        self.connected = False
        self.subscriptions = set()
        self.tick_buffer = deque(maxlen=10000)
        self.dom_buffer = {}
        self.callbacks = {}
        self.latency_ms = 0
        self.message_id = 0
        
    async def connect(self):
        """Conecta ao WebSocket feed"""
        try:
            logger.info(f"Conectando ao feed WebSocket: {CONFIG.WS_FEED_URL}")
            
            # Configurar SSL se necessário
            ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            
            # Conectar
            self.ws = await websockets.connect(
                CONFIG.WS_FEED_URL,
                ssl=ssl_context,
                ping_interval=20,
                ping_timeout=10
            )
            
            self.connected = True
            
            # Iniciar processamento de mensagens
            asyncio.create_task(self._process_messages())
            
            # Autenticar
            await self._authenticate()
            
            logger.info("Feed WebSocket conectado e autenticado")
            
        except Exception as e:
            logger.error(f"Erro ao conectar feed WebSocket: {e}")
            raise
    
    async def _authenticate(self):
        """Autentica no WebSocket"""
        # Gerar timestamp
        timestamp = str(int(time.time() * 1000))
        
        # Criar mensagem de autenticação
        auth_message = {
            "Id": self._get_next_id(),
            "Request": "Login",
            "Params": {
                "AuthType": "HMAC",
                "WebApiId": CONFIG.WEB_API_TOKEN_ID,
                "WebApiKey": CONFIG.WEB_API_TOKEN_KEY,
                "Timestamp": timestamp,
                "Signature": self._generate_signature(timestamp)
            }
        }
        
        await self._send_message(auth_message)
        
        # Aguardar resposta
        await asyncio.sleep(1)
    
    def _generate_signature(self, timestamp: str) -> str:
        """Gera assinatura HMAC para autenticação"""
        message = f"{timestamp}{CONFIG.WEB_API_TOKEN_ID}{CONFIG.WEB_API_TOKEN_KEY}"
        signature = hmac.new(
            CONFIG.WEB_API_TOKEN_SECRET.encode(),
            message.encode(),
            hashlib.sha256
        ).hexdigest()
        return signature
    
    async def subscribe_symbol(self, symbol: str):
        """Inscreve para receber ticks de um símbolo"""
        if symbol in self.subscriptions:
            return
        
        message = {
            "Id": self._get_next_id(),
            "Request": "SubscribeToSpots",
            "Params": {
                "Symbols": [symbol],
                "SubscribeToTicks": True
            }
        }
        
        await self._send_message(message)
        self.subscriptions.add(symbol)
        logger.info(f"Inscrito em ticks para {symbol}")
    
    async def subscribe_dom(self, symbol: str, depth: int = 10):
        """Inscreve para receber DOM (profundidade de mercado)"""
        message = {
            "Id": self._get_next_id(),
            "Request": "SubscribeToDepth",
            "Params": {
                "Symbol": symbol,
                "Depth": depth
            }
        }
        
        await self._send_message(message)
        logger.info(f"Inscrito em DOM para {symbol} (profundidade: {depth})")
    
    async def _process_messages(self):
        """Processa mensagens recebidas"""
        while self.connected:
            try:
                message = await self.ws.recv()
                data = json.loads(message)
                
                # Calcular latência
                if 'Timestamp' in data:
                    server_time = data['Timestamp']
                    local_time = int(time.time() * 1000)
                    self.latency_ms = abs(local_time - server_time)
                
                # Processar por tipo
                if data.get('Response') == 'Tick':
                    await self._handle_tick(data)
                elif data.get('Response') == 'Depth':
                    await self._handle_depth(data)
                elif data.get('Response') == 'Error':
                    logger.error(f"Erro do servidor: {data}")
                
            except websockets.exceptions.ConnectionClosed:
                logger.warning("Conexão WebSocket fechada")
                self.connected = False
                await self._reconnect()
            except Exception as e:
                logger.error(f"Erro ao processar mensagem: {e}")
    
    async def _handle_tick(self, data: Dict):
        """Processa tick recebido"""
        tick = TickData(data['Result'])
        
        # Adicionar ao buffer
        self.tick_buffer.append(tick)
        
        # Executar callbacks
        if 'tick' in self.callbacks:
            for callback in self.callbacks['tick']:
                await callback(tick)
    
    async def _handle_depth(self, data: Dict):
        """Processa snapshot de profundidade"""
        dom = DOMSnapshot(data['Result'])
        
        # Atualizar buffer
        self.dom_buffer[dom.symbol] = dom
        
        # Executar callbacks
        if 'dom' in self.callbacks:
            for callback in self.callbacks['dom']:
                await callback(dom)
    
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
    
    async def _send_message(self, message: Dict):
        """Envia mensagem ao servidor"""
        if self.ws and self.connected:
            await self.ws.send(json.dumps(message))
    
    def _get_next_id(self) -> str:
        """Gera próximo ID de mensagem"""
        self.message_id += 1
        return f"msg_{self.message_id}"
    
    async def _reconnect(self):
        """Reconecta ao WebSocket"""
        logger.info("Tentando reconectar...")
        await asyncio.sleep(5)
        await self.connect()
        
        # Re-inscrever
        for symbol in self.subscriptions:
            await self.subscribe_symbol(symbol)
    
    def is_connected(self) -> bool:
        """Verifica se está conectado"""
        return self.connected
    
    async def get_latency(self) -> float:
        """Retorna latência em ms"""
        return self.latency_ms
    
    async def disconnect(self):
        """Desconecta do WebSocket"""
        self.connected = False
        if self.ws:
            await self.ws.close()
        logger.info("Feed WebSocket desconectado")


class TickTraderTrade:
    """Cliente WebSocket para execução de ordens"""
    
    def __init__(self):
        self.ws = None
        self.connected = False
        self.authenticated = False
        self.account_info = {}
        self.positions = {}
        self.orders = {}
        self.callbacks = {}
        self.message_id = 0
        
    async def connect(self):
        """Conecta ao WebSocket de trading"""
        try:
            logger.info(f"Conectando ao trade WebSocket: {CONFIG.WS_TRADE_URL}")
            
            # Configurar SSL
            ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            
            # Conectar
            self.ws = await websockets.connect(
                CONFIG.WS_TRADE_URL,
                ssl=ssl_context,
                ping_interval=20,
                ping_timeout=10
            )
            
            self.connected = True
            
            # Iniciar processamento
            asyncio.create_task(self._process_messages())
            
            logger.info("Trade WebSocket conectado")
            
        except Exception as e:
            logger.error(f"Erro ao conectar trade WebSocket: {e}")
            raise
    
    async def authenticate(self):
        """Autentica conta de trading"""
        timestamp = str(int(time.time() * 1000))
        
        auth_message = {
            "Id": self._get_next_id(),
            "Request": "Login",
            "Params": {
                "AuthType": "HMAC",
                "WebApiId": CONFIG.WEB_API_TOKEN_ID,
                "WebApiKey": CONFIG.WEB_API_TOKEN_KEY,
                "Timestamp": timestamp,
                "Signature": self._generate_signature(timestamp)
            }
        }
        
        response = await self._send_and_wait(auth_message)
        
        if response and response.get('Result'):
            self.authenticated = True
            await self._load_account_info()
            logger.info("Autenticado com sucesso no trade WebSocket")
        else:
            raise Exception("Falha na autenticação")
    
    def _generate_signature(self, timestamp: str) -> str:
        """Gera assinatura HMAC"""
        message = f"{timestamp}{CONFIG.WEB_API_TOKEN_ID}{CONFIG.WEB_API_TOKEN_KEY}"
        signature = hmac.new(
            CONFIG.WEB_API_TOKEN_SECRET.encode(),
            message.encode(),
            hashlib.sha256
        ).hexdigest()
        return signature
    
    async def _load_account_info(self):
        """Carrega informações da conta"""
        # Obter informações da conta
        account_msg = {
            "Id": self._get_next_id(),
            "Request": "GetAccount"
        }
        
        response = await self._send_and_wait(account_msg)
        if response and response.get('Result'):
            self.account_info = response['Result']
            logger.info(f"Conta carregada: {self.account_info.get('Login')} | "
                       f"Balanço: {self.account_info.get('Balance')} {self.account_info.get('Currency')}")
    
    async def create_market_order(self, symbol: str, side: str, volume: float) -> Optional[Dict]:
        """Cria ordem a mercado"""
        order_msg = {
            "Id": self._get_next_id(),
            "Request": "CreateOrder",
            "Params": {
                "Symbol": symbol,
                "Type": "Market",
                "Side": side,
                "Volume": volume,
                "Comment": f"Bot_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            }
        }
        
        response = await self._send_and_wait(order_msg)
        
        if response and response.get('Result'):
            order = response['Result']
            self.orders[order['Id']] = order
            logger.info(f"Ordem criada: {order['Id']} | {side} {volume} {symbol}")
            return order
        
        return None
    
    async def create_limit_order(self, symbol: str, side: str, volume: float, 
                                price: float, stop_loss: Optional[float] = None,
                                take_profit: Optional[float] = None) -> Optional[Dict]:
        """Cria ordem limite"""
        params = {
            "Symbol": symbol,
            "Type": "Limit",
            "Side": side,
            "Volume": volume,
            "Price": price,
            "Comment": f"Bot_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        }
        
        if stop_loss:
            params["StopLoss"] = stop_loss
        
        if take_profit:
            params["TakeProfit"] = take_profit
        
        order_msg = {
            "Id": self._get_next_id(),
            "Request": "CreateOrder",
            "Params": params
        }
        
        response = await self._send_and_wait(order_msg)
        
        if response and response.get('Result'):
            order = response['Result']
            self.orders[order['Id']] = order
            logger.info(f"Ordem limite criada: {order['Id']} | {side} {volume} @ {price}")
            return order
        
        return None
    
    async def modify_order(self, order_id: str, stop_loss: Optional[float] = None,
                          take_profit: Optional[float] = None) -> bool:
        """Modifica ordem existente"""
        params = {"OrderId": order_id}
        
        if stop_loss is not None:
            params["StopLoss"] = stop_loss
        
        if take_profit is not None:
            params["TakeProfit"] = take_profit
        
        modify_msg = {
            "Id": self._get_next_id(),
            "Request": "ModifyOrder",
            "Params": params
        }
        
        response = await self._send_and_wait(modify_msg)
        return response and response.get('Result') is not None
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancela ordem"""
        cancel_msg = {
            "Id": self._get_next_id(),
            "Request": "CancelOrder",
            "Params": {
                "OrderId": order_id
            }
        }
        
        response = await self._send_and_wait(cancel_msg)
        
        if response and response.get('Result'):
            self.orders.pop(order_id, None)
            logger.info(f"Ordem cancelada: {order_id}")
            return True
        
        return False
    
    async def close_position(self, position_id: str) -> bool:
        """Fecha posição"""
        close_msg = {
            "Id": self._get_next_id(),
            "Request": "ClosePosition",
            "Params": {
                "PositionId": position_id
            }
        }
        
        response = await self._send_and_wait(close_msg)
        
        if response and response.get('Result'):
            self.positions.pop(position_id, None)
            logger.info(f"Posição fechada: {position_id}")
            return True
        
        return False
    
    async def get_positions(self) -> List[Dict]:
        """Obtém posições abertas"""
        positions_msg = {
            "Id": self._get_next_id(),
            "Request": "GetPositions"
        }
        
        response = await self._send_and_wait(positions_msg)
        
        if response and response.get('Result'):
            self.positions = {p['Id']: p for p in response['Result']}
            return response['Result']
        
        return []
    
    async def get_account_balance(self) -> float:
        """Obtém balanço da conta"""
        await self._load_account_info()
        return float(self.account_info.get('Balance', 0))
    
    async def _process_messages(self):
        """Processa mensagens do servidor"""
        while self.connected:
            try:
                message = await self.ws.recv()
                data = json.loads(message)
                
                # Processar resposta
                message_id = data.get('Id')
                if message_id and message_id in self.pending_responses:
                    self.pending_responses[message_id] = data
                
                # Processar eventos
                if data.get('Event'):
                    await self._handle_event(data)
                
            except websockets.exceptions.ConnectionClosed:
                logger.warning("Trade WebSocket desconectado")
                self.connected = False
                await self._reconnect()
            except Exception as e:
                logger.error(f"Erro ao processar mensagem trade: {e}")
    
    async def _handle_event(self, data: Dict):
        """Processa eventos do servidor"""
        event_type = data.get('Event')
        
        if event_type == 'OrderUpdate':
            order = data.get('Order')
            if order:
                self.orders[order['Id']] = order
                
        elif event_type == 'PositionUpdate':
            position = data.get('Position')
            if position:
                self.positions[position['Id']] = position
                
        elif event_type == 'TradeUpdate':
            # Processar execução de trade
            pass
        
        # Executar callbacks
        if event_type in self.callbacks:
            for callback in self.callbacks[event_type]:
                await callback(data)
    
    async def _send_and_wait(self, message: Dict, timeout: float = 10) -> Optional[Dict]:
        """Envia mensagem e aguarda resposta"""
        if not hasattr(self, 'pending_responses'):
            self.pending_responses = {}
        
        message_id = message['Id']
        self.pending_responses[message_id] = None
        
        # Enviar mensagem
        await self._send_message(message)
        
        # Aguardar resposta
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.pending_responses[message_id] is not None:
                response = self.pending_responses.pop(message_id)
                return response
            await asyncio.sleep(0.1)
        
        # Timeout
        self.pending_responses.pop(message_id, None)
        logger.warning(f"Timeout aguardando resposta para {message_id}")
        return None
    
    async def _send_message(self, message: Dict):
        """Envia mensagem ao servidor"""
        if self.ws and self.connected:
            await self.ws.send(json.dumps(message))
    
    def _get_next_id(self) -> str:
        """Gera próximo ID de mensagem"""
        self.message_id += 1
        return f"trade_{self.message_id}"
    
    async def _reconnect(self):
        """Reconecta ao WebSocket"""
        logger.info("Reconectando trade WebSocket...")
        await asyncio.sleep(5)
        await self.connect()
        await self.authenticate()
    
    def is_connected(self) -> bool:
        """Verifica conexão"""
        return self.connected and self.authenticated
    
    async def disconnect(self):
        """Desconecta"""
        self.connected = False
        if self.ws:
            await self.ws.close()
        logger.info("Trade WebSocket desconectado")