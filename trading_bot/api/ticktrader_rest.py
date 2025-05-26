# api/ticktrader_rest.py
"""Cliente REST para TickTrader API"""
import aiohttp
import asyncio
import hmac
import hashlib
import time
import json
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import pandas as pd

from config.settings import CONFIG
from utils.logger import setup_logger

logger = setup_logger("ticktrader_rest")

class TickTraderREST:
    """Cliente REST para operações não-realtime do TickTrader"""

    def __init__(self):
        self.base_url = CONFIG.REST_API_URL
        self.session: Optional[aiohttp.ClientSession] = None # Adicionada tipagem
        self.authenticated = False # Este atributo não parece ser usado ativamente para gate requests

    async def __aenter__(self):
        """Context manager entry"""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        await self.disconnect()

    async def connect(self):
        """Cria sessão HTTP"""
        if not self.session or self.session.closed: # Adicionado check se a sessão está fechada
            timeout = aiohttp.ClientTimeout(total=30) # Timeout total de 30s para requisições
            self.session = aiohttp.ClientSession(timeout=timeout)
            logger.info("Sessão REST criada")
            # A autenticação via headers é feita por requisição,
            # então self.authenticated pode não ser necessário aqui.

    async def disconnect(self):
        """Fecha sessão HTTP"""
        if self.session and not self.session.closed: # Adicionado check se a sessão não está fechada
            await self.session.close()
            self.session = None # Definir como None após fechar
            logger.info("Sessão REST fechada")

    def _generate_signature(self, timestamp: str, body: str = "") -> str:
        """Gera assinatura HMAC para autenticação"""
        # A documentação da API do TickTrader deve especificar a ordem exata e o conteúdo da mensagem.
        # O formato comum é: timestamp + tokenId + apiKey + body
        message = f"{timestamp}{CONFIG.WEB_API_TOKEN_ID}{CONFIG.WEB_API_TOKEN_KEY}{body}"
        signature = hmac.new(
            CONFIG.WEB_API_TOKEN_SECRET.encode('utf-8'), # Especificar encoding
            message.encode('utf-8'), # Especificar encoding
            hashlib.sha256
        ).hexdigest()
        return signature

    def _get_headers(self, body: str = "") -> Dict[str, str]:
        """Gera headers com autenticação"""
        timestamp = str(int(time.time() * 1000)) # Timestamp em milissegundos

        headers = {
            "Content-Type": "application/json",
            # "Authorization": f"Bearer {CONFIG.WEB_API_TOKEN_KEY}", # Geralmente não é "Bearer" para HMAC com chaves separadas
            "X-TT-APIKEY": CONFIG.WEB_API_TOKEN_ID, # Ajustado para um header comum de API Key ID
            "X-TT-TIMESTAMP": timestamp, # Ajustado para um header comum de Timestamp
            "X-TT-SIGNATURE": self._generate_signature(timestamp, body) # Ajustado para um header comum de Signature
        }
        # Nota: Os nomes exatos dos headers (X-TT-APIKEY, X-TT-TIMESTAMP, X-TT-SIGNATURE)
        # dependem da especificação da API TickTrader. Os usados aqui são exemplos comuns.
        # O original usava X-API-KEY, X-API-TIMESTAMP, X-API-SIGNATURE e Authorization: Bearer.
        # É importante verificar qual o formato correto. Se Authorization: Bearer é usado,
        # geralmente o token é um JWT ou OAuth token, não uma API Key para HMAC.
        # Mantendo o formato original do seu código para os headers:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {CONFIG.WEB_API_TOKEN_KEY}", # Se este for realmente o formato esperado
            "X-API-KEY": CONFIG.WEB_API_TOKEN_ID,
            "X-API-TIMESTAMP": timestamp,
            "X-API-SIGNATURE": self._generate_signature(timestamp, body)
        }


        return headers

    async def _request(self, method: str, endpoint: str,
                      params: Optional[Dict] = None, # Adicionado params para GET requests
                      data: Optional[Dict] = None) -> Optional[Dict]:
        """Faz requisição HTTP"""
        if not self.session or self.session.closed: # Adicionado check se a sessão está fechada
            await self.connect()
            if not self.session: # Se a conexão falhar
                logger.error("Não foi possível estabelecer sessão HTTP para _request.")
                return None


        url = f"{self.base_url}{endpoint}"
        # Para requisições GET, os parâmetros vão na URL, não no body.
        # O body para assinatura HMAC deve ser vazio para GET se não houver payload.
        body_for_signature = json.dumps(data) if data and method.upper() != "GET" else ""
        headers = self._get_headers(body_for_signature)

        request_kwargs = {"headers": headers}
        if params and method.upper() == "GET":
            request_kwargs["params"] = params
        if data and method.upper() != "GET": # Data é para POST, PUT, etc.
            request_kwargs["data"] = json.dumps(data)


        try:
            async with self.session.request(
                method=method,
                url=url,
                **request_kwargs # Usar os kwargs montados
            ) as response:
                response_text = await response.text() # Ler o texto para debug em caso de erro
                if response.status >= 200 and response.status < 300: # Sucesso é geralmente 2xx
                    # Lidar com respostas vazias que não são JSON válido
                    if response.content_type == 'application/json':
                        return await response.json()
                    elif not response_text: # Resposta vazia com status de sucesso
                        return {"status": "success", "message": "Operation successful, no content returned."}
                    else: # Resposta não-JSON mas com sucesso
                        return {"status": "success", "content": response_text}

                else:
                    logger.error(f"Erro na requisição {method} {url}: "
                               f"{response.status} - {response_text}")
                    # Tentar parsear erro JSON se possível
                    if response.content_type == 'application/json':
                        try:
                            error_json = json.loads(response_text)
                            return {"error": error_json, "status_code": response.status}
                        except json.JSONDecodeError:
                            pass # cai no retorno genérico abaixo
                    return {"error": response_text, "status_code": response.status}


        except aiohttp.ClientConnectorError as e:
            logger.error(f"Erro de conexão em {method} {url}: {e}")
            return {"error": str(e), "status_code": "CONNECTION_ERROR"}
        except asyncio.TimeoutError:
            logger.error(f"Timeout na requisição {method} {url}")
            return {"error": "Request timed out", "status_code": "TIMEOUT_ERROR"}
        except Exception as e:
            logger.exception(f"Erro inesperado na requisição REST {method} {url}:") # Usar logger.exception
            return {"error": str(e), "status_code": "UNKNOWN_ERROR"}

    # === Account Methods ===

    async def get_account_info(self) -> Optional[Dict]:
        """Obtém informações da conta"""
        return await self._request("GET", "/api/v2/account")

    async def get_account_balance(self) -> Optional[float]:
        """Obtém saldo da conta"""
        account_info_response = await self.get_account_info() # Renomeado para evitar conflito
        if account_info_response and not account_info_response.get("error"):
            return float(account_info_response.get('balance', 0.0)) # Adicionado default
        logger.warning(f"Não foi possível obter saldo da conta: {account_info_response}")
        return None


    async def get_trading_statistics(self,
                                   start_date: Optional[datetime] = None,
                                   end_date: Optional[datetime] = None) -> Optional[Dict]:
        """Obtém estatísticas de trading"""
        request_params: Dict[str, Any] = {} # Renomeado para evitar conflito com 'params' da função _request

        if start_date:
            request_params['from'] = int(start_date.timestamp() * 1000)
        if end_date:
            request_params['to'] = int(end_date.timestamp() * 1000)

        # Para requisições GET, os parâmetros devem ser passados no argumento 'params'
        return await self._request("GET", "/api/v2/statistics", params=request_params)


    # === Historical Data Methods ===

    async def download_history(self,
                             symbol: str,
                             start_date: datetime,
                             end_date: datetime,
                             timeframe: str = "M1") -> Optional[pd.DataFrame]:
        """
        Baixa dados históricos

        Args:
            symbol: Símbolo do par
            start_date: Data inicial
            end_date: Data final
            timeframe: Timeframe (M1, M5, H1, etc)

        Returns:
            DataFrame com dados históricos
        """
        logger.info(f"Baixando histórico {symbol} de {start_date.strftime('%Y-%m-%d')} a {end_date.strftime('%Y-%m-%d')} ({timeframe})")

        payload = { # Renomeado de data para payload para clareza
            "symbol": symbol,
            "from": int(start_date.timestamp() * 1000),
            "to": int(end_date.timestamp() * 1000),
            "period": timeframe # O campo geralmente é 'period' ou 'interval', verificar API
        }
        # Se a API realmente usa "timeframe" no payload, mantenha. Se não, ajuste.
        # Supondo que o original "timeframe" está correto para o payload:
        payload["timeframe"] = timeframe


        result = await self._request("POST", "/api/v2/history/download", data=payload)

        if result and not result.get("error") and 'bars' in result:
            if not result['bars']: # Lista de barras vazia
                logger.info(f"Nenhum dado histórico encontrado para {symbol} no período especificado.")
                return pd.DataFrame()

            df = pd.DataFrame(result['bars'])
            # Verificar se colunas esperadas existem antes de converter
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True) # Especificar UTC
                df.set_index('timestamp', inplace=True)
            else:
                logger.warning("Coluna 'timestamp' não encontrada nos dados históricos.")
                return pd.DataFrame()

            # Renomear colunas para um padrão (ex: open, high, low, close, volume) se necessário
            # Ex: df.rename(columns={'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'}, inplace=True)
            # Isso depende do formato retornado pela API. Assumindo que as colunas já são o,h,l,c,v ou similar.


            logger.info(f"Baixados {len(df)} candles para {symbol}")
            return df
        elif result and result.get("error"):
            logger.error(f"Erro da API ao baixar histórico para {symbol}: {result.get('error')}")
        else:
            logger.warning(f"Resposta inesperada ou vazia ao baixar histórico para {symbol}: {result}")


        return pd.DataFrame() # Retornar DataFrame vazio em caso de falha ou nenhum dado


    async def download_ticks(self,
                           symbol: str,
                           start_date: datetime,
                           end_date: datetime) -> Optional[pd.DataFrame]:
        """Baixa dados de ticks"""
        logger.info(f"Baixando ticks {symbol} de {start_date.strftime('%Y-%m-%d')} a {end_date.strftime('%Y-%m-%d')}")


        all_ticks_list = [] # Renomeado para clareza
        current_req_date = start_date # Renomeado para clareza
        batch_days = 1  # 1 dia por vez (ajuste conforme limites da API)

        while current_req_date < end_date:
            batch_end_date = min(current_req_date + timedelta(days=batch_days), end_date) # Renomeado

            payload = { # Renomeado
                "symbol": symbol,
                "from": int(current_req_date.timestamp() * 1000),
                "to": int(batch_end_date.timestamp() * 1000),
                "level": 1 # Para obter dados de tick (Level 1)
            }
            # Se a API realmente usa "level", mantenha. Caso contrário, pode ser desnecessário.
            # O endpoint é /api/v2/ticks/download, então o tipo de dado já está implícito.

            result = await self._request("POST", "/api/v2/ticks/download", data=payload)

            if result and not result.get("error") and 'ticks' in result:
                if result['ticks']: # Se a lista de ticks não estiver vazia
                    all_ticks_list.extend(result['ticks'])
                    logger.info(f"Baixados {len(result['ticks'])} ticks para {symbol} em {current_req_date.date()}")
                else:
                    logger.info(f"Nenhum tick encontrado para {symbol} em {current_req_date.date()}")

            elif result and result.get("error"):
                logger.error(f"Erro da API ao baixar ticks para {symbol} em {current_req_date.date()}: {result.get('error')}")
                # Decidir se deve parar ou continuar para o próximo batch
                # break # Exemplo: parar em caso de erro
            else:
                 logger.warning(f"Resposta inesperada ou vazia ao baixar ticks para {symbol} em {current_req_date.date()}: {result}")


            current_req_date = batch_end_date
            await asyncio.sleep(CONFIG.REST_API_URL.count("fxopen") and 0.5 or 0.1)  # Rate limiting mais curto se não for FXOpen, ou configurável

        if all_ticks_list:
            df = pd.DataFrame(all_ticks_list)
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True) # Especificar UTC
                df.set_index('timestamp', inplace=True)
            else:
                logger.warning("Coluna 'timestamp' não encontrada nos dados de ticks.")
                return pd.DataFrame()

            # Adicionar colunas 'bid', 'ask' se não existirem, baseadas em uma estrutura comum de tick
            # Exemplo: se 'ticks' retorna uma lista de listas [timestamp, bid, ask, bidVol, askVol]
            # Ou se retorna uma lista de dicionários {'ts': ..., 'b': ..., 'a': ...}
            # Isso depende da estrutura exata dos dados de tick da API
            # Assumindo que os ticks já têm as colunas necessárias ou são processados para tê-las.

            logger.info(f"Total de {len(df)} ticks baixados para {symbol}")
            return df

        return pd.DataFrame()

    # === Symbol Information ===

    async def get_symbols(self) -> Optional[List[Dict]]:
        """Obtém lista de símbolos disponíveis"""
        result = await self._request("GET", "/api/v2/symbols")
        return result if result and not result.get("error") else None


    async def get_symbol_info(self, symbol: str) -> Optional[Dict]:
        """Obtém informações detalhadas de um símbolo"""
        result = await self._request("GET", f"/api/v2/symbols/{symbol}")
        return result if result and not result.get("error") else None


    async def get_symbol_specification(self, symbol: str) -> Optional[Dict]:
        """Obtém especificações de trading do símbolo"""
        info = await self.get_symbol_info(symbol)

        if info and not info.get("error"): # Verificar se não houve erro
            # Usar .get com defaults para evitar KeyError se algum campo estiver faltando
            return {
                'symbol': symbol,
                'digits': info.get('digits', 5),
                'contract_size': info.get('contractSize', 100000.0), # Float para consistência
                'min_lot': info.get('minVolume', 0.01),
                'max_lot': info.get('maxVolume', 100.0), # Float
                'lot_step': info.get('volumeStep', 0.01),
                'spread': info.get('spread', 0.0), # Float
                'swap_long': info.get('swapLong', 0.0), # Float
                'swap_short': info.get('swapShort', 0.0), # Float
                'margin_required': info.get('marginFactor', 0.0) # API pode usar 'marginFactor' ou 'marginRate'
                                                                # 'marginRequired' como está no original pode ser um campo calculado ou diferente.
                                                                # Verifique a documentação da API para o campo correto que representa a margem.
            }

        return None

    # === Trading History ===

    async def get_closed_positions(self,
                                 start_date: Optional[datetime] = None,
                                 end_date: Optional[datetime] = None) -> Optional[List[Dict]]:
        """Obtém histórico de posições fechadas"""
        request_params: Dict[str, Any] = {} # Renomeado

        if start_date:
            request_params['from'] = int(start_date.timestamp() * 1000)
        if end_date:
            request_params['to'] = int(end_date.timestamp() * 1000)

        result = await self._request("GET", "/api/v2/trades/closed", params=request_params) # Usar 'params' para GET
        return result.get('trades') if result and not result.get("error") and 'trades' in result else None # A API pode retornar uma lista sob a chave 'trades' ou 'positions'


    async def get_order_history(self,
                              start_date: Optional[datetime] = None,
                              end_date: Optional[datetime] = None) -> Optional[List[Dict]]:
        """Obtém histórico de ordens"""
        request_params: Dict[str, Any] = {} # Renomeado

        if start_date:
            request_params['from'] = int(start_date.timestamp() * 1000)
        if end_date:
            request_params['to'] = int(end_date.timestamp() * 1000)

        result = await self._request("GET", "/api/v2/orders/history", params=request_params) # Usar 'params' para GET
        return result.get('orders') if result and not result.get("error") and 'orders' in result else None


    # === Market Data ===

    async def get_current_prices(self, symbols: List[str]) -> Optional[Dict[str, Dict]]:
        """Obtém preços atuais para múltiplos símbolos"""
        payload = {"symbols": symbols} # Renomeado
        result = await self._request("POST", "/api/v2/quotes", data=payload) # POST é comum para enviar uma lista de símbolos

        if result and not result.get("error") and 'quotes' in result:
            # Adicionar tratamento para 'bid'/'ask' possivelmente não serem floats diretos
            quotes_dict = {}
            for quote in result['quotes']:
                try:
                    bid_price = float(quote.get('bid', 0.0))
                    ask_price = float(quote.get('ask', 0.0))
                    quotes_dict[quote['symbol']] = {
                        'bid': bid_price,
                        'ask': ask_price,
                        'timestamp': quote.get('timestamp', int(time.time() * 1000)) # Default para agora se não vier
                    }
                except (ValueError, TypeError) as e:
                    logger.warning(f"Erro ao parsear cotação para {quote.get('symbol')}: {e}. Cotação: {quote}")
            return quotes_dict


        return None

    async def get_market_depth(self, symbol: str, depth: int = 10) -> Optional[Dict]:
        """Obtém profundidade de mercado"""
        payload = { # Renomeado
            "symbol": symbol,
            "depth": depth
        }

        result = await self._request("POST", "/api/v2/depth", data=payload) # POST é comum aqui
        # A resposta da API deve ser verificada. Se 'result' contiver o book, retorná-lo.
        # Ex: if result and not result.get("error") and 'bids' in result and 'asks' in result:
        return result if result and not result.get("error") else None


    # === Economic Calendar ===

    async def get_economic_calendar(self,
                                  start_date: datetime,
                                  end_date: datetime,
                                  currencies: Optional[List[str]] = None,
                                  importance: Optional[List[str]] = None) -> Optional[List[Dict]]: # Adicionado importance
        """Obtém calendário econômico"""
        payload = { # Renomeado
            "from": int(start_date.timestamp() * 1000),
            "to": int(end_date.timestamp() * 1000)
        }

        if currencies:
            payload["currencies"] = currencies
        if importance: # Adicionado filtro por importância
            payload["importance"] = importance # ex: ["high", "medium"]


        result = await self._request("POST", "/api/v2/calendar", data=payload) # POST é comum

        if result and not result.get("error") and 'events' in result:
            # A filtragem de 'high_impact' no original pode ser mantida ou tornada configurável
            # Se o filtro 'importance' for passado para a API, essa filtragem manual pode não ser necessária.
            # events_to_return = []
            # for event in result['events']:
            #     if not importance or event.get('impact', '').lower() in [imp.lower() for imp in importance]:
            #         events_to_return.append(event)
            # return events_to_return
            return result['events']


        return None

    # === Utility Methods ===

    async def test_connection(self) -> bool:
        """Testa conexão com a API"""
        try:
            result = await self._request("GET", "/api/v2/ping")
            # Uma resposta de sucesso para ping pode não ser um JSON, ou ser um JSON simples
            return result is not None and not result.get("error")
        except Exception as e:
            logger.exception(f"Erro ao testar conexão REST:")
            return False

    async def get_server_time(self) -> Optional[datetime]:
        """Obtém horário do servidor"""
        result = await self._request("GET", "/api/v2/time")

        if result and not result.get("error") and 'time' in result: # API pode retornar 'time' ou 'timestamp'
            # Assumindo que o timestamp é em milissegundos
            server_timestamp_ms = result.get('time', result.get('timestamp'))
            if server_timestamp_ms is not None:
                return datetime.fromtimestamp(int(server_timestamp_ms) / 1000, tz=timezone.utc) # Adicionar timezone.utc
            else:
                logger.warning(f"Campo de timestamp não encontrado na resposta /api/v2/time: {result}")
        elif result and result.get("error"):
            logger.error(f"Erro da API ao obter horário do servidor: {result.get('error')}")
        else:
            logger.warning(f"Resposta inesperada ao obter horário do servidor: {result}")


        return None