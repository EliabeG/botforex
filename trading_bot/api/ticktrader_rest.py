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
        self.session = None
        self.authenticated = False
        
    async def __aenter__(self):
        """Context manager entry"""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        await self.disconnect()
    
    async def connect(self):
        """Cria sessão HTTP"""
        if not self.session:
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(timeout=timeout)
            logger.info("Sessão REST criada")
    
    async def disconnect(self):
        """Fecha sessão HTTP"""
        if self.session:
            await self.session.close()
            self.session = None
            logger.info("Sessão REST fechada")
    
    def _generate_signature(self, timestamp: str, body: str = "") -> str:
        """Gera assinatura HMAC para autenticação"""
        message = f"{timestamp}{CONFIG.WEB_API_TOKEN_ID}{CONFIG.WEB_API_TOKEN_KEY}{body}"
        signature = hmac.new(
            CONFIG.WEB_API_TOKEN_SECRET.encode(),
            message.encode(),
            hashlib.sha256
        ).hexdigest()
        return signature
    
    def _get_headers(self, body: str = "") -> Dict[str, str]:
        """Gera headers com autenticação"""
        timestamp = str(int(time.time() * 1000))
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {CONFIG.WEB_API_TOKEN_KEY}",
            "X-API-KEY": CONFIG.WEB_API_TOKEN_ID,
            "X-API-TIMESTAMP": timestamp,
            "X-API-SIGNATURE": self._generate_signature(timestamp, body)
        }
        
        return headers
    
    async def _request(self, method: str, endpoint: str, 
                      data: Optional[Dict] = None) -> Optional[Dict]:
        """Faz requisição HTTP"""
        if not self.session:
            await self.connect()
        
        url = f"{self.base_url}{endpoint}"
        body = json.dumps(data) if data else ""
        headers = self._get_headers(body)
        
        try:
            async with self.session.request(
                method=method,
                url=url,
                headers=headers,
                data=body
            ) as response:
                
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    logger.error(f"Erro na requisição {method} {endpoint}: "
                               f"{response.status} - {error_text}")
                    return None
                    
        except Exception as e:
            logger.error(f"Erro na requisição REST: {e}")
            return None
    
    # === Account Methods ===
    
    async def get_account_info(self) -> Optional[Dict]:
        """Obtém informações da conta"""
        return await self._request("GET", "/api/v2/account")
    
    async def get_account_balance(self) -> Optional[float]:
        """Obtém saldo da conta"""
        account = await self.get_account_info()
        if account:
            return float(account.get('balance', 0))
        return None
    
    async def get_trading_statistics(self, 
                                   start_date: Optional[datetime] = None,
                                   end_date: Optional[datetime] = None) -> Optional[Dict]:
        """Obtém estatísticas de trading"""
        params = {}
        
        if start_date:
            params['from'] = int(start_date.timestamp() * 1000)
        if end_date:
            params['to'] = int(end_date.timestamp() * 1000)
        
        return await self._request("GET", "/api/v2/statistics", params)
    
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
        logger.info(f"Baixando histórico {symbol} de {start_date} a {end_date}")
        
        data = {
            "symbol": symbol,
            "from": int(start_date.timestamp() * 1000),
            "to": int(end_date.timestamp() * 1000),
            "timeframe": timeframe
        }
        
        result = await self._request("POST", "/api/v2/history/download", data)
        
        if result and 'bars' in result:
            # Converter para DataFrame
            df = pd.DataFrame(result['bars'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            logger.info(f"Baixados {len(df)} candles")
            return df
        
        return None
    
    async def download_ticks(self,
                           symbol: str,
                           start_date: datetime,
                           end_date: datetime) -> Optional[pd.DataFrame]:
        """Baixa dados de ticks"""
        logger.info(f"Baixando ticks {symbol} de {start_date} a {end_date}")
        
        # TickTrader pode ter limites, fazer em batches
        all_ticks = []
        current_date = start_date
        batch_days = 1  # 1 dia por vez
        
        while current_date < end_date:
            batch_end = min(current_date + timedelta(days=batch_days), end_date)
            
            data = {
                "symbol": symbol,
                "from": int(current_date.timestamp() * 1000),
                "to": int(batch_end.timestamp() * 1000)
            }
            
            result = await self._request("POST", "/api/v2/ticks/download", data)
            
            if result and 'ticks' in result:
                all_ticks.extend(result['ticks'])
                logger.info(f"Baixados {len(result['ticks'])} ticks para "
                           f"{current_date.date()}")
            
            current_date = batch_end
            await asyncio.sleep(0.5)  # Rate limiting
        
        if all_ticks:
            # Converter para DataFrame
            df = pd.DataFrame(all_ticks)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            logger.info(f"Total de {len(df)} ticks baixados")
            return df
        
        return None
    
    # === Symbol Information ===
    
    async def get_symbols(self) -> Optional[List[Dict]]:
        """Obtém lista de símbolos disponíveis"""
        return await self._request("GET", "/api/v2/symbols")
    
    async def get_symbol_info(self, symbol: str) -> Optional[Dict]:
        """Obtém informações detalhadas de um símbolo"""
        return await self._request("GET", f"/api/v2/symbols/{symbol}")
    
    async def get_symbol_specification(self, symbol: str) -> Optional[Dict]:
        """Obtém especificações de trading do símbolo"""
        info = await self.get_symbol_info(symbol)
        
        if info:
            return {
                'symbol': symbol,
                'digits': info.get('digits', 5),
                'contract_size': info.get('contractSize', 100000),
                'min_lot': info.get('minVolume', 0.01),
                'max_lot': info.get('maxVolume', 100),
                'lot_step': info.get('volumeStep', 0.01),
                'spread': info.get('spread', 0),
                'swap_long': info.get('swapLong', 0),
                'swap_short': info.get('swapShort', 0),
                'margin_required': info.get('marginRequired', 0)
            }
        
        return None
    
    # === Trading History ===
    
    async def get_closed_positions(self,
                                 start_date: Optional[datetime] = None,
                                 end_date: Optional[datetime] = None) -> Optional[List[Dict]]:
        """Obtém histórico de posições fechadas"""
        params = {}
        
        if start_date:
            params['from'] = int(start_date.timestamp() * 1000)
        if end_date:
            params['to'] = int(end_date.timestamp() * 1000)
        
        return await self._request("GET", "/api/v2/trades/closed", params)
    
    async def get_order_history(self,
                              start_date: Optional[datetime] = None,
                              end_date: Optional[datetime] = None) -> Optional[List[Dict]]:
        """Obtém histórico de ordens"""
        params = {}
        
        if start_date:
            params['from'] = int(start_date.timestamp() * 1000)
        if end_date:
            params['to'] = int(end_date.timestamp() * 1000)
        
        return await self._request("GET", "/api/v2/orders/history", params)
    
    # === Market Data ===
    
    async def get_current_prices(self, symbols: List[str]) -> Optional[Dict[str, Dict]]:
        """Obtém preços atuais para múltiplos símbolos"""
        data = {"symbols": symbols}
        result = await self._request("POST", "/api/v2/quotes", data)
        
        if result and 'quotes' in result:
            return {
                quote['symbol']: {
                    'bid': float(quote['bid']),
                    'ask': float(quote['ask']),
                    'timestamp': quote['timestamp']
                }
                for quote in result['quotes']
            }
        
        return None
    
    async def get_market_depth(self, symbol: str, depth: int = 10) -> Optional[Dict]:
        """Obtém profundidade de mercado"""
        data = {
            "symbol": symbol,
            "depth": depth
        }
        
        return await self._request("POST", "/api/v2/depth", data)
    
    # === Economic Calendar ===
    
    async def get_economic_calendar(self,
                                  start_date: datetime,
                                  end_date: datetime,
                                  currencies: Optional[List[str]] = None) -> Optional[List[Dict]]:
        """Obtém calendário econômico"""
        data = {
            "from": int(start_date.timestamp() * 1000),
            "to": int(end_date.timestamp() * 1000)
        }
        
        if currencies:
            data["currencies"] = currencies
        
        result = await self._request("POST", "/api/v2/calendar", data)
        
        if result and 'events' in result:
            # Filtrar apenas eventos de alto impacto
            high_impact = [
                event for event in result['events']
                if event.get('impact', '').lower() == 'high'
            ]
            
            return high_impact
        
        return None
    
    # === Utility Methods ===
    
    async def test_connection(self) -> bool:
        """Testa conexão com a API"""
        try:
            result = await self._request("GET", "/api/v2/ping")
            return result is not None
        except Exception as e:
            logger.error(f"Erro ao testar conexão: {e}")
            return False
    
    async def get_server_time(self) -> Optional[datetime]:
        """Obtém horário do servidor"""
        result = await self._request("GET", "/api/v2/time")
        
        if result and 'timestamp' in result:
            return datetime.fromtimestamp(result['timestamp'] / 1000)
        
        return None