# core/data_manager.py
"""Gerenciador de dados para buscar, armazenar e processar dados de mercado."""
import asyncio
from datetime import datetime, timedelta, timezone, date # Adicionado date para compatibilidade com get_historical_ticks
from typing import Dict, List, Optional, Any, Union
import pandas as pd
import numpy as np
import talib # Certifique-se que TA-Lib C library esta instalada no Dockerfile

# REMOVER a seguinte linha:
# import aioredis # Mantido -> ESTA LINHA DEVE SER REMOVIDA

from config.settings import CONFIG, DATA_MANAGER_CONFIG as DM_CFG # Esta linha esta correta, mas DATA_MANAGER_CONFIG precisa existir em settings.py
from data.tick_storage import TickStorage
from data.redis_cache import RedisCache # RedisCache ja foi atualizado para redis-py
from utils.logger import setup_logger

logger = setup_logger("data_manager")

class DataManager:
    """
    Gerencia o acesso e armazenamento de dados de ticks, OHLC, indicadores e performance.
    Usa TickStorage para persistencia de longo prazo e RedisCache para acesso rapido.
    """

    def __init__(self):
        self.tick_storage = TickStorage(base_path=CONFIG.PARQUET_PATH if hasattr(CONFIG, 'PARQUET_PATH') else "data/parquet_data")
        self.redis_cache = RedisCache() # RedisCache agora usa redis-py
        self._is_initialized: bool = False
        self._data_lock = asyncio.Lock() # Lock para operacoes criticas de dados

    async def initialize(self):
        """Inicializa o DataManager, conectando ao RedisCache."""
        if self._is_initialized:
            logger.info("DataManager ja inicializado.")
            return
        try:
            await self.redis_cache.connect() # redis_cache.connect() ja e async
            if not self.redis_cache.connected:
                logger.warning("DataManager: Falha ao conectar ao RedisCache na inicializacao.")
            else:
                logger.info("DataManager conectado ao RedisCache com sucesso.")

            logger.info("DataManager inicializado com sucesso.")
            self._is_initialized = True
        except Exception as e:
            logger.exception("Erro critico ao inicializar DataManager:")
            self._is_initialized = False
            raise


    async def store_tick(self, tick_data: Dict[str, Any]):
        if not self._is_initialized:
            logger.warning("DataManager nao inicializado. Nao foi possivel armazenar tick.")
            return
        if not isinstance(tick_data, dict):
            logger.error(f"Formato de tick_data invalido para store_tick: {type(tick_data)}. Esperado dict.")
            return

        ts = tick_data.get('timestamp')
        if isinstance(ts, (int, float)):
            ts_value = ts / 1000.0 if ts > 1e12 else ts
            tick_data['timestamp'] = datetime.fromtimestamp(ts_value, tz=timezone.utc)
        elif isinstance(ts, str):
            try:
                tick_data['timestamp'] = pd.Timestamp(ts, tz='UTC').to_pydatetime(warn=False)
            except Exception as e_ts_parse:
                logger.error(f"Erro ao parsear timestamp string '{ts}': {e_ts_parse}. Usando now().")
                tick_data['timestamp'] = datetime.now(timezone.utc)
        elif isinstance(ts, datetime):
            if ts.tzinfo is None:
                tick_data['timestamp'] = ts.replace(tzinfo=timezone.utc)
            else:
                tick_data['timestamp'] = ts.astimezone(timezone.utc)
        else:
            logger.warning(f"Timestamp ausente ou invalido no tick: {ts}. Usando now().")
            tick_data['timestamp'] = datetime.now(timezone.utc)

        symbol = tick_data.get('symbol', CONFIG.SYMBOL if hasattr(CONFIG, 'SYMBOL') else "EURUSD")
        if not isinstance(symbol, str): symbol = str(CONFIG.SYMBOL if hasattr(CONFIG, 'SYMBOL') else "EURUSD")
        tick_data['symbol'] = symbol

        try:
            tick_data['bid'] = float(tick_data.get('bid', 0.0))
            tick_data['ask'] = float(tick_data.get('ask', 0.0))
            tick_data['bid_volume'] = float(tick_data.get('bid_volume', 0.0))
            tick_data['ask_volume'] = float(tick_data.get('ask_volume', 0.0))
        except (ValueError, TypeError) as e_type:
            logger.error(f"Erro ao converter bid/ask/volume para float: {e_type}. Tick: {tick_data}")
            return

        if 'mid' not in tick_data or not isinstance(tick_data['mid'], (float, np.floating)):
            tick_data['mid'] = (tick_data['bid'] + tick_data['ask']) / 2.0
        if 'spread' not in tick_data or not isinstance(tick_data['spread'], (float, np.floating)):
            tick_data['spread'] = tick_data['ask'] - tick_data['bid']

        async with self._data_lock:
            await self.tick_storage.store_tick(tick_data)
            if self.redis_cache.connected:
                await self.redis_cache.store_tick(symbol, tick_data)
            else:
                logger.debug(f"Redis nao conectado, tick para {symbol} nao armazenado no cache.")


    async def get_recent_ticks(self, symbol: str, count: int = 100,
                               from_cache_only: bool = False) -> List[Dict[str, Any]]: # Retornar List em vez de Optional[List]
        if not self._is_initialized:
            logger.warning("DataManager nao inicializado. Nao foi possivel obter ticks recentes.")
            return []

        if self.redis_cache.connected:
            ticks_from_cache = await self.redis_cache.get_recent_ticks(symbol, count)
            if ticks_from_cache: # Nao precisa checar len > 0, lista vazia ja e falsy no if
                logger.debug(f"Obtidos {len(ticks_from_cache)} ticks recentes para {symbol} do cache Redis.")
                return ticks_from_cache
            elif from_cache_only:
                logger.debug(f"Nenhum tick encontrado no cache para {symbol} e from_cache_only=True.")
                return []

        if from_cache_only:
            return []

        logger.debug(f"Nenhum tick recente para {symbol} no cache Redis ou cache desabilitado. Tentando TickStorage...")
        
        lookback_minutes_cfg = getattr(DM_CFG, 'RECENT_TICKS_LOOKBACK_MINUTES', 60) # Default 60
        lookback_minutes = max(5, count // 60 if count > 60 else 1, lookback_minutes_cfg)

        end_dt = datetime.now(timezone.utc)
        start_dt = end_dt - timedelta(minutes=lookback_minutes)

        df_ticks = await self.tick_storage.read_ticks(symbol, start_dt, end_dt)
        if not df_ticks.empty:
            if 'timestamp' in df_ticks.columns and not pd.api.types.is_datetime64_ns_dtype(df_ticks['timestamp']):
                 df_ticks['timestamp'] = pd.to_datetime(df_ticks['timestamp'], utc=True)
            elif 'timestamp' in df_ticks.columns and df_ticks['timestamp'].dt.tz is None:
                 df_ticks['timestamp'] = df_ticks['timestamp'].dt.tz_localize('UTC')

            if 'timestamp' in df_ticks.columns and not df_ticks.empty and isinstance(df_ticks['timestamp'].iloc[0], pd.Timestamp):
                df_ticks['timestamp'] = df_ticks['timestamp'].apply(lambda x: x.to_pydatetime(warn=False))

            recent_list = df_ticks.tail(count).to_dict('records')
            logger.debug(f"Obtidos {len(recent_list)} ticks recentes para {symbol} do TickStorage.")
            return recent_list

        logger.warning(f"Nenhum tick recente encontrado para {symbol} nem no cache nem no TickStorage.")
        return []


    async def get_historical_ticks(self, symbol: str,
                                 start_date: Union[str, datetime, date],
                                 end_date: Union[str, datetime, date],
                                 columns: Optional[List[str]] = None) -> pd.DataFrame: # Retornar pd.DataFrame (pode ser vazio)
        if not self._is_initialized:
            logger.warning("DataManager nao inicializado. Nao foi possivel obter ticks historicos.")
            return pd.DataFrame()

        def parse_date_to_datetime(d: Union[str, datetime, date], start_of_day: bool) -> datetime:
            if isinstance(d, datetime):
                return d.astimezone(timezone.utc) if d.tzinfo else d.replace(tzinfo=timezone.utc)
            if isinstance(d, date):
                dt_naive = datetime.combine(d, datetime.min.time())
                dt = dt_naive.replace(tzinfo=timezone.utc)
                return dt if start_of_day else dt.replace(hour=23, minute=59, second=59, microsecond=999999)
            if isinstance(d, str):
                try:
                    dt_naive = datetime.strptime(d, '%Y-%m-%d')
                    dt_aware = dt_naive.replace(tzinfo=timezone.utc)
                    return dt_aware if start_of_day else dt_aware.replace(hour=23, minute=59, second=59, microsecond=999999)
                except ValueError:
                    logger.error(f"Formato de data string invalido '{d}'. Use YYYY-MM-DD.")
                    raise
            raise TypeError(f"Tipo de data invalido: {type(d)}")

        try:
            start_dt_utc = parse_date_to_datetime(start_date, start_of_day=True)
            end_dt_utc = parse_date_to_datetime(end_date, start_of_day=False)
        except (TypeError, ValueError):
            return pd.DataFrame()

        df = await self.tick_storage.read_ticks(symbol, start_dt_utc, end_dt_utc, columns=columns)
        if not df.empty:
             logger.info(f"Lidos {len(df)} ticks historicos para {symbol} de {start_dt_utc} a {end_dt_utc}.")
        else:
            logger.info(f"Nenhum tick historico encontrado para {symbol} de {start_dt_utc} a {end_dt_utc}.")
        return df if df is not None else pd.DataFrame()


    async def calculate_ohlc(self, symbol: str, timeframe: str,
                           ticks_df: Optional[pd.DataFrame] = None,
                           num_bars: int = 200) -> Optional[pd.DataFrame]:
        if not self._is_initialized:
            logger.warning("DataManager nao inicializado. Nao foi possivel calcular OHLC.")
            return None

        if ticks_df is None or ticks_df.empty:
            lookback_days_ohlc_cfg = getattr(DM_CFG, 'OHLC_LOOKBACK_DAYS', 10) # Default 10
            end_ohlc = datetime.now(timezone.utc)
            start_ohlc = end_ohlc - timedelta(days=lookback_days_ohlc_cfg)

            ticks_df_ohlc = await self.get_historical_ticks(symbol, start_ohlc, end_ohlc, columns=['timestamp', 'mid', 'bid_volume', 'ask_volume'])
            if ticks_df_ohlc is None or ticks_df_ohlc.empty:
                logger.warning(f"Nao foi possivel obter ticks para calcular OHLC {timeframe} para {symbol}.")
                return None
            source_ticks_df = ticks_df_ohlc
        else:
            source_ticks_df = ticks_df.copy()

        if 'timestamp' not in source_ticks_df.columns or 'mid' not in source_ticks_df.columns:
            logger.error(f"DataFrame de ticks para OHLC nao contem 'timestamp' ou 'mid'. Colunas: {source_ticks_df.columns}")
            return None

        if not isinstance(source_ticks_df.index, pd.DatetimeIndex):
            if pd.api.types.is_datetime64_any_dtype(source_ticks_df['timestamp']):
                 source_ticks_df = source_ticks_df.set_index('timestamp')
            else:
                try:
                    source_ticks_df['timestamp'] = pd.to_datetime(source_ticks_df['timestamp'], utc=True)
                    source_ticks_df = source_ticks_df.set_index('timestamp')
                except Exception as e_idx:
                    logger.error(f"Erro ao definir indice de timestamp para calculo OHLC: {e_idx}")
                    return None

        if source_ticks_df.index.tz is None:
             source_ticks_df = source_ticks_df.tz_localize('UTC')
        elif source_ticks_df.index.tz != timezone.utc:
             source_ticks_df = source_ticks_df.tz_convert('UTC')

        resample_rule = timeframe.upper().replace("MIN", "T")
        if resample_rule == "D": resample_rule = "1D"

        try:
            ohlc_df = source_ticks_df['mid'].resample(resample_rule).ohlc()
            if 'bid_volume' in source_ticks_df.columns and 'ask_volume' in source_ticks_df.columns:
                 source_ticks_df['total_volume'] = source_ticks_df['bid_volume'] + source_ticks_df['ask_volume']
                 ohlc_df['volume'] = source_ticks_df['total_volume'].resample(resample_rule).sum()
            elif 'volume' in source_ticks_df.columns: # Se ja houver uma coluna de volume (ex: de dados OHLC externos)
                 ohlc_df['volume'] = source_ticks_df['volume'].resample(resample_rule).sum()
            else:
                ohlc_df['volume'] = 0.0

            ohlc_df.dropna(subset=['open'], inplace=True)

            if ohlc_df.empty:
                logger.info(f"Nenhuma barra OHLC gerada para {symbol} com timeframe {timeframe}.")
                return None

            return ohlc_df.tail(num_bars)

        except Exception as e:
            logger.exception(f"Erro ao calcular OHLC {timeframe} para {symbol}:")
            return None


    async def calculate_volatility(self, symbol: str, period: int = 20,
                                 ohlc_df: Optional[pd.DataFrame] = None) -> Optional[float]:
        if not self._is_initialized:
            logger.warning("DataManager nao inicializado. Nao foi possivel calcular volatilidade.")
            return None

        if ohlc_df is None or ohlc_df.empty or not all(c in ohlc_df.columns for c in ['high', 'low', 'close']):
            logger.debug(f"OHLC nao fornecido para calculo de volatilidade de {symbol}. Tentando obter dados M1.")
            # Usar um timeframe menor para calculo de ATR, como '1T' (1 minuto) ou '5T'
            # O numero de barras deve ser suficiente para o periodo do ATR + algum buffer
            ohlc_df = await self.calculate_ohlc(symbol, "5T", num_bars=period + 50) # Aumentar buffer
            if ohlc_df is None or ohlc_df.empty or not all(c in ohlc_df.columns for c in ['high', 'low', 'close']):
                 logger.warning(f"Nao foi possivel obter/calcular OHLC para calculo de volatilidade de {symbol}.")
                 return None

        try:
            high_prices = ohlc_df['high'].values.astype(float)
            low_prices = ohlc_df['low'].values.astype(float)
            close_prices = ohlc_df['close'].values.astype(float)

            if len(close_prices) < period: # TA-Lib geralmente precisa de 'period' pontos para o primeiro valor
                logger.warning(f"Dados insuficientes ({len(close_prices)} barras) para calcular volatilidade com periodo {period} para {symbol}.")
                return None

            atr_values = talib.ATR(high_prices, low_prices, close_prices, timeperiod=period)

            # Pegar o ultimo valor nao NaN
            last_valid_atr = atr_values[~np.isnan(atr_values)][-1] if not np.all(np.isnan(atr_values)) else None
            
            if last_valid_atr is not None:
                return float(last_valid_atr) # Garantir que e float Python
            else:
                logger.warning(f"Calculo de ATR retornou NaN ou vazio para {symbol} com periodo {period}.")
                return None
        except Exception as e:
            logger.exception(f"Erro ao calcular volatilidade para {symbol}:")
            return None


    async def load_strategy_params(self, strategy_name: str) -> Optional[Dict[str, Any]]:
        if not self._is_initialized or not self.redis_cache.connected:
            logger.warning(f"DataManager nao inicializado ou Redis nao conectado. Nao foi possivel carregar parametros para {strategy_name}.")
            return None

        try:
            params_data = await self.redis_cache.get_indicator(
                symbol="strategy_params",
                indicator_name=strategy_name,
                timeframe="optimized"
            )
            if params_data and isinstance(params_data, dict):
                logger.info(f"Parametros carregados do Redis para {strategy_name}: {params_data}")
                return params_data
            # get_indicator ja tenta json.loads, entao nao precisa de parse duplo
            logger.info(f"Nenhum parametro otimizado encontrado no Redis para {strategy_name} ou formato invalido: {type(params_data)}")
            return None
        except Exception as e:
            logger.exception(f"Erro ao carregar parametros para {strategy_name} do Redis:")
            return None


    async def save_strategy_params(self, strategy_name: str, params: Dict[str, Any]):
        if not self._is_initialized or not self.redis_cache.connected:
            logger.error(f"DataManager nao inicializado ou Redis nao conectado. Nao foi possivel salvar parametros para {strategy_name}.")
            return

        try:
            ttl_params_days = getattr(DM_CFG, 'STRATEGY_PARAMS_TTL_DAYS', 30) # Default 30 dias
            ttl_seconds = ttl_params_days * 86400

            await self.redis_cache.store_indicator(
                symbol="strategy_params",
                indicator_name=strategy_name,
                value=params,
                timeframe="optimized"
            )
            # store_indicator ja usa um TTL, mas podemos garantir um TTL especifico aqui se necessario.
            # A logica do redis_cache.store_indicator usa self.ttl_config.get('indicator', 300)
            # Se quisermos um TTL diferente para parametros, precisamos ajustar ou chamar expire diretamente.
            if self.redis_cache.redis: # Garantir que o cliente redis existe
                key_params = f"indicator:strategy_params:optimized:{strategy_name}"
                await self.redis_cache.redis.expire(key_params, ttl_seconds)
                logger.info(f"Parametros para {strategy_name} salvos no Redis com TTL de {ttl_params_days} dias.")
            else:
                 logger.warning(f"Nao foi possivel definir TTL customizado para parametros de {strategy_name} pois self.redis_cache.redis e None.")

        except Exception as e:
            logger.exception(f"Erro ao salvar parametros para {strategy_name} no Redis:")


    async def get_strategy_performance(self, strategy_name: str) -> Optional[Dict[str, Any]]: # Removido days, buscar apenas o ultimo salvo
        if not self._is_initialized or not self.redis_cache.connected:
            logger.warning(f"DataManager nao inicializado ou Redis nao conectado. Nao foi possivel obter performance para {strategy_name}.")
            return None
        try:
            cached_performance = await self.redis_cache.get_strategy_performance(strategy_name)
            if cached_performance:
                logger.debug(f"Performance para {strategy_name} obtida do cache Redis.")
                return cached_performance
            logger.debug(f"Performance para {strategy_name} nao encontrada no cache.")
            return None
        except Exception as e:
            logger.exception(f"Erro ao obter performance de {strategy_name} do Redis:")
            return None


    async def get_recent_closed_trades(self, symbol: Optional[str] = None, count: int = 20) -> List[Dict[str, Any]]:
        # Esta funcao e um placeholder e deve ser implementada para buscar de um DB de trades real.
        logger.warning("DataManager.get_recent_closed_trades e um placeholder e retorna dados simulados.")
        simulated_trades = []
        now = datetime.now(timezone.utc)
        base_symbol = symbol or (CONFIG.SYMBOL if hasattr(CONFIG, 'SYMBOL') else "EURUSD")
        for i in range(count):
            pnl = np.random.uniform(-50, 75) * (0.01 if "JPY" not in base_symbol.upper() else 1)
            simulated_trades.append({
                'id': f'sim_trade_{now.timestamp()}_{i}',
                'symbol': base_symbol,
                'close_time': now - timedelta(minutes=i * np.random.randint(5, 60)),
                'pnl': round(pnl, 2),
                'side': 'buy' if np.random.rand() > 0.5 else 'sell',
                'entry_price': round(1.1000 + np.random.uniform(-0.0050, 0.0050), 5),
                'exit_price': round(1.1000 + np.random.uniform(-0.0050, 0.0050), 5),
                'size_lots': round(np.random.uniform(0.01, 0.5), 2)
            })
        return sorted(simulated_trades, key=lambda x: x['close_time'], reverse=True)


    async def close(self):
        """Fecha conexoes e faz flush final de dados."""
        logger.info("Fechando DataManager...")
        if self.tick_storage: # Checar se existe
            await self.tick_storage.close()
        if self.redis_cache:
            await self.redis_cache.disconnect()
        self._is_initialized = False
        logger.info("DataManager fechado.")