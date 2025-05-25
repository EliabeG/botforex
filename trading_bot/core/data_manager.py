# core/data_manager.py
import asyncio
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import pyarrow.parquet as pq
import pyarrow as pa
import aioredis
import aiosqlite
import json
import os
from pathlib import Path

from config.settings import CONFIG
from api.ticktrader_ws import TickData
from utils.logger import setup_logger

logger = setup_logger("data_manager")

class DataManager:
    """Gerenciador central de dados - histórico e tempo real"""
    
    def __init__(self):
        self.redis_client = None
        self.db_path = CONFIG.SQLITE_PATH
        self.parquet_path = CONFIG.PARQUET_PATH
        self.tick_buffer = []
        self.buffer_size = 1000
        self.last_flush = datetime.now()
        self.flush_interval = 60  # segundos
        
        # Cache em memória
        self.price_cache = {}
        self.performance_cache = {}
        self.high_water_marks = {}
        
        # Criar diretórios se não existirem
        Path(self.parquet_path).mkdir(parents=True, exist_ok=True)
        Path(os.path.dirname(self.db_path)).mkdir(parents=True, exist_ok=True)
    
    async def initialize(self):
        """Inicializa conexões e estruturas"""
        try:
            # Conectar Redis
            self.redis_client = await aioredis.create_redis_pool(
                f'redis://{CONFIG.REDIS_HOST}:{CONFIG.REDIS_PORT}/{CONFIG.REDIS_DB}',
                encoding='utf-8'
            )
            
            # Criar tabelas SQLite
            await self._create_tables()
            
            # Carregar dados em cache
            await self._load_cache()
            
            # Iniciar flush periódico
            asyncio.create_task(self._periodic_flush())
            
            logger.info("DataManager inicializado")
            
        except Exception as e:
            logger.error(f"Erro ao inicializar DataManager: {e}")
            raise
    
    async def _create_tables(self):
        """Cria tabelas no SQLite"""
        async with aiosqlite.connect(self.db_path) as db:
            # Tabela de metadados de estratégias
            await db.execute("""
                CREATE TABLE IF NOT EXISTS strategy_metadata (
                    strategy_name TEXT PRIMARY KEY,
                    parameters TEXT,
                    last_optimization TIMESTAMP,
                    performance_metrics TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Tabela de trades
            await db.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id TEXT PRIMARY KEY,
                    strategy_name TEXT,
                    symbol TEXT,
                    side TEXT,
                    entry_price REAL,
                    exit_price REAL,
                    size REAL,
                    pnl REAL,
                    commission REAL,
                    open_time TIMESTAMP,
                    close_time TIMESTAMP,
                    duration INTEGER,
                    metadata TEXT
                )
            """)
            
            # Tabela de eventos de risco
            await db.execute("""
                CREATE TABLE IF NOT EXISTS risk_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT,
                    severity TEXT,
                    description TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    data TEXT
                )
            """)
            
            # Tabela de performance diária
            await db.execute("""
                CREATE TABLE IF NOT EXISTS daily_performance (
                    date DATE PRIMARY KEY,
                    starting_balance REAL,
                    ending_balance REAL,
                    total_trades INTEGER,
                    winning_trades INTEGER,
                    total_pnl REAL,
                    max_drawdown REAL,
                    sharpe_ratio REAL
                )
            """)
            
            await db.commit()
    
    async def store_tick(self, tick: TickData):
        """Armazena tick em buffer e Redis"""
        try:
            # Adicionar ao buffer
            tick_dict = tick.to_dict()
            self.tick_buffer.append(tick_dict)
            
            # Armazenar no Redis (cache de curto prazo)
            key = f"tick:{tick.symbol}:latest"
            await self.redis_client.setex(
                key,
                CONFIG.REDIS_TTL_HOURS * 3600,
                json.dumps(tick_dict, default=str)
            )
            
            # Atualizar lista de ticks recentes
            list_key = f"ticks:{tick.symbol}:recent"
            await self.redis_client.lpush(list_key, json.dumps(tick_dict, default=str))
            await self.redis_client.ltrim(list_key, 0, 10000)  # Manter últimos 10k ticks
            
            # Atualizar cache de preço
            self.price_cache[tick.symbol] = tick.mid
            
            # Verificar se precisa fazer flush
            if (len(self.tick_buffer) >= self.buffer_size or 
                (datetime.now() - self.last_flush).total_seconds() > self.flush_interval):
                await self._flush_ticks()
                
        except Exception as e:
            logger.error(f"Erro ao armazenar tick: {e}")
    
    async def _flush_ticks(self):
        """Persiste ticks do buffer para Parquet"""
        if not self.tick_buffer:
            return
        
        try:
            # Converter para DataFrame
            df = pd.DataFrame(self.tick_buffer)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Particionar por dia
            df['date'] = df['timestamp'].dt.date
            
            # Salvar cada dia em arquivo separado
            for date, group in df.groupby('date'):
                filename = f"{self.parquet_path}/ticks_{date.strftime('%Y%m%d')}.parquet"
                
                # Se arquivo existe, append
                if os.path.exists(filename):
                    existing_df = pd.read_parquet(filename)
                    combined_df = pd.concat([existing_df, group], ignore_index=True)
                    combined_df.to_parquet(filename, compression='snappy')
                else:
                    group.to_parquet(filename, compression='snappy')
            
            logger.info(f"Flush de {len(self.tick_buffer)} ticks para Parquet")
            
            # Limpar buffer
            self.tick_buffer = []
            self.last_flush = datetime.now()
            
        except Exception as e:
            logger.error(f"Erro no flush de ticks: {e}")
    
    async def get_recent_ticks(self, symbol: str, count: int = 1000) -> List[TickData]:
        """Obtém ticks recentes do Redis"""
        try:
            list_key = f"ticks:{symbol}:recent"
            
            # Buscar do Redis
            tick_strings = await self.redis_client.lrange(list_key, 0, count - 1)
            
            ticks = []
            for tick_str in tick_strings:
                tick_dict = json.loads(tick_str)
                # Converter de volta para TickData
                tick = TickData({
                    'Symbol': tick_dict['symbol'],
                    'Timestamp': int(datetime.fromisoformat(tick_dict['timestamp'].replace('Z', '+00:00')).timestamp() * 1000),
                    'BestBid': {'Price': tick_dict['bid'], 'Volume': tick_dict['bid_volume']},
                    'BestAsk': {'Price': tick_dict['ask'], 'Volume': tick_dict['ask_volume']}
                })
                ticks.append(tick)
            
            return list(reversed(ticks))  # Ordem cronológica
            
        except Exception as e:
            logger.error(f"Erro ao obter ticks recentes: {e}")
            return []
    
    async def get_historical_ticks(self, symbol: str, days: int = 30) -> pd.DataFrame:
        """Obtém dados históricos de ticks"""
        try:
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=days)
            
            dfs = []
            current_date = start_date
            
            while current_date <= end_date:
                filename = f"{self.parquet_path}/ticks_{current_date.strftime('%Y%m%d')}.parquet"
                
                if os.path.exists(filename):
                    df = pd.read_parquet(filename)
                    df = df[df['symbol'] == symbol]
                    dfs.append(df)
                
                current_date += timedelta(days=1)
            
            if dfs:
                combined_df = pd.concat(dfs, ignore_index=True)
                combined_df.sort_values('timestamp', inplace=True)
                return combined_df
            
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Erro ao obter dados históricos: {e}")
            return pd.DataFrame()
    
    async def load_strategy_params(self, strategy_name: str) -> Optional[Dict]:
        """Carrega parâmetros otimizados de estratégia"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                cursor = await db.execute(
                    "SELECT parameters FROM strategy_metadata WHERE strategy_name = ?",
                    (strategy_name,)
                )
                row = await cursor.fetchone()
                
                if row:
                    return json.loads(row[0])
                
            return None
            
        except Exception as e:
            logger.error(f"Erro ao carregar parâmetros: {e}")
            return None
    
    async def save_strategy_params(self, strategy_name: str, parameters: Dict):
        """Salva parâmetros otimizados de estratégia"""
        try:
            params_json = json.dumps(parameters)
            
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    INSERT OR REPLACE INTO strategy_metadata 
                    (strategy_name, parameters, last_optimization, updated_at)
                    VALUES (?, ?, ?, ?)
                """, (strategy_name, params_json, datetime.now(), datetime.now()))
                
                await db.commit()
                
            logger.info(f"Parâmetros salvos para {strategy_name}")
            
        except Exception as e:
            logger.error(f"Erro ao salvar parâmetros: {e}")
    
    async def record_trade(self, trade_data: Dict):
        """Registra trade no banco de dados"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    INSERT INTO trades (
                        id, strategy_name, symbol, side, entry_price, 
                        exit_price, size, pnl, commission, open_time, 
                        close_time, duration, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    trade_data.get('id'),
                    trade_data.get('strategy_name'),
                    trade_data.get('symbol'),
                    trade_data.get('side'),
                    trade_data.get('entry_price'),
                    trade_data.get('exit_price'),
                    trade_data.get('size'),
                    trade_data.get('pnl'),
                    trade_data.get('commission', 0),
                    trade_data.get('open_time'),
                    trade_data.get('close_time'),
                    trade_data.get('duration'),
                    json.dumps(trade_data.get('metadata', {}))
                ))
                
                await db.commit()
                
            # Atualizar cache de performance
            await self._update_performance_cache(trade_data['strategy_name'])
            
        except Exception as e:
            logger.error(f"Erro ao registrar trade: {e}")
    
    async def get_strategy_performance(self, strategy_name: str, days: int = 30) -> Dict:
        """Obtém métricas de performance de uma estratégia"""
        try:
            # Verificar cache primeiro
            cache_key = f"{strategy_name}_{days}"
            if cache_key in self.performance_cache:
                cached_data, cache_time = self.performance_cache[cache_key]
                if (datetime.now() - cache_time).total_seconds() < 300:  # Cache de 5 min
                    return cached_data
            
            # Buscar do banco
            start_date = datetime.now() - timedelta(days=days)
            
            async with aiosqlite.connect(self.db_path) as db:
                cursor = await db.execute("""
                    SELECT 
                        COUNT(*) as total_trades,
                        SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
                        SUM(pnl) as total_pnl,
                        AVG(pnl) as avg_pnl,
                        MAX(pnl) as max_win,
                        MIN(pnl) as max_loss,
                        AVG(duration) as avg_duration
                    FROM trades
                    WHERE strategy_name = ? AND open_time > ?
                """, (strategy_name, start_date))
                
                row = await cursor.fetchone()
                
                if row:
                    performance = {
                        'total_trades': row[0] or 0,
                        'winning_trades': row[1] or 0,
                        'total_pnl': row[2] or 0,
                        'avg_pnl': row[3] or 0,
                        'max_win': row[4] or 0,
                        'max_loss': row[5] or 0,
                        'avg_duration': row[6] or 0,
                        'win_rate': (row[1] / row[0]) if row[0] > 0 else 0
                    }
                    
                    # Calcular expectancy
                    if performance['total_trades'] > 0:
                        performance['expectancy'] = performance['total_pnl'] / performance['total_trades']
                    else:
                        performance['expectancy'] = 0
                    
                    # Calcular Sharpe (simplificado)
                    cursor = await db.execute("""
                        SELECT pnl FROM trades
                        WHERE strategy_name = ? AND open_time > ?
                        ORDER BY close_time
                    """, (strategy_name, start_date))
                    
                    pnls = [row[0] for row in await cursor.fetchall()]
                    
                    if len(pnls) > 1:
                        returns = np.array(pnls)
                        sharpe = np.mean(returns) / (np.std(returns) + 1e-6) * np.sqrt(252)
                        performance['sharpe_ratio'] = sharpe
                    else:
                        performance['sharpe_ratio'] = 0
                    
                    # Cachear resultado
                    self.performance_cache[cache_key] = (performance, datetime.now())
                    
                    return performance
            
            return {}
            
        except Exception as e:
            logger.error(f"Erro ao obter performance: {e}")
            return {}
    
    async def calculate_volatility(self, symbol: str, period: int = 20) -> float:
        """Calcula volatilidade recente"""
        try:
            ticks = await self.get_recent_ticks(symbol, period * 10)
            
            if len(ticks) < period:
                return 0.0
            
            prices = [t.mid for t in ticks]
            returns = np.diff(np.log(prices))
            
            # Volatilidade anualizada
            volatility = np.std(returns) * np.sqrt(252 * 24 * 60)  # Minutely data
            
            return volatility
            
        except Exception as e:
            logger.error(f"Erro ao calcular volatilidade: {e}")
            return 0.0
    
    async def get_current_price(self, symbol: str) -> float:
        """Obtém preço atual"""
        # Primeiro verificar cache em memória
        if symbol in self.price_cache:
            return self.price_cache[symbol]
        
        # Depois Redis
        try:
            key = f"tick:{symbol}:latest"
            tick_str = await self.redis_client.get(key)
            
            if tick_str:
                tick_dict = json.loads(tick_str)
                return (tick_dict['bid'] + tick_dict['ask']) / 2
                
        except Exception as e:
            logger.error(f"Erro ao obter preço atual: {e}")
        
        return 0.0
    
    async def get_high_water_mark(self) -> float:
        """Obtém high water mark para cálculo de drawdown"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                cursor = await db.execute("""
                    SELECT MAX(ending_balance) FROM daily_performance
                """)
                row = await cursor.fetchone()
                
                if row and row[0]:
                    return row[0]
                
            return CONFIG.INITIAL_BALANCE if hasattr(CONFIG, 'INITIAL_BALANCE') else 10000
            
        except Exception as e:
            logger.error(f"Erro ao obter high water mark: {e}")
            return 10000
    
    async def save_daily_performance(self, performance_data: Dict):
        """Salva performance diária"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    INSERT OR REPLACE INTO daily_performance (
                        date, starting_balance, ending_balance, total_trades,
                        winning_trades, total_pnl, max_drawdown, sharpe_ratio
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    performance_data['date'],
                    performance_data['starting_balance'],
                    performance_data['ending_balance'],
                    performance_data['total_trades'],
                    performance_data['winning_trades'],
                    performance_data['total_pnl'],
                    performance_data['max_drawdown'],
                    performance_data['sharpe_ratio']
                ))
                
                await db.commit()
                
        except Exception as e:
            logger.error(f"Erro ao salvar performance diária: {e}")
    
    async def _periodic_flush(self):
        """Flush periódico de dados"""
        while True:
            try:
                await asyncio.sleep(self.flush_interval)
                await self._flush_ticks()
                
            except Exception as e:
                logger.error(f"Erro no flush periódico: {e}")
    
    async def _update_performance_cache(self, strategy_name: str):
        """Invalida cache de performance após novo trade"""
        keys_to_remove = [k for k in self.performance_cache.keys() if k.startswith(strategy_name)]
        for key in keys_to_remove:
            del self.performance_cache[key]
    
    async def _load_cache(self):
        """Carrega dados importantes em cache"""
        try:
            # Carregar high water mark
            self.high_water_marks['global'] = await self.get_high_water_mark()
            
            logger.info("Cache carregado")
            
        except Exception as e:
            logger.error(f"Erro ao carregar cache: {e}")
    
    async def save_state(self):
        """Salva estado atual antes de desligar"""
        try:
            # Flush final de ticks
            await self._flush_ticks()
            
            # Fechar conexões
            if self.redis_client:
                self.redis_client.close()
                await self.redis_client.wait_closed()
            
            logger.info("Estado salvo com sucesso")
            
        except Exception as e:
            logger.error(f"Erro ao salvar estado: {e}")
    
    async def download_historical_data(self, symbol: str, start_date: datetime, end_date: datetime):
        """Baixa dados históricos via REST API (implementar com TickTrader API)"""
        # TODO: Implementar download via TickTrader REST API
        logger.info(f"Download de dados históricos: {symbol} de {start_date} até {end_date}")
        pass