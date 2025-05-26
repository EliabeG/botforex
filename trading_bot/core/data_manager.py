# core/data_manager.py
import asyncio
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union # Adicionado Union
from datetime import datetime, timedelta, timezone # Adicionado timezone
import pyarrow.parquet as pq
import pyarrow as pa
import aioredis # Mantido
import aiosqlite
import json
import os
from pathlib import Path

from config.settings import CONFIG
from api.ticktrader_ws import TickData # Importação correta
from utils.logger import setup_logger

logger = setup_logger("data_manager")

class DataManager:
    """Gerenciador central de dados - histórico e tempo real"""

    def __init__(self):
        self.redis_client: Optional[aioredis.Redis] = None # Tipagem mais específica
        self.db_path: str = CONFIG.SQLITE_PATH
        self.parquet_path: str = CONFIG.PARQUET_PATH
        self.tick_buffer: List[Dict[str, Any]] = [] # Tipagem para o buffer
        self.buffer_size: int = 1000 # Pode ser configurável via CONFIG
        self.last_flush: datetime = datetime.now(timezone.utc) # Usar UTC
        self.flush_interval_seconds: int = 60  # Renomeado para clareza e pode ser de CONFIG

        # Cache em memória
        self.price_cache: Dict[str, float] = {}
        self.performance_cache: Dict[str, Tuple[Dict[str, Any], datetime]] = {} # Cache com timestamp
        self.high_water_marks: Dict[str, float] = {} # Pode ser útil para HWM por estratégia ou global

        # Garantir que os diretórios de base existam (já feito em settings.py ao carregar CONFIG)
        # Path(self.parquet_path).mkdir(parents=True, exist_ok=True)
        # Path(os.path.dirname(self.db_path)).mkdir(parents=True, exist_ok=True)
        self._flush_task: Optional[asyncio.Task] = None # Para a tarefa de flush periódico

    async def initialize(self):
        """Inicializa conexões e estruturas"""
        try:
            logger.info(f"Inicializando DataManager com db_path: {self.db_path} e parquet_path: {self.parquet_path}")
            # Conectar Redis
            self.redis_client = await aioredis.from_url( # Método recomendado para aioredis > 2.0
                f'redis://{CONFIG.REDIS_HOST}:{CONFIG.REDIS_PORT}/{CONFIG.REDIS_DB}',
                encoding='utf-8',
                # minsize=5, # minsize/maxsize são para create_redis_pool, from_url usa um pool interno
                # maxsize=10
            )
            await self.redis_client.ping() # Verificar conexão
            logger.info("Conexão Redis estabelecida com sucesso.")


            # Criar tabelas SQLite
            await self._create_tables()

            # Carregar dados em cache
            await self._load_cache()

            # Iniciar flush periódico
            if self._flush_task is None or self._flush_task.done():
                self._flush_task = asyncio.create_task(self._periodic_flush())

            logger.info("DataManager inicializado")

        except Exception as e:
            logger.exception("Erro ao inicializar DataManager:")
            # Se a conexão com Redis falhar, o bot pode não funcionar corretamente.
            # Considerar se deve relançar o erro ou tentar reconectar.
            if self.redis_client: # Tentar fechar se parcialmente conectado
                await self.redis_client.close()
            raise

    async def _create_tables(self):
        """Cria tabelas no SQLite"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                # Tabela de metadados de estratégias
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS strategy_metadata (
                        strategy_name TEXT PRIMARY KEY,
                        parameters TEXT,
                        last_optimization TIMESTAMP,
                        performance_metrics TEXT, /* Considerar colunas separadas para métricas chave */
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
                        pnl_pips REAL, /* Adicionado */
                        commission REAL,
                        open_time TIMESTAMP,
                        close_time TIMESTAMP,
                        duration_seconds INTEGER, /* Renomeado de duration */
                        exit_reason TEXT, /* Adicionado */
                        metadata TEXT
                    )
                """)

                # Tabela de eventos de risco
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS risk_events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        event_type TEXT NOT NULL, /* Adicionado NOT NULL */
                        severity TEXT NOT NULL,   /* Adicionado NOT NULL */
                        description TEXT,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        data TEXT /* JSON com detalhes do evento */
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
                        max_drawdown_pct REAL, /* Renomeado para clareza _pct */
                        sharpe_ratio REAL
                    )
                """)

                await db.commit()
                logger.info("Tabelas SQLite verificadas/criadas com sucesso.")
        except Exception as e:
            logger.exception("Erro ao criar tabelas SQLite:")
            raise


    async def store_tick(self, tick: TickData):
        """Armazena tick em buffer e Redis"""
        if not self.redis_client:
            logger.warning("Cliente Redis não inicializado. Tick não será armazenado no Redis.")
            # Ainda pode adicionar ao buffer para Parquet, dependendo da política
            # return # Descomente se Redis for mandatório para continuar

        try:
            tick_dict = tick.to_dict() # .to_dict() agora retorna timestamp como string ISO
            self.tick_buffer.append(tick_dict)

            if self.redis_client:
                # Armazenar no Redis (cache de curto prazo)
                key_latest = f"tick:{tick.symbol}:latest"
                # json.dumps com default=str não é mais necessário se to_dict já formata bem
                await self.redis_client.setex(
                    key_latest,
                    CONFIG.REDIS_TTL_HOURS * 3600,
                    json.dumps(tick_dict)
                )

                list_key = f"ticks:{tick.symbol}:recent"
                await self.redis_client.lpush(list_key, json.dumps(tick_dict))
                await self.redis_client.ltrim(list_key, 0, 9999) # Ajustado de 10000 para 0-9999 (10k itens)
                await self.redis_client.expire(list_key, CONFIG.REDIS_TTL_HOURS * 3600) # Definir TTL na lista também

            self.price_cache[tick.symbol] = tick.mid # Atualizar cache de preço em memória

            # Verificar se precisa fazer flush (condição do original)
            if (len(self.tick_buffer) >= self.buffer_size or
                (datetime.now(timezone.utc) - self.last_flush).total_seconds() > self.flush_interval_seconds):
                await self._flush_ticks()

        except Exception as e:
            logger.exception(f"Erro ao armazenar tick para {tick.symbol if tick else 'N/A'}:")


    async def _flush_ticks(self):
        """Persiste ticks do buffer para Parquet"""
        if not self.tick_buffer:
            return

        ticks_to_write = self.tick_buffer.copy() # Copiar antes de limpar
        self.tick_buffer = [] # Limpar buffer original imediatamente
        self.last_flush = datetime.now(timezone.utc)

        try:
            df = pd.DataFrame(ticks_to_write)
            if df.empty:
                logger.info("Buffer de ticks para flush estava vazio após cópia ou conversão.")
                return

            # Garantir que timestamp seja datetime64[ns, UTC] para Parquet
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)

            # Particionar por símbolo e data
            # df['date'] = df['timestamp'].dt.date # dt.date não é timezone-aware, usar dt.normalize()
            df['date_partition'] = df['timestamp'].dt.normalize() # Cria uma data sem hora, mas TZ-aware

            # Schema para Parquet (opcional, mas bom para consistência)
            # Se o schema for definido em tick_storage.py, pode ser referenciado ou duplicado aqui.
            # Por enquanto, confiando na inferência do Pandas ou no schema implícito.

            for (symbol_partition, date_partition), group_df in df.groupby(['symbol', 'date_partition']):
                if group_df.empty:
                    continue

                # Construir caminho do arquivo
                # Ex: parquet_path/EURUSD/2025/05/ticks_EURUSD_20250525.parquet
                year = date_partition.year
                month = f"{date_partition.month:02d}"
                day_str = date_partition.strftime('%Y%m%d')

                symbol_path = Path(self.parquet_path) / str(symbol_partition) / str(year) / str(month)
                symbol_path.mkdir(parents=True, exist_ok=True)
                filename = symbol_path / f"ticks_{symbol_partition}_{day_str}.parquet"

                # Remover colunas de partição antes de salvar
                group_to_save = group_df.drop(columns=['date_partition'])


                if filename.exists():
                    try:
                        existing_table = pq.read_table(filename)
                        new_table = pa.Table.from_pandas(group_to_save, preserve_index=False)
                        combined_table = pa.concat_tables([existing_table, new_table])
                        # Remover duplicatas (mais complexo em PyArrow diretamente, mais fácil com Pandas)
                        # Por simplicidade, vamos re-ler, dedup, e salvar (como no original, mas com PyArrow)
                        combined_df_pd = combined_table.to_pandas()
                        combined_df_pd.drop_duplicates(subset=['timestamp'], keep='last', inplace=True) # Assumindo timestamp é único
                        combined_df_pd.sort_values('timestamp', inplace=True)
                        final_table_to_save = pa.Table.from_pandas(combined_df_pd, preserve_index=False)
                        pq.write_table(final_table_to_save, filename, compression='snappy')
                    except Exception as e:
                        logger.error(f"Erro ao fazer append ao arquivo Parquet {filename}: {e}. Tentando sobrescrever.")
                        # Fallback para sobrescrever se o append falhar (pode perder dados se o erro for na leitura)
                        # Ou, melhor, salvar o novo grupo em um arquivo temporário e lidar com a união depois.
                        # Para simplificar a correção, vamos apenas logar e continuar, o que pode levar a dados não salvos.
                        # A lógica original de sobrescrita com concat do pandas é mais robusta para dedup.
                        # Revertendo para a lógica de Pandas para append/dedup por simplicidade e robustez aqui:
                        existing_df = pd.read_parquet(filename)
                        combined_df = pd.concat([existing_df, group_to_save], ignore_index=True)
                        combined_df.drop_duplicates(subset=['timestamp'], keep='last', inplace=True)
                        combined_df.sort_values('timestamp', inplace=True)
                        combined_df.to_parquet(filename, compression='snappy', index=False)

                else:
                    group_to_save.to_parquet(filename, compression='snappy', index=False)

            logger.info(f"Flush de {len(ticks_to_write)} ticks para Parquet concluído.")

        except Exception as e:
            logger.exception("Erro no flush de ticks:")
            # Em caso de erro, readicionar os ticks ao buffer para tentar novamente mais tarde
            self.tick_buffer.extend(ticks_to_write)
            logger.warning(f"{len(ticks_to_write)} ticks retornados ao buffer devido a erro de flush.")


    async def get_recent_ticks(self, symbol: str, count: int = 1000) -> List[TickData]:
        """Obtém ticks recentes do Redis"""
        if not self.redis_client:
            logger.warning("Cliente Redis não disponível para get_recent_ticks.")
            # Fallback para ler do buffer de memória se necessário, ou retornar vazio.
            # Pegando do buffer em memória como um fallback simples:
            if self.tick_buffer:
                relevant_ticks = [td for td in self.tick_buffer if td.get('symbol') == symbol]
                return [TickData.from_dict(td) for td in relevant_ticks[-count:]][::-1] # Inverter para cronológico
            return []


        try:
            list_key = f"ticks:{symbol}:recent"
            tick_strings = await self.redis_client.lrange(list_key, 0, count - 1)

            ticks = []
            for tick_str_bytes in tick_strings: # lrange retorna bytes se encoding não for utf-8 no cliente
                try:
                    # A decodificação já deve ser feita pelo aioredis se encoding='utf-8' foi usado na conexão.
                    # Se ainda vier como bytes, decodifique:
                    # tick_str = tick_str_bytes.decode('utf-8') if isinstance(tick_str_bytes, bytes) else tick_str_bytes
                    tick_dict = json.loads(tick_str_bytes)
                    ticks.append(TickData.from_dict(tick_dict)) # Usar from_dict
                except json.JSONDecodeError as jde:
                    logger.warning(f"Erro ao decodificar JSON do tick do Redis: {jde}. Dado: {tick_str_bytes[:100]}")
                except Exception as ex:
                    logger.warning(f"Erro ao converter tick do Redis para TickData: {ex}. Dado: {tick_str_bytes[:100]}")


            return list(reversed(ticks))

        except Exception as e:
            logger.exception(f"Erro ao obter ticks recentes de {symbol} do Redis:")
            return []


    async def get_historical_ticks(self, symbol: str, days: int = 30) -> pd.DataFrame:
        """Obtém dados históricos de ticks de arquivos Parquet"""
        try:
            end_date_obj = datetime.now(timezone.utc).date() # Usar timezone.utc
            start_date_obj = end_date_obj - timedelta(days=days -1) # -1 para incluir o dia atual na contagem de 'days'

            dfs = []
            current_scan_date = start_date_obj # Renomeado

            while current_scan_date <= end_date_obj:
                year = current_scan_date.year
                month = f"{current_scan_date.month:02d}"
                day_str = current_scan_date.strftime('%Y%m%d')
                symbol_path = Path(self.parquet_path) / symbol / str(year) / str(month)
                filename = symbol_path / f"ticks_{symbol}_{day_str}.parquet"


                if filename.exists():
                    try:
                        df = pd.read_parquet(filename)
                        # Filtrar pelo símbolo novamente se o arquivo contiver múltiplos (não deveria pela estrutura)
                        df_symbol_specific = df[df['symbol'] == symbol]
                        if not df_symbol_specific.empty:
                             dfs.append(df_symbol_specific)
                    except Exception as e_read:
                        logger.error(f"Erro ao ler arquivo Parquet {filename}: {e_read}")

                current_scan_date += timedelta(days=1)

            if dfs:
                combined_df = pd.concat(dfs, ignore_index=True)
                if 'timestamp' in combined_df.columns:
                    combined_df['timestamp'] = pd.to_datetime(combined_df['timestamp'], utc=True)
                    combined_df.sort_values('timestamp', inplace=True)
                    # Filtrar novamente pelo range exato se os arquivos diários puderem conter dados fora do dia
                    start_datetime_exact = datetime.combine(start_date_obj, datetime.min.time(), tzinfo=timezone.utc)
                    end_datetime_exact = datetime.combine(end_date_obj, datetime.max.time(), tzinfo=timezone.utc)
                    combined_df = combined_df[
                        (combined_df['timestamp'] >= start_datetime_exact) &
                        (combined_df['timestamp'] <= end_datetime_exact)
                    ]
                    return combined_df
                else:
                    logger.warning("DataFrame histórico combinado não possui coluna 'timestamp'.")
                    return pd.DataFrame()


            logger.info(f"Nenhum dado histórico encontrado para {symbol} nos últimos {days} dias.")
            return pd.DataFrame()

        except Exception as e:
            logger.exception(f"Erro ao obter dados históricos de ticks para {symbol}:")
            return pd.DataFrame()


    async def load_strategy_params(self, strategy_name: str) -> Optional[Dict[str, Any]]:
        """Carrega parâmetros otimizados de estratégia do SQLite"""
        if not Path(self.db_path).exists():
            logger.warning(f"Arquivo de banco de dados {self.db_path} não encontrado. Não é possível carregar parâmetros.")
            return None
        try:
            async with aiosqlite.connect(self.db_path) as db:
                async with db.execute( # Usar 'async with' para o cursor
                    "SELECT parameters FROM strategy_metadata WHERE strategy_name = ?",
                    (strategy_name,)
                ) as cursor:
                    row = await cursor.fetchone()

                if row and row[0]:
                    try:
                        return json.loads(row[0])
                    except json.JSONDecodeError as jde:
                        logger.error(f"Erro ao decodificar JSON dos parâmetros para {strategy_name}: {jde}")
                        return None
            return None # Se strategy_name não for encontrado

        except Exception as e:
            logger.exception(f"Erro ao carregar parâmetros para {strategy_name} do SQLite:")
            return None


    async def save_strategy_params(self, strategy_name: str, parameters: Dict[str, Any]):
        """Salva parâmetros otimizados de estratégia no SQLite"""
        try:
            params_json = json.dumps(parameters)
            now_utc = datetime.now(timezone.utc)

            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    INSERT INTO strategy_metadata (strategy_name, parameters, last_optimization, updated_at, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(strategy_name) DO UPDATE SET
                        parameters = excluded.parameters,
                        last_optimization = excluded.last_optimization,
                        updated_at = excluded.updated_at
                """, (strategy_name, params_json, now_utc, now_utc, now_utc)) # Adicionado created_at para INSERT

                await db.commit()
            logger.info(f"Parâmetros salvos para {strategy_name}")

        except Exception as e:
            logger.exception(f"Erro ao salvar parâmetros para {strategy_name} no SQLite:")


    async def record_trade(self, trade_data: Dict[str, Any]): # Adicionada tipagem
        """Registra trade no banco de dados"""
        # Garantir que todos os campos esperados existam ou tenham defaults
        required_fields = {
            'id': None, 'strategy_name': 'UnknownStrategy', 'symbol': CONFIG.SYMBOL,
            'side': 'N/A', 'entry_price': 0.0, 'exit_price': 0.0, 'size': 0.0,
            'pnl': 0.0, 'pnl_pips': 0.0, 'commission': 0.0,
            'open_time': datetime.now(timezone.utc), 'close_time': datetime.now(timezone.utc),
            'duration_seconds': 0, 'exit_reason': 'N/A', 'metadata': {}
        }
        for field, default_value in required_fields.items():
            trade_data.setdefault(field, default_value)

        # Converter datetimes para string ISO para SQLite, ou garantir que sejam objetos datetime
        for dt_field in ['open_time', 'close_time']:
            if isinstance(trade_data[dt_field], datetime):
                 trade_data[dt_field] = trade_data[dt_field].isoformat()


        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    INSERT INTO trades (
                        id, strategy_name, symbol, side, entry_price,
                        exit_price, size, pnl, pnl_pips, commission, open_time,
                        close_time, duration_seconds, exit_reason, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    str(trade_data['id']), # Garantir que ID seja string
                    trade_data['strategy_name'],
                    trade_data['symbol'],
                    trade_data['side'],
                    trade_data['entry_price'],
                    trade_data['exit_price'],
                    trade_data['size'],
                    trade_data['pnl'],
                    trade_data.get('pnl_pips', 0.0), # Adicionado pnl_pips
                    trade_data['commission'],
                    trade_data['open_time'],
                    trade_data['close_time'],
                    int(trade_data['duration_seconds']), # Garantir que seja int
                    trade_data.get('exit_reason', 'N/A'), # Adicionado exit_reason
                    json.dumps(trade_data['metadata'])
                ))
                await db.commit()

            # Atualizar cache de performance
            if trade_data['strategy_name'] != 'UnknownStrategy':
                await self._update_performance_cache(trade_data['strategy_name'])
            logger.info(f"Trade {trade_data['id']} registrado para estratégia {trade_data['strategy_name']}.")


        except Exception as e:
            logger.exception(f"Erro ao registrar trade {trade_data.get('id', 'N/A')} no SQLite:")


    async def get_strategy_performance(self, strategy_name: str, days: int = 30) -> Dict[str, Any]:
        """Obtém métricas de performance de uma estratégia do SQLite"""
        if not Path(self.db_path).exists():
            logger.warning(f"Arquivo de banco de dados {self.db_path} não encontrado. Retornando performance vazia.")
            return {}

        try:
            cache_key = f"perf:{strategy_name}:{days}d" # Melhor formato de chave
            if self.redis_client: # Tentar obter do Redis primeiro
                cached_perf_json = await self.redis_client.get(cache_key)
                if cached_perf_json:
                    try:
                        logger.debug(f"Performance para {strategy_name} ({days}d) encontrada no cache Redis.")
                        return json.loads(cached_perf_json)
                    except json.JSONDecodeError:
                        logger.warning(f"Erro ao decodificar performance do Redis para {strategy_name}.")


            # Se não estiver no Redis, verificar cache em memória (embora o original não use para isso)
            # Ou calcular do banco e depois salvar no Redis

            start_date_dt = datetime.now(timezone.utc) - timedelta(days=days)
            # SQLite armazena timestamps como strings ISO, então formatar para comparação
            start_date_iso = start_date_dt.isoformat()


            async with aiosqlite.connect(self.db_path) as db:
                async with db.execute("""
                    SELECT
                        COUNT(*) as total_trades,
                        SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
                        SUM(pnl) as total_pnl,
                        AVG(pnl) as avg_pnl,
                        MAX(CASE WHEN pnl > 0 THEN pnl ELSE NULL END) as max_win, /* Ignorar perdas para max_win */
                        MIN(CASE WHEN pnl < 0 THEN pnl ELSE NULL END) as max_loss_val, /* Ignorar ganhos para max_loss_val */
                        AVG(duration_seconds) as avg_duration_seconds
                    FROM trades
                    WHERE strategy_name = ? AND open_time >= ?
                """, (strategy_name, start_date_iso)) as cursor: # Usar ISO format
                    row = await cursor.fetchone()

                performance = {}
                if row and row[0] is not None and row[0] > 0: # Se total_trades > 0
                    performance = {
                        'total_trades': row[0],
                        'winning_trades': row[1] or 0,
                        'total_pnl': row[2] or 0.0,
                        'avg_pnl': row[3] or 0.0,
                        'max_win': row[4] or 0.0,
                        'max_loss': abs(row[5] or 0.0), # max_loss é geralmente positivo
                        'avg_duration_seconds': row[6] or 0.0,
                        'win_rate': (row[1] / row[0]) if row[0] > 0 else 0.0,
                        'expectancy': (row[2] / row[0]) if row[0] > 0 else 0.0
                    }

                    # Calcular Sharpe (simplificado com PnLs)
                    async with db.execute("""
                        SELECT pnl FROM trades
                        WHERE strategy_name = ? AND open_time >= ?
                        ORDER BY close_time
                    """, (strategy_name, start_date_iso)) as pnl_cursor:
                        pnls = [r[0] for r in await pnl_cursor.fetchall() if r[0] is not None]

                    if len(pnls) > 1:
                        returns_pct = np.array(pnls) / CONFIG.INITIAL_BALANCE # Assumindo um capital inicial para % de retorno
                        # Ou, se pnl já é % do risco, usar pnl diretamente
                        # std_dev_returns = np.std(returns_pct)
                        # performance['sharpe_ratio'] = (np.mean(returns_pct) / std_dev_returns if std_dev_returns > 0 else 0) * np.sqrt(252) # Exemplo de anualização diária
                        # Para simplificar, e como Sharpe é mais complexo sem retornos diários fixos:
                        std_dev_pnl = np.std(pnls)
                        performance['sharpe_ratio'] = (np.mean(pnls) / std_dev_pnl if std_dev_pnl > 0 else 0) # Sharpe não anualizado sobre PnL
                    else:
                        performance['sharpe_ratio'] = 0.0
                else: # Nenhum trade encontrado
                    performance = {
                        'total_trades': 0, 'winning_trades': 0, 'total_pnl': 0.0, 'avg_pnl': 0.0,
                        'max_win': 0.0, 'max_loss': 0.0, 'avg_duration_seconds': 0.0,
                        'win_rate': 0.0, 'expectancy': 0.0, 'sharpe_ratio': 0.0
                    }


            if self.redis_client and performance: # Cachear resultado no Redis
                await self.redis_client.setex(
                    cache_key,
                    self.performance_cache.get(cache_key, (None, datetime.now(timezone.utc) - timedelta(days=1)))[1] < datetime.now(timezone.utc) - timedelta(minutes=5) and 300 or 60, # TTL menor se já existia
                    json.dumps(performance)
                )
                self.performance_cache[cache_key] = (performance, datetime.now(timezone.utc)) # Atualizar cache em memória


            return performance

        except Exception as e:
            logger.exception(f"Erro ao obter performance para {strategy_name} do SQLite:")
            return {}


    async def calculate_volatility(self, symbol: str, period: int = 20, timeframe_minutes: int = 1) -> float: # Adicionado timeframe_minutes
        """Calcula volatilidade recente (ex: ATR ou std dev de retornos)."""
        try:
            # Tentar obter ~period*2 ticks para ter dados suficientes para cálculo de 'period' retornos.
            # Se timeframe_minutes=1, queremos 20 minutos de dados. Se ticks são a cada segundo, são 20*60 ticks.
            # O 'count' para get_recent_ticks deve ser ajustado com base na frequência esperada de ticks.
            # Exemplo: para volatilidade sobre retornos de 1 minuto, com 20 períodos.
            num_ticks_to_fetch = period * timeframe_minutes * 60 # Estimativa grosseira
            ticks = await self.get_recent_ticks(symbol, count=num_ticks_to_fetch) # Aumentar contagem para ter dados suficientes

            if len(ticks) < period + 1: # Precisa de pelo menos period+1 ticks para 'period' retornos
                logger.warning(f"Dados insuficientes ({len(ticks)} ticks) para calcular volatilidade de {period} períodos para {symbol}.")
                return 0.01 # Retornar um valor default não-zero para evitar divisão por zero

            prices = [t.mid for t in ticks if t.mid > 0] # Filtrar preços inválidos
            if len(prices) < period + 1:
                 logger.warning(f"Dados de preço válidos insuficientes ({len(prices)}) para volatilidade de {symbol}.")
                 return 0.01

            # Calcular retornos logarítmicos
            log_returns = np.diff(np.log(prices))

            if len(log_returns) < period:
                logger.warning(f"Retornos insuficientes ({len(log_returns)}) para volatilidade de {symbol}.")
                return 0.01

            # Volatilidade como desvio padrão dos retornos
            # A anualização/escalonamento depende do que esta volatilidade representa.
            # Se for volatilidade de curto prazo para, por exemplo, stops ATR, pode não precisar de anualização.
            # Se for volatilidade de retornos de X minutos, escalar de acordo.
            # Ex: para retornos de 1 minuto, anualizado: std(log_returns_1min) * sqrt(252*24*60)
            # Se for para stops, um ATR pode ser mais direto.
            # Aqui, vamos calcular o desvio padrão dos últimos 'period' retornos.
            std_dev_recent_returns = np.std(log_returns[-period:])

            # Se esta volatilidade é usada para algo como % do preço:
            # current_price = prices[-1]
            # volatility_pct = (std_dev_recent_returns / current_price) if current_price > 0 else std_dev_recent_returns
            # return volatility_pct
            return std_dev_recent_returns # Retorna o desvio padrão dos retornos log


        except Exception as e:
            logger.exception(f"Erro ao calcular volatilidade para {symbol}:")
            return 0.01 # Retornar um valor default


    async def get_current_price(self, symbol: str) -> float:
        """Obtém preço atual do cache em memória, depois Redis."""
        if symbol in self.price_cache:
            # Adicionar um check de "staleness" para o cache em memória
            # Se o cache for muito antigo, buscar do Redis.
            # Isso requer armazenar o timestamp da última atualização do price_cache.
            return self.price_cache[symbol]

        if not self.redis_client:
            logger.warning("Cliente Redis não disponível para get_current_price.")
            return 0.0 # Ou levantar uma exceção


        try:
            key = f"tick:{symbol}:latest"
            tick_str_bytes = await self.redis_client.get(key)

            if tick_str_bytes:
                # tick_str = tick_str_bytes.decode('utf-8') if isinstance(tick_str_bytes, bytes) else tick_str_bytes
                tick_dict = json.loads(tick_str_bytes)
                current_mid_price = (tick_dict['bid'] + tick_dict['ask']) / 2.0
                self.price_cache[symbol] = current_mid_price # Atualizar cache em memória
                return current_mid_price

        except Exception as e:
            logger.exception(f"Erro ao obter preço atual de {symbol} do Redis:")

        logger.warning(f"Preço atual para {symbol} não encontrado no cache.")
        return 0.0 # Ou o último preço conhecido de outra fonte


    async def get_high_water_mark(self, account_id: str = "global") -> float:
        """Obtém high water mark para cálculo de drawdown (pode ser por conta/global)."""
        # Tentar obter do cache em memória primeiro
        if account_id in self.high_water_marks:
            return self.high_water_marks[account_id]

        # Se não estiver no cache em memória, buscar do banco (performance diária)
        # ou calcular com base no saldo inicial se não houver histórico.
        if not Path(self.db_path).exists():
            logger.warning(f"Arquivo de banco de dados {self.db_path} não encontrado. Usando saldo inicial para HWM.")
            hwm = CONFIG.INITIAL_BALANCE if hasattr(CONFIG, 'INITIAL_BALANCE') else 10000.0
            self.high_water_marks[account_id] = hwm
            return hwm

        try:
            async with aiosqlite.connect(self.db_path) as db:
                # Pegar o maior ending_balance da tabela daily_performance
                async with db.execute("SELECT MAX(ending_balance) FROM daily_performance") as cursor:
                    row = await cursor.fetchone()

                if row and row[0] is not None:
                    hwm = float(row[0])
                else:
                    # Se não houver registros, usar o saldo inicial configurado
                    hwm = CONFIG.INITIAL_BALANCE if hasattr(CONFIG, 'INITIAL_BALANCE') else 10000.0

                self.high_water_marks[account_id] = hwm # Cachear
                return hwm

        except Exception as e:
            logger.exception(f"Erro ao obter high water mark do SQLite para {account_id}:")
            # Fallback para saldo inicial
            hwm = CONFIG.INITIAL_BALANCE if hasattr(CONFIG, 'INITIAL_BALANCE') else 10000.0
            self.high_water_marks[account_id] = hwm
            return hwm


    async def save_daily_performance(self, performance_data: Dict[str, Any]): # Tipagem
        """Salva performance diária no SQLite"""
        required_fields = [
            'date', 'starting_balance', 'ending_balance', 'total_trades',
            'winning_trades', 'total_pnl', 'max_drawdown_pct', 'sharpe_ratio'
        ]
        if not all(field in performance_data for field in required_fields):
            logger.error(f"Dados de performance diária incompletos: {performance_data}. Campos faltantes: {[f for f in required_fields if f not in performance_data]}")
            return

        try:
            # Garantir que a data seja no formato YYYY-MM-DD
            date_to_save = performance_data['date']
            if isinstance(date_to_save, datetime):
                date_to_save_str = date_to_save.strftime('%Y-%m-%d')
            elif isinstance(date_to_save, str):
                # Tentar parsear para validar e reformatar
                date_to_save_str = datetime.strptime(date_to_save, '%Y-%m-%d').strftime('%Y-%m-%d')
            else: # date object
                date_to_save_str = date_to_save.strftime('%Y-%m-%d')


            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    INSERT OR REPLACE INTO daily_performance (
                        date, starting_balance, ending_balance, total_trades,
                        winning_trades, total_pnl, max_drawdown_pct, sharpe_ratio
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    date_to_save_str,
                    performance_data['starting_balance'],
                    performance_data['ending_balance'],
                    performance_data['total_trades'],
                    performance_data['winning_trades'],
                    performance_data['total_pnl'],
                    performance_data['max_drawdown_pct'],
                    performance_data['sharpe_ratio']
                ))
                await db.commit()
            logger.info(f"Performance diária salva para {date_to_save_str}.")

        except Exception as e:
            logger.exception(f"Erro ao salvar performance diária para data {performance_data.get('date')}:")


    async def _periodic_flush(self):
        """Flush periódico de dados do buffer de ticks para Parquet."""
        while True: # O loop deve ser controlado externamente (ex: cancelamento da task)
            try:
                await asyncio.sleep(self.flush_interval_seconds)
                if self.tick_buffer: # Apenas fazer flush se houver algo no buffer
                    logger.debug(f"Iniciando flush periódico de {len(self.tick_buffer)} ticks.")
                    await self._flush_ticks()
                else:
                    logger.debug("Flush periódico: buffer de ticks vazio.")
            except asyncio.CancelledError:
                logger.info("Tarefa de flush periódico cancelada.")
                await self._flush_ticks() # Tentar um último flush antes de sair
                break
            except Exception as e:
                logger.exception("Erro no flush periódico de ticks:")
                # Continuar tentando após um erro, com um backoff talvez
                await asyncio.sleep(self.flush_interval_seconds * 2) # Esperar mais em caso de erro


    async def _update_performance_cache(self, strategy_name: str):
        """Invalida cache de performance (Redis e memória) após novo trade"""
        if self.redis_client:
            # Invalidar chaves do Redis que contenham o nome da estratégia
            # Isso é um pouco genérico, pode precisar de um padrão mais específico.
            pattern = f"perf:{strategy_name}:*"
            async for key in self.redis_client.scan_iter(match=pattern):
                await self.redis_client.delete(key)
            logger.debug(f"Cache Redis de performance invalidado para padrão: {pattern}")


        # Invalidar cache em memória
        keys_to_remove_mem = [k for k in self.performance_cache if k.startswith(f"perf:{strategy_name}:")]
        for key_mem in keys_to_remove_mem:
            del self.performance_cache[key_mem]
        logger.debug(f"Cache de performance em memória invalidado para {strategy_name}.")


    async def _load_cache(self):
        """Carrega dados importantes em cache na inicialização."""
        try:
            # Carregar high water mark global
            self.high_water_marks['global'] = await self.get_high_water_mark("global")
            # Poderia carregar outras coisas, como últimos N estados de mercado, etc.
            logger.info(f"Cache inicial carregado. HWM Global: {self.high_water_marks['global']:.2f}")

        except Exception as e:
            logger.exception("Erro ao carregar cache inicial do DataManager:")


    async def save_state(self):
        """Salva estado atual antes de desligar (ex: flush final)."""
        logger.info("Salvando estado do DataManager...")
        try:
            # Flush final de ticks
            if self.tick_buffer:
                logger.info(f"Realizando flush final de {len(self.tick_buffer)} ticks.")
                await self._flush_ticks()

            # Fechar conexões
            if self.redis_client:
                await self.redis_client.close()
                # await self.redis_client.wait_closed() # wait_closed não é mais necessário com from_url e close()
                self.redis_client = None # Marcar como fechado
                logger.info("Conexão Redis fechada.")

            if self._flush_task and not self._flush_task.done():
                self._flush_task.cancel()
                try:
                    await self._flush_task
                except asyncio.CancelledError:
                    logger.info("Tarefa de flush periódico cancelada durante save_state.")


            logger.info("Estado do DataManager salvo com sucesso.")

        except Exception as e:
            logger.exception("Erro ao salvar estado do DataManager:")

    # O método download_historical_data já estava presente, mas marcado como TODO.
    # A implementação real dependeria da API REST do TickTrader para dados históricos.
    # Se a API REST for usada, o método seria algo como:
    # async def download_and_store_historical_data(self, symbol: str, start_date: datetime, end_date: datetime):
    #     from api.ticktrader_rest import TickTraderREST # Importar aqui para evitar dependência circular no __init__
    #     rest_client = TickTraderREST()
    #     async with rest_client: # Usar context manager
    #         ticks_df = await rest_client.download_ticks(symbol, start_date, end_date)
    #         if ticks_df is not None and not ticks_df.empty:
    #             # Converter DataFrame para lista de dicts e armazenar via _flush_ticks
    #             # ou diretamente para Parquet se for um grande volume.
    #             ticks_list_of_dicts = ticks_df.reset_index().to_dict('records')
    #             # Adicionar ao buffer e deixar o _flush_ticks lidar, ou escrever diretamente.
    #             # Exemplo de escrita direta particionada (simplificado):
    #             for date_partition, group_df in ticks_df.groupby(ticks_df.index.date):
    #                 # ... lógica similar a _flush_ticks para salvar group_df ...
    #                 pass
    #             logger.info(f"Dados históricos de {symbol} baixados e armazenados.")
    #         else:
    #             logger.warning(f"Nenhum dado histórico baixado para {symbol} de {start_date} a {end_date}.")