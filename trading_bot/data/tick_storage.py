# data/tick_storage.py
"""Sistema de armazenamento eficiente de ticks em Parquet"""
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from typing import List, Optional, Dict, Union, Any 
from datetime import datetime, timedelta, timezone, date # <--- ADICIONE 'date' AQUI
from pathlib import Path # <--- ADICIONE Path AQUI
import os
import asyncio
from concurrent.futures import ThreadPoolExecutor

from config.settings import CONFIG
from utils.logger import setup_logger

logger = setup_logger("tick_storage")

class TickStorage:
    """Gerenciador de armazenamento de ticks em formato Parquet"""

    def __init__(self, base_path: str = CONFIG.PARQUET_PATH):
        self.base_path: Path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True) 

        self.tick_schema = pa.schema([
            ('timestamp', pa.timestamp('us', tz='UTC')), 
            ('symbol', pa.string()),
            ('bid', pa.float64()),
            ('ask', pa.float64()),
            ('bid_volume', pa.float64()),
            ('ask_volume', pa.float64()),
            ('mid', pa.float64()), 
            ('spread', pa.float64())
        ])

        self.write_buffer: List[Dict[str, Any]] = [] 
        self.buffer_size: int = 10000 if hasattr(CONFIG, 'TICK_HISTORY_YEARS') and CONFIG.TICK_HISTORY_YEARS > 1 else 5000
        self.last_flush: datetime = datetime.now(timezone.utc) 
        self.flush_interval_seconds: int = 60  

        self.executor = ThreadPoolExecutor(max_workers=os.cpu_count() or 1) 
        self._flush_lock = asyncio.Lock() 

    async def store_tick(self, tick: Dict[str, Any]): 
        """Armazena um tick no buffer. O tick ja deve ter 'mid' e 'spread' calculados se necessario."""
        if not all(k in tick for k in ['timestamp', 'symbol', 'bid', 'ask']):
            logger.warning(f"Tick invalido ou incompleto recebido: {tick}. Ignorando.")
            return

        try:
            if not isinstance(tick['timestamp'], datetime):
                if isinstance(tick['timestamp'], (int, float)): 
                    ts_val = tick['timestamp'] / 1000 if tick['timestamp'] > 1e12 else tick['timestamp']
                    tick['timestamp'] = datetime.fromtimestamp(ts_val, tz=timezone.utc)
                elif isinstance(tick['timestamp'], str):
                    tick['timestamp'] = pd.Timestamp(tick['timestamp'], tz='UTC').to_pydatetime(warn=False)
                else: 
                    logger.warning(f"Timestamp do tick em formato desconhecido: {tick['timestamp']}. Ignorando tick.")
                    return
            elif tick['timestamp'].tzinfo is None: 
                 tick['timestamp'] = tick['timestamp'].replace(tzinfo=timezone.utc)

            for key_num in ['bid', 'ask', 'bid_volume', 'ask_volume']:
                tick[key_num] = float(tick.get(key_num, 0.0))

            if 'mid' not in tick or not isinstance(tick['mid'], float):
                tick['mid'] = (tick['bid'] + tick['ask']) / 2.0
            if 'spread' not in tick or not isinstance(tick['spread'], float):
                tick['spread'] = tick['ask'] - tick['bid']
        except Exception as e:
            logger.error(f"Erro ao pre-processar tick para buffer: {e}. Tick: {tick}")
            return

        self.write_buffer.append(tick)

        if (len(self.write_buffer) >= self.buffer_size or
            (datetime.now(timezone.utc) - self.last_flush).total_seconds() >= self.flush_interval_seconds):
            await self.flush()


    async def store_ticks_batch(self, ticks: List[Dict[str, Any]]): 
        """Armazena batch de ticks, pre-processando cada um."""
        processed_ticks = []
        for tick_orig in ticks: 
            if not all(k in tick_orig for k in ['timestamp', 'symbol', 'bid', 'ask']):
                logger.warning(f"Tick invalido ou incompleto no batch: {tick_orig}. Ignorando.")
                continue
            try:
                tick = tick_orig.copy() 
                if not isinstance(tick['timestamp'], datetime):
                    if isinstance(tick['timestamp'], (int, float)):
                        ts_val = tick['timestamp'] / 1000 if tick['timestamp'] > 1e12 else tick['timestamp']
                        tick['timestamp'] = datetime.fromtimestamp(ts_val, tz=timezone.utc)
                    elif isinstance(tick['timestamp'], str):
                        tick['timestamp'] = pd.Timestamp(tick['timestamp'], tz='UTC').to_pydatetime(warn=False)
                    else:
                        logger.warning(f"Timestamp do tick em formato desconhecido no batch: {tick['timestamp']}. Ignorando tick.")
                        continue
                elif tick['timestamp'].tzinfo is None:
                    tick['timestamp'] = tick['timestamp'].replace(tzinfo=timezone.utc)

                for key_num_batch in ['bid', 'ask', 'bid_volume', 'ask_volume']: 
                    tick[key_num_batch] = float(tick.get(key_num_batch, 0.0))

                if 'mid' not in tick or not isinstance(tick['mid'], float):
                    tick['mid'] = (tick['bid'] + tick['ask']) / 2.0
                if 'spread' not in tick or not isinstance(tick['spread'], float):
                    tick['spread'] = tick['ask'] - tick['bid']
                processed_ticks.append(tick)
            except Exception as e:
                logger.error(f"Erro ao pre-processar tick do batch: {e}. Tick: {tick_orig}")
                continue

        if processed_ticks:
            self.write_buffer.extend(processed_ticks)
            if len(self.write_buffer) >= self.buffer_size:
                await self.flush()


    async def flush(self):
        """Persiste buffer para arquivo Parquet de forma segura contra concorrencia."""
        if not self.write_buffer:
            return

        async with self._flush_lock: 
            if not self.write_buffer: 
                return

            ticks_to_write_flush = self.write_buffer[:] 
            self.write_buffer = [] 
            current_flush_time = datetime.now(timezone.utc) 

            try:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    self.executor,
                    self._write_ticks_to_parquet_sync, 
                    ticks_to_write_flush
                )
                self.last_flush = current_flush_time 
                logger.info(f"Flush de {len(ticks_to_write_flush)} ticks para Parquet concluido.")
            except Exception as e:
                logger.exception("Erro critico no flush de ticks para Parquet:") 
                self.write_buffer = ticks_to_write_flush + self.write_buffer
                logger.info(f"{len(ticks_to_write_flush)} ticks devolvidos ao buffer devido a erro no flush.")


    def _write_ticks_to_parquet_sync(self, ticks_list: List[Dict[str, Any]]): 
        """Escreve lista de ticks (dicionarios) em arquivos Parquet (sincrono)."""
        if not ticks_list:
            return

        df = pd.DataFrame(ticks_list)
        if df.empty:
            return
        
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']) or df['timestamp'].dt.tz != timezone.utc:
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)


        df['date_partition'] = df['timestamp'].dt.strftime('%Y%m%d') 

        for (symbol_val, date_str_val), group_df_val in df.groupby(['symbol', 'date_partition']): 
            if group_df_val.empty:
                continue

            file_path = self._get_file_path(symbol_val, date_str_val) 
            file_path.parent.mkdir(parents=True, exist_ok=True) 

            df_to_write = group_df_val.drop(columns=['date_partition'])

            try:
                table = pa.Table.from_pandas(df_to_write, schema=self.tick_schema, preserve_index=False)
            except (pa.ArrowTypeError, pa.ArrowInvalid, ValueError) as e:
                logger.error(f"Erro de schema/tipo ao converter DataFrame para Arrow Table para {file_path}: {e}")
                logger.debug(f"DataFrame Info:\n{df_to_write.info(verbose=True)}") 
                logger.debug(f"Primeiras linhas do DataFrame problematico:\n{df_to_write.head().to_string()}")
                logger.debug(f"Schema esperado: {self.tick_schema}")
                try:
                    logger.warning(f"Tentando criar Arrow Table para {file_path} sem schema explicito como fallback.")
                    table = pa.Table.from_pandas(df_to_write, preserve_index=False)
                except Exception as e_fallback:
                    logger.error(f"Falha no fallback ao criar Arrow Table para {file_path}: {e_fallback}. Pulando este grupo.")
                    continue

            if file_path.exists():
                try:
                    existing_table = pq.read_table(file_path) 
                    existing_df = existing_table.to_pandas()
                    new_df_to_append = table.to_pandas() 
                    
                    combined_df = pd.concat([existing_df, new_df_to_append], ignore_index=True)
                    
                    combined_df.drop_duplicates(subset=['timestamp'], keep='last', inplace=True)
                    combined_df.sort_values('timestamp', inplace=True)
                    
                    final_table_to_write = pa.Table.from_pandas(combined_df, schema=self.tick_schema, preserve_index=False)
                    pq.write_table(final_table_to_write, file_path, compression='snappy')

                except Exception as e_append:
                    logger.error(f"Erro ao fazer append em {file_path}: {e_append}. Tentando sobrescrever com dados atuais do buffer.")
                    try:
                        pq.write_table(table, file_path, compression='snappy') 
                    except Exception as e_overwrite:
                         logger.error(f"Falha ao sobrescrever {file_path} apos erro de append: {e_overwrite}.")
            else: 
                pq.write_table(table, file_path, compression='snappy')


    def _get_file_path(self, symbol: str, date_obj_or_str: Union[datetime, date, str]) -> Path: 
        """Gera caminho do arquivo para simbolo e data. date_obj_or_str pode ser datetime, date, ou 'YYYYMMDD' string."""
        date_str_path: str
        if isinstance(date_obj_or_str, datetime): 
            date_str_path = date_obj_or_str.strftime('%Y%m%d') 
        elif isinstance(date_obj_or_str, date): 
            date_str_path = date_obj_or_str.strftime('%Y%m%d')
        else: 
            date_str_path = str(date_obj_or_str).replace('-', '')

        year_str = date_str_path[:4] 
        month_str = date_str_path[4:6] 

        return self.base_path / symbol / year_str / month_str / f"ticks_{symbol}_{date_str_path}.parquet"


    async def read_ticks(self,
                        symbol: str,
                        start_datetime_utc: datetime, 
                        end_datetime_utc: datetime,   
                        columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Le ticks de um periodo de arquivos Parquet."""
        if not (start_datetime_utc.tzinfo and end_datetime_utc.tzinfo):
            logger.error("Datas de inicio/fim para read_ticks devem ser cientes do fuso horario (UTC).")
            return pd.DataFrame()

        try:
            loop = asyncio.get_event_loop()
            df = await loop.run_in_executor(
                self.executor,
                self._read_ticks_sync,
                symbol,
                start_datetime_utc,
                end_datetime_utc,
                columns
            )
            return df
        except Exception as e:
            logger.exception(f"Erro ao ler ticks para {symbol} de {start_datetime_utc} a {end_datetime_utc}:")
            return pd.DataFrame()


    def _read_ticks_sync(self,
                        symbol: str,
                        start_datetime_utc: datetime,
                        end_datetime_utc: datetime,
                        columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Le ticks de forma sincrona de arquivos Parquet."""
        all_dfs_read: List[pd.DataFrame] = [] 
        current_date_read = start_datetime_utc.date() 
        end_date_read = end_datetime_utc.date() 

        while current_date_read <= end_date_read:
            file_path_read = self._get_file_path(symbol, current_date_read) 

            if file_path_read.exists():
                try:
                    table = pq.read_table(file_path_read, columns=columns)
                    df_day = table.to_pandas(timestamp_as_object=False) 
                    
                    if 'timestamp' in df_day.columns:
                        if not pd.api.types.is_datetime64_any_dtype(df_day['timestamp']) or df_day['timestamp'].dt.tz != timezone.utc:
                             df_day['timestamp'] = pd.to_datetime(df_day['timestamp'], utc=True)

                        df_filtered = df_day[
                            (df_day['timestamp'] >= start_datetime_utc) &
                            (df_day['timestamp'] <= end_datetime_utc)
                        ]
                        if not df_filtered.empty:
                            all_dfs_read.append(df_filtered)
                    else:
                        logger.warning(f"Arquivo {file_path_read} nao contem coluna 'timestamp' apos leitura.")

                except Exception as e_read: 
                    logger.error(f"Erro ao ler ou processar arquivo Parquet {file_path_read}: {e_read}")

            current_date_read += timedelta(days=1)

        if all_dfs_read:
            combined_df = pd.concat(all_dfs_read, ignore_index=True)
            if 'timestamp' in combined_df.columns:
                 combined_df.sort_values('timestamp', inplace=True)
            return combined_df

        return pd.DataFrame()


    async def get_tick_count_for_date(self, symbol: str, target_date: date) -> int: 
        """Retorna contagem de ticks para uma data especifica."""
        file_path_count = self._get_file_path(symbol, target_date) 

        if file_path_count.exists():
            try:
                parquet_file = pq.ParquetFile(file_path_count)
                return parquet_file.metadata.num_rows
            except Exception as e:
                logger.exception(f"Erro ao contar ticks em {file_path_count}:")
        return 0


    async def get_storage_stats(self) -> Dict[str, Any]: 
        """Retorna estatisticas de armazenamento (tamanho total, por simbolo, etc.)."""
        stats: Dict[str, Any] = { 
            'total_files': 0,
            'total_size_mb': 0.0, 
            'symbols': {},
            'buffer_fill_percentage': (len(self.write_buffer) / self.buffer_size) * 100 if self.buffer_size > 0 else 0.0,
            'oldest_tick_in_buffer_age_sec': None,
            'last_flush_ago_sec': (datetime.now(timezone.utc) - self.last_flush).total_seconds()
        }
        if self.write_buffer and 'timestamp' in self.write_buffer[0]:
             oldest_ts_in_buffer = self.write_buffer[0]['timestamp']
             if isinstance(oldest_ts_in_buffer, datetime):
                stats['oldest_tick_in_buffer_age_sec'] = (datetime.now(timezone.utc) - oldest_ts_in_buffer).total_seconds()

        try:
            for symbol_dir in self.base_path.iterdir():
                if symbol_dir.is_dir():
                    symbol_name = symbol_dir.name 
                    symbol_stats_dict: Dict[str, Any] = { 
                        'files': 0,
                        'size_mb': 0.0,
                        'date_range': {'start': None, 'end': None}
                    }

                    for file_path_stat in symbol_dir.rglob('*.parquet'): 
                        if file_path_stat.is_file(): 
                            symbol_stats_dict['files'] += 1
                            try:
                                size_bytes = file_path_stat.stat().st_size 
                                size_mb_val = size_bytes / (1024 * 1024) 
                                symbol_stats_dict['size_mb'] += size_mb_val
                                stats['total_size_mb'] += size_mb_val
                            except FileNotFoundError: 
                                logger.warning(f"Arquivo {file_path_stat} nao encontrado durante stat em get_storage_stats.")
                                continue

                            stats['total_files'] += 1
                            try:
                                date_str_stat = file_path_stat.stem.split('_')[-1] 
                                if len(date_str_stat) == 8 and date_str_stat.isdigit(): 
                                    if symbol_stats_dict['date_range']['start'] is None or date_str_stat < symbol_stats_dict['date_range']['start']:
                                        symbol_stats_dict['date_range']['start'] = date_str_stat
                                    if symbol_stats_dict['date_range']['end'] is None or date_str_stat > symbol_stats_dict['date_range']['end']:
                                        symbol_stats_dict['date_range']['end'] = date_str_stat
                            except IndexError:
                                logger.warning(f"Nao foi possivel extrair data do nome do arquivo: {file_path_stat.name}")

                    stats['symbols'][symbol_name] = symbol_stats_dict
            return stats
        except Exception as e:
            logger.exception("Erro ao obter estatisticas de armazenamento:")
            return stats 


    async def cleanup_old_data(self, days_to_keep: int = (CONFIG.TICK_HISTORY_YEARS * 365 if hasattr(CONFIG, 'TICK_HISTORY_YEARS') else 90)): # Corrigido
        """Remove arquivos Parquet de ticks mais antigos que 'days_to_keep'."""
        if days_to_keep <= 0:
            logger.info("Limpeza de dados antigos desabilitada (days_to_keep <= 0).")
            return

        cutoff_date_obj = (datetime.now(timezone.utc) - timedelta(days=days_to_keep)).date() 
        files_removed_count = 0 
        space_freed_mb_val = 0.0 

        logger.info(f"Iniciando limpeza de dados Parquet anteriores a {cutoff_date_obj.isoformat()}...")
        try:
            for file_path_cleanup in self.base_path.rglob('*.parquet'): 
                if not file_path_cleanup.is_file(): continue
                try:
                    date_str_cleanup = file_path_cleanup.stem.split('_')[-1] 
                    if len(date_str_cleanup) == 8 and date_str_cleanup.isdigit():
                        file_date_obj = datetime.strptime(date_str_cleanup, '%Y%m%d').date() 

                        if file_date_obj < cutoff_date_obj:
                            try:
                                size_bytes_val = file_path_cleanup.stat().st_size 
                                file_path_cleanup.unlink() 
                                files_removed_count += 1
                                space_freed_mb_val += size_bytes_val / (1024 * 1024)
                                logger.debug(f"Removido arquivo antigo: {file_path_cleanup}")
                            except FileNotFoundError:
                                logger.warning(f"Arquivo {file_path_cleanup} nao encontrado durante a remocao (pode ter sido removido por outro processo).")
                            except Exception as e_unlink: 
                                logger.error(f"Erro ao remover arquivo {file_path_cleanup}: {e_unlink}")
                    else:
                        logger.debug(f"Nome de arquivo nao corresponde ao padrao de data para limpeza: {file_path_cleanup.name}")
                except (IndexError, ValueError) as e_parse_date: 
                    logger.warning(f"Nao foi possivel parsear data do arquivo {file_path_cleanup.name} para limpeza: {e_parse_date}")

            logger.info(f"Limpeza de dados Parquet concluida: {files_removed_count} arquivos removidos, "
                       f"{space_freed_mb_val:.2f} MB liberados.")
        except Exception as e_cleanup_main: 
            logger.exception("Erro durante a limpeza de dados antigos:")


    async def optimize_parquet_files(self): 
        """
        Otimiza arquivos Parquet existentes.
        """
        logger.info("Iniciando otimizacao de arquivos Parquet...")
        optimized_files_count = 0 

        try:
            for file_path_opt in self.base_path.rglob('*.parquet'): 
                if not file_path_opt.is_file(): continue
                logger.debug(f"Otimizando arquivo: {file_path_opt}")
                try:
                    table_opt = pq.read_table(file_path_opt) 
                    df_opt = table_opt.to_pandas(timestamp_as_object=False) 

                    if df_opt.empty:
                        logger.debug(f"Arquivo {file_path_opt} vazio, pulando otimizacao.")
                        continue

                    original_rows = len(df_opt)
                    changes_made = False

                    if 'timestamp' in df_opt.columns:
                        if not pd.api.types.is_datetime64_any_dtype(df_opt['timestamp']) or df_opt['timestamp'].dt.tz != timezone.utc:
                            df_opt['timestamp'] = pd.to_datetime(df_opt['timestamp'], utc=True)
                            changes_made = True 
                        
                        if not df_opt['timestamp'].is_monotonic_increasing:
                            df_opt.sort_values('timestamp', inplace=True)
                            changes_made = True
                        
                        len_before_drop = len(df_opt)
                        df_opt.drop_duplicates(subset=['timestamp'], keep='last', inplace=True)
                        if len(df_opt) < len_before_drop:
                            changes_made = True

                    try:
                        re_schematized_table = pa.Table.from_pandas(df_opt, schema=self.tick_schema, preserve_index=False)
                        if not table_opt.schema.equals(re_schematized_table.schema) or changes_made:
                             pq.write_table(re_schematized_table, file_path_opt, compression='snappy')
                             optimized_files_count += 1
                             logger.info(f"Otimizado {file_path_opt.name}: "
                                        f"{original_rows} -> {len(df_opt)} linhas. Schema aplicado/Dados alterados.")

                    except (pa.ArrowTypeError, pa.ArrowInvalid, ValueError) as e_schema_opt:
                        logger.error(f"Erro de schema ao otimizar {file_path_opt}: {e_schema_opt}. Arquivo nao foi reescrito com schema.")

                except Exception as e_opt_file: 
                    logger.exception(f"Erro ao otimizar arquivo Parquet {file_path_opt}:")

            logger.info(f"Otimizacao de arquivos Parquet concluida. {optimized_files_count} arquivos reescritos/otimizados.")
        except Exception as e_opt_main: 
            logger.exception("Erro durante a otimizacao de armazenamento Parquet:")


    async def close(self):
        """Fecha o TickStorage e faz flush final dos ticks no buffer."""
        logger.info("Fechando TickStorage...")
        await self.flush() 
        self.executor.shutdown(wait=True) 
        logger.info("TickStorage fechado e executor finalizado.")