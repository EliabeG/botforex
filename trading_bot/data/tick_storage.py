# data/tick_storage.py
"""Sistema de armazenamento eficiente de ticks em Parquet"""
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from typing import List, Optional, Dict, Union, Any # Adicionado Any
from datetime import datetime, timedelta, timezone # Adicionado timezone
from pathlib import Path
import os
import asyncio
from concurrent.futures import ThreadPoolExecutor

from config.settings import CONFIG
# Supondo que TickData possa vir da API ou ser um dict simples aqui
# from api.ticktrader_ws import TickData # Se for usar o objeto TickData
from utils.logger import setup_logger

logger = setup_logger("tick_storage")

class TickStorage:
    """Gerenciador de armazenamento de ticks em formato Parquet"""

    def __init__(self, base_path: str = CONFIG.PARQUET_PATH):
        self.base_path: Path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True) # Garantir que o diretório base exista

        # Schema para ticks (mais robusto se os tipos forem estritos)
        self.tick_schema = pa.schema([
            ('timestamp', pa.timestamp('us', tz='UTC')), # Microsegundos, com timezone UTC
            ('symbol', pa.string()),
            ('bid', pa.float64()),
            ('ask', pa.float64()),
            ('bid_volume', pa.float64()),
            ('ask_volume', pa.float64()),
            ('mid', pa.float64()), # mid e spread são calculados, mas bom ter no schema
            ('spread', pa.float64())
            # Adicionar outros campos se existirem consistentemente nos ticks, ex: 'last_price', 'last_size'
        ])

        # Buffer para escrita eficiente
        self.write_buffer: List[Dict[str, Any]] = [] # Tipagem mais específica
        self.buffer_size: int = CONFIG.TICK_HISTORY_YEARS > 1 and 10000 or 5000 # Exemplo de ajuste, pode vir de CONFIG
        self.last_flush: datetime = datetime.now(timezone.utc) # Usar UTC
        self.flush_interval_seconds: int = 60  # Renomeado para clareza

        # Executor para operações de I/O (número de workers pode ser configurável)
        self.executor = ThreadPoolExecutor(max_workers=os.cpu_count() or 1) # Usar os.cpu_count()
        self._flush_lock = asyncio.Lock() # Lock para evitar flushes concorrentes

    async def store_tick(self, tick: Dict[str, Any]): # tick é um dicionário
        """Armazena um tick no buffer. O tick já deve ter 'mid' e 'spread' calculados se necessário."""
        # Validação básica do tick
        if not all(k in tick for k in ['timestamp', 'symbol', 'bid', 'ask']):
            logger.warning(f"Tick inválido ou incompleto recebido: {tick}. Ignorando.")
            return

        # Adicionar/Garantir campos calculados e tipos corretos antes de adicionar ao buffer
        try:
            # Converter timestamp para datetime object ciente do UTC se ainda não for
            if not isinstance(tick['timestamp'], datetime):
                if isinstance(tick['timestamp'], (int, float)): # Assumindo ms ou s
                    ts_val = tick['timestamp'] / 1000 if tick['timestamp'] > 1e12 else tick['timestamp']
                    tick['timestamp'] = datetime.fromtimestamp(ts_val, tz=timezone.utc)
                elif isinstance(tick['timestamp'], str):
                    tick['timestamp'] = pd.Timestamp(tick['timestamp'], tz='UTC').to_pydatetime(warn=False)
                else: # Tipo desconhecido, não pode processar
                    logger.warning(f"Timestamp do tick em formato desconhecido: {tick['timestamp']}. Ignorando tick.")
                    return
            elif tick['timestamp'].tzinfo is None: # Se for datetime naive
                 tick['timestamp'] = tick['timestamp'].replace(tzinfo=timezone.utc)


            # Garantir que os campos numéricos sejam floats
            for key_num in ['bid', 'ask', 'bid_volume', 'ask_volume']:
                tick[key_num] = float(tick.get(key_num, 0.0))

            if 'mid' not in tick or not isinstance(tick['mid'], float):
                tick['mid'] = (tick['bid'] + tick['ask']) / 2.0
            if 'spread' not in tick or not isinstance(tick['spread'], float):
                tick['spread'] = tick['ask'] - tick['bid']
        except Exception as e:
            logger.error(f"Erro ao pré-processar tick para buffer: {e}. Tick: {tick}")
            return


        self.write_buffer.append(tick)

        if (len(self.write_buffer) >= self.buffer_size or
            (datetime.now(timezone.utc) - self.last_flush).total_seconds() >= self.flush_interval_seconds):
            await self.flush()


    async def store_ticks_batch(self, ticks: List[Dict[str, Any]]): # Usar Any
        """Armazena batch de ticks, pré-processando cada um."""
        processed_ticks = []
        for tick_orig in ticks: # Renomeado tick para tick_orig
            # Validação e pré-processamento similar ao store_tick individual
            if not all(k in tick_orig for k in ['timestamp', 'symbol', 'bid', 'ask']):
                logger.warning(f"Tick inválido ou incompleto no batch: {tick_orig}. Ignorando.")
                continue
            try:
                tick = tick_orig.copy() # Trabalhar com uma cópia
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

                for key_num_batch in ['bid', 'ask', 'bid_volume', 'ask_volume']: # Renomeado
                    tick[key_num_batch] = float(tick.get(key_num_batch, 0.0))

                if 'mid' not in tick or not isinstance(tick['mid'], float):
                    tick['mid'] = (tick['bid'] + tick['ask']) / 2.0
                if 'spread' not in tick or not isinstance(tick['spread'], float):
                    tick['spread'] = tick['ask'] - tick['bid']
                processed_ticks.append(tick)
            except Exception as e:
                logger.error(f"Erro ao pré-processar tick do batch: {e}. Tick: {tick_orig}")
                continue


        if processed_ticks:
            self.write_buffer.extend(processed_ticks)
            if len(self.write_buffer) >= self.buffer_size:
                await self.flush()


    async def flush(self):
        """Persiste buffer para arquivo Parquet de forma segura contra concorrência."""
        if not self.write_buffer:
            return

        async with self._flush_lock: # Adquirir lock para este bloco
            if not self.write_buffer: # Dupla verificação após adquirir o lock
                return

            ticks_to_write_flush = self.write_buffer[:] # Renomeado, Copiar buffer
            self.write_buffer = [] # Limpar buffer original
            current_flush_time = datetime.now(timezone.utc) # Usar UTC

            try:
                loop = asyncio.get_event_loop()
                # _write_to_parquet agora espera uma lista de dicionários
                await loop.run_in_executor(
                    self.executor,
                    self._write_ticks_to_parquet_sync, # Renomeado para clareza
                    ticks_to_write_flush
                )
                self.last_flush = current_flush_time # Atualizar last_flush apenas se o write for bem-sucedido
                logger.info(f"Flush de {len(ticks_to_write_flush)} ticks para Parquet concluído.")
            except Exception as e:
                logger.exception("Erro crítico no flush de ticks para Parquet:") # Usar logger.exception
                # Devolver ticks ao buffer em caso de erro para tentar novamente depois
                self.write_buffer = ticks_to_write_flush + self.write_buffer
                logger.info(f"{len(ticks_to_write_flush)} ticks devolvidos ao buffer devido a erro no flush.")



    def _write_ticks_to_parquet_sync(self, ticks_list: List[Dict[str, Any]]): # Renomeado, Usar Any
        """Escreve lista de ticks (dicionários) em arquivos Parquet (síncrono)."""
        if not ticks_list:
            return

        # Converter para DataFrame de uma vez
        df = pd.DataFrame(ticks_list)
        if df.empty:
            return

        # Garantir que 'timestamp' é datetime[ns, UTC]
        # Os timestamps já devem ser datetime objects UTC do pré-processamento
        # df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)

        # Agrupar por símbolo e data para salvar arquivos separados
        # É importante que 'timestamp' seja datetime para extrair 'date_partition'
        df['date_partition'] = df['timestamp'].dt.strftime('%Y%m%d')

        for (symbol_val, date_str_val), group_df_val in df.groupby(['symbol', 'date_partition']): # Renomeado
            if group_df_val.empty:
                continue

            file_path = self._get_file_path(symbol_val, date_str_val) # Passar date_str_val
            file_path.parent.mkdir(parents=True, exist_ok=True) # Garantir que o diretório do dia exista

            # Manter apenas colunas do schema e na ordem do schema para consistência
            # (se o schema for estrito e não permitir colunas extras)
            # df_to_write = group_df_val[self.tick_schema.names].copy()
            # Mas é mais flexível permitir colunas extras e deixar o schema do Arrow Table lidar com isso
            df_to_write = group_df_val.drop(columns=['date_partition'])


            try:
                # Tentar criar a tabela com o schema definido
                table = pa.Table.from_pandas(df_to_write, schema=self.tick_schema, preserve_index=False)
            except (pa.ArrowTypeError, pa.ArrowInvalid, ValueError) as e:
                logger.error(f"Erro de schema/tipo ao converter DataFrame para Arrow Table para {file_path}: {e}")
                logger.debug(f"DataFrame Info:\n{df_to_write.info()}")
                logger.debug(f"Primeiras linhas do DataFrame problemático:\n{df_to_write.head()}")
                logger.debug(f"Schema esperado: {self.tick_schema}")
                # Tentar sem schema explícito como fallback (pode gerar schemas inconsistentes entre arquivos)
                try:
                    logger.warning(f"Tentando criar Arrow Table para {file_path} sem schema explícito como fallback.")
                    table = pa.Table.from_pandas(df_to_write, preserve_index=False)
                except Exception as e_fallback:
                    logger.error(f"Falha no fallback ao criar Arrow Table para {file_path}: {e_fallback}. Pulando este grupo.")
                    continue


            # Escrever ou fazer append
            if file_path.exists():
                try:
                    existing_table = pq.read_table(file_path, schema=self.tick_schema)
                    # Para append eficiente e correto, os schemas devem ser compatíveis.
                    # Se o schema for o mesmo, concatenação de tabelas Arrow é preferível.
                    combined_table = pa.concat_tables([existing_table, table])

                    # Remoção de duplicatas e ordenação em Arrow/Pandas pode ser custosa para append.
                    # Uma estratégia comum é escrever arquivos menores e depois ter um job de compactação/otimização.
                    # Para este append simples:
                    pq.write_table(combined_table, file_path, compression='snappy')
                except Exception as e_append:
                    logger.error(f"Erro ao fazer append em {file_path}: {e_append}. Tentando sobrescrever (RISCO DE PERDA DE DADOS ANTERIORES SE NÃO FOR O MESMO DIA).")
                    # CUIDADO: Sobrescrever pode não ser o ideal se o arquivo contém dados de flushes anteriores NO MESMO DIA.
                    # Uma estratégia mais segura seria numerar os arquivos por flush (ex: ticks_sym_date_001.parquet)
                    # e depois consolidá-los. A lógica atual de append (read-concat-write) é mais segura contra perda
                    # mas mais lenta. Se o append falha, a sobrescrita aqui seria com os dados *atuais do buffer apenas*.
                    try:
                        pq.write_table(table, file_path, compression='snappy')
                    except Exception as e_overwrite:
                         logger.error(f"Falha ao sobrescrever {file_path} após erro de append: {e_overwrite}.")

            else: # Arquivo não existe, criar novo
                pq.write_table(table, file_path, compression='snappy')


    def _get_file_path(self, symbol: str, date_obj_or_str: Union[datetime, date, str]) -> Path: # Adicionado date
        """Gera caminho do arquivo para símbolo e data. date_obj_or_str pode ser datetime, date, ou 'YYYYMMDD' string."""
        if isinstance(date_obj_or_str, (datetime, date)): # Usar date da lib datetime
            date_str_path = date_obj_or_str.strftime('%Y%m%d') # Renomeado
        else: # Assumir que é uma string 'YYYYMMDD'
            date_str_path = str(date_obj_or_str).replace('-', '')

        year_str = date_str_path[:4] # Renomeado
        month_str = date_str_path[4:6] # Renomeado

        return self.base_path / symbol / year_str / month_str / f"ticks_{symbol}_{date_str_path}.parquet"


    async def read_ticks(self,
                        symbol: str,
                        start_datetime_utc: datetime, # Renomeado e especificado UTC
                        end_datetime_utc: datetime,   # Renomeado e especificado UTC
                        columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Lê ticks de um período de arquivos Parquet."""
        if not (start_datetime_utc.tzinfo and end_datetime_utc.tzinfo):
            logger.error("Datas de início/fim para read_ticks devem ser cientes do fuso horário (UTC).")
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
        """Lê ticks de forma síncrona de arquivos Parquet."""
        all_dfs_read: List[pd.DataFrame] = [] # Renomeado
        current_date_read = start_datetime_utc.date() # Renomeado
        end_date_read = end_datetime_utc.date() # Renomeado

        while current_date_read <= end_date_read:
            file_path_read = self._get_file_path(symbol, current_date_read) # Renomeado

            if file_path_read.exists():
                try:
                    # Ler arquivo, especificando colunas se fornecido
                    # O schema do Parquet será usado implicitamente se consistente
                    table = pq.read_table(file_path_read, columns=columns)
                    df_day = table.to_pandas(timestamp_as_object=False) # Melhor performance com tipos Arrow
                    
                    # Garantir que a coluna timestamp seja datetime e UTC
                    if 'timestamp' in df_day.columns:
                        # pd.to_datetime já lida com Arrow timestamps, mas tz='UTC' garante
                        df_day['timestamp'] = pd.to_datetime(df_day['timestamp'], utc=True)

                        # Filtrar pelo range exato de datetime
                        # (necessário porque lemos arquivos diários inteiros)
                        df_filtered = df_day[
                            (df_day['timestamp'] >= start_datetime_utc) &
                            (df_day['timestamp'] <= end_datetime_utc)
                        ]
                        if not df_filtered.empty:
                            all_dfs_read.append(df_filtered)
                    else:
                        logger.warning(f"Arquivo {file_path_read} não contém coluna 'timestamp' após leitura.")


                except Exception as e_read: # Renomeado
                    logger.error(f"Erro ao ler ou processar arquivo Parquet {file_path_read}: {e_read}")

            current_date_read += timedelta(days=1)

        if all_dfs_read:
            combined_df = pd.concat(all_dfs_read, ignore_index=True)
            # A ordenação pode não ser necessária se os arquivos diários já estiverem ordenados
            # e a concatenação mantiver a ordem dos DataFrames.
            # No entanto, para garantir:
            if 'timestamp' in combined_df.columns:
                 combined_df.sort_values('timestamp', inplace=True)
            return combined_df

        return pd.DataFrame()


    async def get_tick_count_for_date(self, symbol: str, target_date: date) -> int: # Renomeado e tipo de target_date
        """Retorna contagem de ticks para uma data específica."""
        file_path_count = self._get_file_path(symbol, target_date) # Renomeado

        if file_path_count.exists():
            try:
                parquet_file = pq.ParquetFile(file_path_count)
                return parquet_file.metadata.num_rows
            except Exception as e:
                logger.exception(f"Erro ao contar ticks em {file_path_count}:")
        return 0


    async def get_storage_stats(self) -> Dict[str, Any]: # Usar Any
        """Retorna estatísticas de armazenamento (tamanho total, por símbolo, etc.)."""
        stats: Dict[str, Any] = { # Adicionada tipagem
            'total_files': 0,
            'total_size_mb': 0.0, # Float
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
                    symbol_name = symbol_dir.name # Renomeado
                    symbol_stats_dict: Dict[str, Any] = { # Renomeado e tipado
                        'files': 0,
                        'size_mb': 0.0,
                        'date_range': {'start': None, 'end': None}
                    }

                    # Usar rglob para encontrar todos os .parquet recursivamente
                    for file_path_stat in symbol_dir.rglob('*.parquet'): # Renomeado
                        if file_path_stat.is_file(): # Checar se é arquivo
                            symbol_stats_dict['files'] += 1
                            try:
                                size_bytes = file_path_stat.stat().st_size
                                size_mb_val = size_bytes / (1024 * 1024) # Renomeado
                                symbol_stats_dict['size_mb'] += size_mb_val
                                stats['total_size_mb'] += size_mb_val
                            except FileNotFoundError: # Arquivo pode ser removido entre iterdir e stat
                                logger.warning(f"Arquivo {file_path_stat} não encontrado durante stat em get_storage_stats.")
                                continue


                            stats['total_files'] += 1

                            try:
                                date_str_stat = file_path_stat.stem.split('_')[-1] # Renomeado
                                if len(date_str_stat) == 8 and date_str_stat.isdigit(): # Validação básica
                                    if symbol_stats_dict['date_range']['start'] is None or date_str_stat < symbol_stats_dict['date_range']['start']:
                                        symbol_stats_dict['date_range']['start'] = date_str_stat
                                    if symbol_stats_dict['date_range']['end'] is None or date_str_stat > symbol_stats_dict['date_range']['end']:
                                        symbol_stats_dict['date_range']['end'] = date_str_stat
                            except IndexError:
                                logger.warning(f"Não foi possível extrair data do nome do arquivo: {file_path_stat.name}")


                    stats['symbols'][symbol_name] = symbol_stats_dict

            return stats
        except Exception as e:
            logger.exception("Erro ao obter estatísticas de armazenamento:")
            return stats # Retornar stats parciais ou vazios


    async def cleanup_old_data(self, days_to_keep: int = 90):
        """Remove arquivos Parquet de ticks mais antigos que 'days_to_keep'."""
        if days_to_keep <= 0:
            logger.info("Limpeza de dados antigos desabilitada (days_to_keep <= 0).")
            return

        cutoff_date_obj = (datetime.now(timezone.utc) - timedelta(days=days_to_keep)).date() # Renomeado
        files_removed_count = 0 # Renomeado
        space_freed_mb_val = 0.0 # Renomeado

        logger.info(f"Iniciando limpeza de dados Parquet anteriores a {cutoff_date_obj.isoformat()}...")
        try:
            for file_path_cleanup in self.base_path.rglob('*.parquet'): # Renomeado
                if not file_path_cleanup.is_file(): continue

                try:
                    # Extrair data do nome do arquivo (ex: ticks_EURUSD_20230101.parquet)
                    date_str_cleanup = file_path_cleanup.stem.split('_')[-1] # Renomeado
                    if len(date_str_cleanup) == 8 and date_str_cleanup.isdigit():
                        file_date_obj = datetime.strptime(date_str_cleanup, '%Y%m%d').date() # Renomeado

                        if file_date_obj < cutoff_date_obj:
                            try:
                                size_bytes_val = file_path_cleanup.stat().st_size # Renomeado
                                file_path_cleanup.unlink() # Remover arquivo
                                files_removed_count += 1
                                space_freed_mb_val += size_bytes_val / (1024 * 1024)
                                logger.debug(f"Removido arquivo antigo: {file_path_cleanup}")
                            except FileNotFoundError:
                                logger.warning(f"Arquivo {file_path_cleanup} não encontrado durante a remoção (pode ter sido removido por outro processo).")
                            except Exception as e_unlink: # Renomeado
                                logger.error(f"Erro ao remover arquivo {file_path_cleanup}: {e_unlink}")

                    else:
                        logger.debug(f"Nome de arquivo não corresponde ao padrão de data para limpeza: {file_path_cleanup.name}")

                except (IndexError, ValueError) as e_parse_date: # Renomeado
                    logger.warning(f"Não foi possível parsear data do arquivo {file_path_cleanup.name} para limpeza: {e_parse_date}")

            logger.info(f"Limpeza de dados Parquet concluída: {files_removed_count} arquivos removidos, "
                       f"{space_freed_mb_val:.2f} MB liberados.")

        except Exception as e_cleanup_main: # Renomeado
            logger.exception("Erro durante a limpeza de dados antigos:")


    async def optimize_parquet_files(self): # Renomeado de optimize_storage
        """
        Otimiza arquivos Parquet existentes (ex: reescrevendo com schema, removendo duplicatas,
        ordenando, ou consolidando arquivos pequenos - escopo atual foca em schema e duplicatas).
        Esta é uma operação potencialmente longa e intensiva em I/O.
        """
        logger.info("Iniciando otimização de arquivos Parquet...")
        optimized_files_count = 0 # Renomeado

        try:
            for file_path_opt in self.base_path.rglob('*.parquet'): # Renomeado
                if not file_path_opt.is_file(): continue
                logger.debug(f"Otimizando arquivo: {file_path_opt}")
                try:
                    table_opt = pq.read_table(file_path_opt) # Renomeado
                    df_opt = table_opt.to_pandas(timestamp_as_object=False) # Renomeado

                    if df_opt.empty:
                        logger.debug(f"Arquivo {file_path_opt} vazio, pulando otimização.")
                        continue

                    original_rows = len(df_opt)
                    changes_made = False

                    # 1. Garantir timestamp UTC e ordenar
                    if 'timestamp' in df_opt.columns:
                        df_opt['timestamp'] = pd.to_datetime(df_opt['timestamp'], utc=True)
                        if not df_opt['timestamp'].is_monotonic_increasing:
                            df_opt.sort_values('timestamp', inplace=True)
                            changes_made = True
                        # 2. Remover duplicatas baseadas em timestamp (mantendo a última ocorrência)
                        #    Isso assume que 'timestamp' deve ser único ou que apenas o último é relevante.
                        #    Para ticks, duplicatas exatas de timestamp podem ser raras mas possíveis.
                        #    Se outros campos definem unicidade, ajustar 'subset'.
                        len_before_drop = len(df_opt)
                        df_opt.drop_duplicates(subset=['timestamp'], keep='last', inplace=True)
                        if len(df_opt) < len_before_drop:
                            changes_made = True

                    # 3. Garantir schema (reescrever com o schema da classe)
                    # Isso pode falhar se os dados não forem compatíveis com self.tick_schema
                    try:
                        # Recriar a tabela com o schema explícito.
                        # Isso irá reordenar colunas e potencialmente converter tipos.
                        re_schematized_table = pa.Table.from_pandas(df_opt, schema=self.tick_schema, preserve_index=False)
                        # Comparar schemas pode ser complexo. Uma forma simples é verificar se a reescrita é necessária.
                        # Se a tabela original já tinha o schema correto, table_opt.schema.equals(self.tick_schema) seria True.
                        # Aqui, vamos assumir que se conseguimos converter para o schema, é uma forma de "otimização".
                        if not table_opt.schema.equals(re_schematized_table.schema) or changes_made:
                             pq.write_table(re_schematized_table, file_path_opt, compression='snappy')
                             optimized_files_count += 1
                             logger.info(f"Otimizado {file_path_opt.name}: "
                                        f"{original_rows} -> {len(df_opt)} linhas (duplicatas/ordem). Schema aplicado.")
                        elif changes_made: # Schema igual, mas dados mudaram (duplicatas/ordem)
                             pq.write_table(re_schematized_table, file_path_opt, compression='snappy')
                             optimized_files_count += 1
                             logger.info(f"Otimizado {file_path_opt.name}: "
                                        f"{original_rows} -> {len(df_opt)} linhas (duplicatas/ordem).")


                    except (pa.ArrowTypeError, pa.ArrowInvalid, ValueError) as e_schema_opt:
                        logger.error(f"Erro de schema ao otimizar {file_path_opt}: {e_schema_opt}. Arquivo não foi reescrito com schema.")


                except Exception as e_opt_file: # Renomeado
                    logger.exception(f"Erro ao otimizar arquivo Parquet {file_path_opt}:")


            logger.info(f"Otimização de arquivos Parquet concluída. {optimized_files_count} arquivos reescritos/otimizados.")
        except Exception as e_opt_main: # Renomeado
            logger.exception("Erro durante a otimização de armazenamento Parquet:")


    async def close(self):
        """Fecha o TickStorage e faz flush final dos ticks no buffer."""
        logger.info("Fechando TickStorage...")
        await self.flush() # Garantir que todos os ticks no buffer sejam escritos
        self.executor.shutdown(wait=True) # Esperar que todas as tarefas de escrita pendentes terminem
        logger.info("TickStorage fechado e executor finalizado.")