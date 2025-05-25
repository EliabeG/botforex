# data/tick_storage.py
"""Sistema de armazenamento eficiente de ticks em Parquet"""
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from typing import List, Optional, Dict, Union
from datetime import datetime, timedelta
from pathlib import Path
import os
import asyncio
from concurrent.futures import ThreadPoolExecutor

from config.settings import CONFIG
from utils.logger import setup_logger

logger = setup_logger("tick_storage")

class TickStorage:
    """Gerenciador de armazenamento de ticks em formato Parquet"""
    
    def __init__(self, base_path: str = CONFIG.PARQUET_PATH):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Schema para ticks
        self.tick_schema = pa.schema([
            ('timestamp', pa.timestamp('ms')),
            ('symbol', pa.string()),
            ('bid', pa.float64()),
            ('ask', pa.float64()),
            ('bid_volume', pa.float64()),
            ('ask_volume', pa.float64()),
            ('mid', pa.float64()),
            ('spread', pa.float64())
        ])
        
        # Buffer para escrita eficiente
        self.write_buffer = []
        self.buffer_size = 10000
        self.last_flush = datetime.now()
        self.flush_interval = 60  # segundos
        
        # Executor para operações de I/O
        self.executor = ThreadPoolExecutor(max_workers=2)
        
    async def store_tick(self, tick: Dict):
        """Armazena um tick no buffer"""
        # Adicionar campos calculados se não existirem
        if 'mid' not in tick:
            tick['mid'] = (tick['bid'] + tick['ask']) / 2
        if 'spread' not in tick:
            tick['spread'] = tick['ask'] - tick['bid']
        
        self.write_buffer.append(tick)
        
        # Verificar se deve fazer flush
        if (len(self.write_buffer) >= self.buffer_size or 
            (datetime.now() - self.last_flush).seconds > self.flush_interval):
            await self.flush()
    
    async def store_ticks_batch(self, ticks: List[Dict]):
        """Armazena batch de ticks"""
        for tick in ticks:
            if 'mid' not in tick:
                tick['mid'] = (tick['bid'] + tick['ask']) / 2
            if 'spread' not in tick:
                tick['spread'] = tick['ask'] - tick['bid']
        
        self.write_buffer.extend(ticks)
        
        if len(self.write_buffer) >= self.buffer_size:
            await self.flush()
    
    async def flush(self):
        """Persiste buffer para arquivo Parquet"""
        if not self.write_buffer:
            return
        
        try:
            # Copiar buffer e limpar
            ticks_to_write = self.write_buffer.copy()
            self.write_buffer = []
            self.last_flush = datetime.now()
            
            # Executar escrita em thread separada
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                self.executor,
                self._write_to_parquet,
                ticks_to_write
            )
            
            logger.info(f"Flush de {len(ticks_to_write)} ticks concluído")
            
        except Exception as e:
            logger.error(f"Erro no flush de ticks: {e}")
            # Retornar ticks ao buffer em caso de erro
            self.write_buffer = ticks_to_write + self.write_buffer
    
    def _write_to_parquet(self, ticks: List[Dict]):
        """Escreve ticks em arquivo Parquet (síncrono)"""
        # Converter para DataFrame
        df = pd.DataFrame(ticks)
        
        # Garantir tipos corretos
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        numeric_columns = ['bid', 'ask', 'bid_volume', 'ask_volume', 'mid', 'spread']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Agrupar por data e símbolo
        df['date'] = df['timestamp'].dt.date
        
        for (date, symbol), group in df.groupby(['date', 'symbol']):
            # Caminho do arquivo
            file_path = self._get_file_path(symbol, date)
            
            # Remover coluna auxiliar
            group = group.drop('date', axis=1)
            
            # Escrever ou fazer append
            if file_path.exists():
                # Ler existente
                existing_df = pd.read_parquet(file_path)
                # Combinar
                combined_df = pd.concat([existing_df, group], ignore_index=True)
                # Ordenar por timestamp
                combined_df.sort_values('timestamp', inplace=True)
                # Remover duplicatas
                combined_df.drop_duplicates(subset=['timestamp'], keep='last', inplace=True)
                # Escrever de volta
                table = pa.Table.from_pandas(combined_df, schema=self.tick_schema)
                pq.write_table(table, file_path, compression='snappy')
            else:
                # Criar novo arquivo
                file_path.parent.mkdir(parents=True, exist_ok=True)
                table = pa.Table.from_pandas(group, schema=self.tick_schema)
                pq.write_table(table, file_path, compression='snappy')
    
    def _get_file_path(self, symbol: str, date: Union[datetime, str]) -> Path:
        """Gera caminho do arquivo para símbolo e data"""
        if isinstance(date, datetime):
            date_str = date.strftime('%Y%m%d')
        else:
            date_str = str(date).replace('-', '')
        
        year = date_str[:4]
        month = date_str[4:6]
        
        return self.base_path / symbol / year / month / f"{symbol}_{date_str}.parquet"
    
    async def read_ticks(self, 
                        symbol: str,
                        start_date: datetime,
                        end_date: datetime,
                        columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Lê ticks de um período"""
        try:
            # Executar leitura em thread
            loop = asyncio.get_event_loop()
            df = await loop.run_in_executor(
                self.executor,
                self._read_ticks_sync,
                symbol,
                start_date,
                end_date,
                columns
            )
            
            return df
            
        except Exception as e:
            logger.error(f"Erro ao ler ticks: {e}")
            return pd.DataFrame()
    
    def _read_ticks_sync(self,
                        symbol: str,
                        start_date: datetime,
                        end_date: datetime,
                        columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Lê ticks de forma síncrona"""
        all_dfs = []
        current_date = start_date.date()
        
        while current_date <= end_date.date():
            file_path = self._get_file_path(symbol, current_date)
            
            if file_path.exists():
                try:
                    # Ler arquivo
                    df = pd.read_parquet(file_path, columns=columns)
                    
                    # Filtrar por timestamp se necessário
                    if not df.empty:
                        df = df[(df['timestamp'] >= start_date) & 
                               (df['timestamp'] <= end_date)]
                        
                        if not df.empty:
                            all_dfs.append(df)
                            
                except Exception as e:
                    logger.error(f"Erro ao ler {file_path}: {e}")
            
            current_date += timedelta(days=1)
        
        if all_dfs:
            combined_df = pd.concat(all_dfs, ignore_index=True)
            combined_df.sort_values('timestamp', inplace=True)
            return combined_df
        
        return pd.DataFrame()
    
    async def get_tick_count(self, symbol: str, date: datetime) -> int:
        """Retorna contagem de ticks para uma data"""
        file_path = self._get_file_path(symbol, date)
        
        if file_path.exists():
            try:
                parquet_file = pq.ParquetFile(file_path)
                return parquet_file.metadata.num_rows
            except Exception as e:
                logger.error(f"Erro ao contar ticks: {e}")
        
        return 0
    
    async def get_storage_stats(self) -> Dict:
        """Retorna estatísticas de armazenamento"""
        stats = {
            'total_files': 0,
            'total_size_mb': 0,
            'symbols': {},
            'buffer_size': len(self.write_buffer)
        }
        
        try:
            # Percorrer diretórios
            for symbol_dir in self.base_path.iterdir():
                if symbol_dir.is_dir():
                    symbol = symbol_dir.name
                    symbol_stats = {
                        'files': 0,
                        'size_mb': 0,
                        'date_range': {'start': None, 'end': None}
                    }
                    
                    # Percorrer arquivos
                    for file_path in symbol_dir.rglob('*.parquet'):
                        symbol_stats['files'] += 1
                        size_mb = file_path.stat().st_size / (1024 * 1024)
                        symbol_stats['size_mb'] += size_mb
                        stats['total_size_mb'] += size_mb
                        stats['total_files'] += 1
                        
                        # Extrair data do nome do arquivo
                        date_str = file_path.stem.split('_')[-1]
                        if symbol_stats['date_range']['start'] is None:
                            symbol_stats['date_range']['start'] = date_str
                            symbol_stats['date_range']['end'] = date_str
                        else:
                            symbol_stats['date_range']['start'] = min(
                                symbol_stats['date_range']['start'], date_str
                            )
                            symbol_stats['date_range']['end'] = max(
                                symbol_stats['date_range']['end'], date_str
                            )
                    
                    stats['symbols'][symbol] = symbol_stats
            
            return stats
            
        except Exception as e:
            logger.error(f"Erro ao obter estatísticas: {e}")
            return stats
    
    async def cleanup_old_data(self, days_to_keep: int = 90):
        """Remove dados antigos"""
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        files_removed = 0
        space_freed_mb = 0
        
        try:
            for file_path in self.base_path.rglob('*.parquet'):
                # Extrair data do arquivo
                try:
                    date_str = file_path.stem.split('_')[-1]
                    file_date = datetime.strptime(date_str, '%Y%m%d')
                    
                    if file_date < cutoff_date:
                        size_mb = file_path.stat().st_size / (1024 * 1024)
                        file_path.unlink()
                        files_removed += 1
                        space_freed_mb += size_mb
                        
                except Exception as e:
                    logger.warning(f"Erro ao processar {file_path}: {e}")
            
            logger.info(f"Limpeza concluída: {files_removed} arquivos removidos, "
                       f"{space_freed_mb:.1f} MB liberados")
            
        except Exception as e:
            logger.error(f"Erro na limpeza: {e}")
    
    async def optimize_storage(self):
        """Otimiza arquivos de armazenamento"""
        logger.info("Iniciando otimização de armazenamento...")
        
        try:
            for file_path in self.base_path.rglob('*.parquet'):
                try:
                    # Ler arquivo
                    df = pd.read_parquet(file_path)
                    
                    # Otimizações
                    # 1. Remover duplicatas
                    original_size = len(df)
                    df.drop_duplicates(subset=['timestamp'], keep='last', inplace=True)
                    
                    # 2. Ordenar por timestamp
                    df.sort_values('timestamp', inplace=True)
                    
                    # 3. Reescrever se houve mudanças
                    if len(df) < original_size:
                        table = pa.Table.from_pandas(df, schema=self.tick_schema)
                        pq.write_table(table, file_path, compression='snappy')
                        logger.info(f"Otimizado {file_path.name}: "
                                   f"{original_size} -> {len(df)} ticks")
                        
                except Exception as e:
                    logger.error(f"Erro ao otimizar {file_path}: {e}")
            
            logger.info("Otimização concluída")
            
        except Exception as e:
            logger.error(f"Erro na otimização: {e}")
    
    async def close(self):
        """Fecha storage e faz flush final"""
        await self.flush()
        self.executor.shutdown(wait=True)
        logger.info("TickStorage fechado")