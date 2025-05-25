# utils/ntp_sync.py
"""Sincronização de tempo via NTP para precisão em trading"""
import asyncio
import ntplib
import time
import socket
from datetime import datetime, timedelta
from typing import Optional, List, Tuple
import statistics

from utils.logger import setup_logger

logger = setup_logger("ntp_sync")

class NTPSynchronizer:
    """Sincronizador de tempo com servidores NTP"""
    
    def __init__(self):
        self.ntp_client = ntplib.NTPClient()
        self.time_offset = 0.0  # Diferença em segundos
        self.last_sync = None
        self.sync_history = []
        self.max_history = 100
        
        # Servidores NTP confiáveis (Stratum 1)
        self.ntp_servers = [
            'time.nist.gov',
            'time.google.com',
            'time.cloudflare.com',
            'pool.ntp.org',
            '0.pool.ntp.org',
            '1.pool.ntp.org',
            'time1.google.com',
            'time2.google.com',
            'ntp1.stratum1.time.nl',
            'ntp2.stratum1.time.nl'
        ]
        
        self.timeout = 3  # segundos
        self.max_retries = 3
        
    async def sync(self) -> bool:
        """Sincroniza com servidor NTP"""
        try:
            # Tentar múltiplos servidores
            for server in self.ntp_servers:
                offset = await self._sync_with_server(server)
                
                if offset is not None:
                    self.time_offset = offset
                    self.last_sync = datetime.now()
                    
                    # Adicionar ao histórico
                    self.sync_history.append({
                        'timestamp': self.last_sync,
                        'server': server,
                        'offset': offset,
                        'latency': 0  # Será calculado
                    })
                    
                    # Manter apenas últimas N sincronizações
                    if len(self.sync_history) > self.max_history:
                        self.sync_history = self.sync_history[-self.max_history:]
                    
                    logger.info(f"Sincronizado com {server} - Offset: {offset*1000:.1f}ms")
                    return True
            
            logger.error("Falha ao sincronizar com todos os servidores NTP")
            return False
            
        except Exception as e:
            logger.error(f"Erro na sincronização NTP: {e}")
            return False
    
    async def _sync_with_server(self, server: str) -> Optional[float]:
        """Sincroniza com servidor específico"""
        for attempt in range(self.max_retries):
            try:
                # Executar em thread separada para não bloquear
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None,
                    self._ntp_request,
                    server
                )
                
                if response:
                    # Calcular offset
                    offset = response.offset
                    
                    # Validar offset (não deve ser muito grande)
                    if abs(offset) > 60:  # Mais de 1 minuto
                        logger.warning(f"Offset muito grande detectado: {offset}s")
                        return None
                    
                    return offset
                    
            except Exception as e:
                if attempt == self.max_retries - 1:
                    logger.debug(f"Falha ao conectar com {server}: {e}")
            
            await asyncio.sleep(0.5)
        
        return None
    
    def _ntp_request(self, server: str) -> Optional[ntplib.NTPStats]:
        """Faz requisição NTP (síncrono)"""
        try:
            response = self.ntp_client.request(
                server,
                version=3,
                timeout=self.timeout
            )
            return response
        except (ntplib.NTPException, socket.timeout, socket.gaierror):
            return None
    
    async def periodic_sync(self, interval_seconds: int = 300):
        """Sincronização periódica"""
        logger.info(f"Iniciando sincronização periódica a cada {interval_seconds}s")
        
        while True:
            try:
                # Sincronizar
                success = await self.sync()
                
                if not success and self.last_sync:
                    # Calcular tempo desde última sincronização
                    time_since_sync = (datetime.now() - self.last_sync).total_seconds()
                    
                    if time_since_sync > 3600:  # 1 hora
                        logger.warning("Mais de 1 hora sem sincronização NTP!")
                
                # Aguardar próxima sincronização
                await asyncio.sleep(interval_seconds)
                
            except asyncio.CancelledError:
                logger.info("Sincronização periódica cancelada")
                break
            except Exception as e:
                logger.error(f"Erro na sincronização periódica: {e}")
                await asyncio.sleep(60)  # Retry em 1 minuto
    
    def get_accurate_time(self) -> datetime:
        """Retorna tempo preciso considerando offset NTP"""
        local_time = datetime.now()
        
        if self.time_offset != 0:
            # Aplicar offset
            accurate_time = local_time + timedelta(seconds=self.time_offset)
            return accurate_time
        
        return local_time
    
    def get_accurate_timestamp(self) -> float:
        """Retorna timestamp preciso em segundos"""
        return time.time() + self.time_offset
    
    def get_accurate_timestamp_ms(self) -> int:
        """Retorna timestamp preciso em milissegundos"""
        return int((time.time() + self.time_offset) * 1000)
    
    def get_offset_ms(self) -> float:
        """Retorna offset atual em milissegundos"""
        return self.time_offset * 1000
    
    def get_sync_quality(self) -> dict:
        """Avalia qualidade da sincronização"""
        if not self.sync_history:
            return {
                'quality': 'unknown',
                'last_sync': None,
                'avg_offset_ms': 0,
                'stability': 0
            }
        
        # Calcular estatísticas dos últimos offsets
        recent_offsets = [s['offset'] for s in self.sync_history[-10:]]
        
        avg_offset = statistics.mean(recent_offsets)
        std_offset = statistics.stdev(recent_offsets) if len(recent_offsets) > 1 else 0
        
        # Determinar qualidade
        quality = 'excellent'
        if abs(avg_offset) > 0.010 or std_offset > 0.005:  # 10ms ou 5ms desvio
            quality = 'good'
        if abs(avg_offset) > 0.050 or std_offset > 0.020:  # 50ms ou 20ms desvio
            quality = 'fair'
        if abs(avg_offset) > 0.100 or std_offset > 0.050:  # 100ms ou 50ms desvio
            quality = 'poor'
        
        return {
            'quality': quality,
            'last_sync': self.last_sync,
            'avg_offset_ms': avg_offset * 1000,
            'std_offset_ms': std_offset * 1000,
            'stability': 1 - min(std_offset / 0.050, 1),  # 0-1, onde 1 é mais estável
            'sync_count': len(self.sync_history)
        }
    
    def is_synchronized(self) -> bool:
        """Verifica se está sincronizado"""
        if not self.last_sync:
            return False
        
        # Considerar dessincronizado após 10 minutos
        time_since_sync = (datetime.now() - self.last_sync).total_seconds()
        return time_since_sync < 600
    
    async def benchmark_servers(self) -> List[Tuple[str, float, float]]:
        """Testa latência de todos os servidores"""
        results = []
        
        for server in self.ntp_servers:
            try:
                start_time = time.time()
                offset = await self._sync_with_server(server)
                latency = time.time() - start_time
                
                if offset is not None:
                    results.append((server, latency * 1000, offset * 1000))
                    logger.info(f"{server}: Latência {latency*1000:.1f}ms, "
                               f"Offset {offset*1000:.1f}ms")
                
            except Exception as e:
                logger.debug(f"Erro ao testar {server}: {e}")
        
        # Ordenar por latência
        results.sort(key=lambda x: x[1])
        
        # Reorganizar servidores por performance
        if results:
            self.ntp_servers = [r[0] for r in results] + \
                               [s for s in self.ntp_servers if s not in [r[0] for r in results]]
        
        return results

# Singleton global
_ntp_synchronizer = None

def get_ntp_synchronizer() -> NTPSynchronizer:
    """Retorna instância singleton do sincronizador"""
    global _ntp_synchronizer
    if _ntp_synchronizer is None:
        _ntp_synchronizer = NTPSynchronizer()
    return _ntp_synchronizer

# Funções de conveniência
def get_accurate_time() -> datetime:
    """Retorna tempo preciso"""
    return get_ntp_synchronizer().get_accurate_time()

def get_accurate_timestamp_ms() -> int:
    """Retorna timestamp preciso em ms"""
    return get_ntp_synchronizer().get_accurate_timestamp_ms()

async def ensure_time_sync():
    """Garante que o tempo está sincronizado"""
    syncer = get_ntp_synchronizer()
    
    if not syncer.is_synchronized():
        logger.info("Sincronizando tempo com NTP...")
        success = await syncer.sync()
        
        if not success:
            logger.warning("Operando sem sincronização NTP!")
        
    return syncer.is_synchronized()