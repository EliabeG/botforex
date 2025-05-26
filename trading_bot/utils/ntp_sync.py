# utils/ntp_sync.py
"""Sincronização de tempo via NTP para precisão em trading."""
import asyncio
import ntplib # type: ignore # Adicionado type: ignore se ntplib não tiver stubs
import time
import socket
from datetime import datetime, timedelta, timezone # Adicionado timezone
from typing import Optional, List, Tuple, Dict, Any # Adicionado Dict, Any
import statistics

from utils.logger import setup_logger # setup_logger já está importado

# Usar um nome de logger específico para este módulo
ntp_logger = setup_logger("ntp_synchronizer") # Renomeado de logger para ntp_logger

class NTPSynchronizer:
    """Sincronizador de tempo com servidores NTP para garantir precisão.""" # Descrição ajustada

    def __init__(self, ntp_servers_list: Optional[List[str]] = None, # Permitir passar lista de servidores
                 max_history_entries: int = 100, # Renomeado
                 request_timeout_seconds: int = 3, # Renomeado
                 max_sync_retries: int = 3): # Renomeado
        self.ntp_client = ntplib.NTPClient()
        self.time_offset_seconds: float = 0.0  # Diferença em segundos, nome mais claro
        self.last_successful_sync_time: Optional[datetime] = None # Renomeado
        self.sync_history_log: List[Dict[str, Any]] = [] # Renomeado e tipado
        self.max_history_entries = max_history_entries # Usar o parâmetro

        # Servidores NTP confiáveis (Stratum 1 e pools populares)
        # Permitir override via construtor, senão usar lista padrão.
        self.ntp_server_pool: List[str] = ntp_servers_list or [ # Renomeado
            'time.nist.gov', 'time.google.com', 'time.cloudflare.com',
            'pool.ntp.org', '0.pool.ntp.org', '1.pool.ntp.org',
            '2.pool.ntp.org', '3.pool.ntp.org', # Adicionado mais do pool
            'time1.google.com', 'time2.google.com', 'time3.google.com', 'time4.google.com', # Mais do Google
            'ntp1.stratum1.time.nl', 'ntp2.stratum1.time.nl',
            # Adicionar servidores regionais pode ser útil se a latência for crítica
            # 'a.st1.ntp.br', 'b.st1.ntp.br', 'c.st1.ntp.br' # Exemplo Brasil
        ]

        self.ntp_request_timeout: int = request_timeout_seconds # Renomeado
        self.max_server_retries: int = max_sync_retries # Renomeado
        self._sync_lock = asyncio.Lock() # Lock para evitar syncs concorrentes

    async def sync_time_with_server_pool(self) -> bool: # Renomeado de sync
        """
        Tenta sincronizar o tempo com os servidores NTP da lista.
        Retorna True se a sincronização for bem-sucedida com pelo menos um servidor.
        """
        async with self._sync_lock: # Garantir que apenas uma corrotina de sync execute por vez
            ntp_logger.debug(f"Tentando sincronização NTP com {len(self.ntp_server_pool)} servidores.")
            # O benchmark pode reordenar os servidores, então iterar sobre a lista atual
            servers_to_try = list(self.ntp_server_pool) # Copiar para evitar modificação durante iteração se benchmark rodar
            
            for server_address in servers_to_try: # Renomeado server
                # Tentar obter offset do servidor específico
                offset_val = await self._get_offset_from_ntp_server(server_address) # Renomeado

                if offset_val is not None:
                    # Validar offset (não deve ser absurdamente grande)
                    if abs(offset_val) > 60:  # Mais de 1 minuto de offset é suspeito
                        ntp_logger.warning(f"Offset NTP muito grande detectado com {server_address}: {offset_val:.3f}s. Ignorando este servidor para o offset atual.")
                        # Continuar para o próximo servidor em vez de usar um offset potencialmente ruim
                        # ou poderia ter uma lógica para aceitar se múltiplos servidores concordarem com offset grande.
                        continue # Pular para o próximo servidor


                    self.time_offset_seconds = offset_val
                    self.last_successful_sync_time = datetime.now(timezone.utc) # Usar UTC

                    # Adicionar ao histórico de sincronização
                    # A latência real da requisição NTP pode ser estimada se `response.delay` for usado.
                    # O código original tinha 'latency': 0 e mencionava que seria calculado.
                    # Se _ntp_request_sync retornar response, podemos obter response.delay.
                    # Por ora, vamos manter 0 ou estimar de forma simples.
                    estimated_latency_ms = 0 # Placeholder
                    # Se _ntp_request_sync retornasse o NTPStats:
                    # estimated_latency_ms = response.delay * 1000 if response else 0

                    self.sync_history_log.append({
                        'timestamp_utc': self.last_successful_sync_time.isoformat(), # Renomeado e formato ISO
                        'ntp_server': server_address, # Renomeado
                        'time_offset_seconds': offset_val, # Renomeado
                        'estimated_latency_ms': estimated_latency_ms
                    })

                    # Manter histórico dentro do limite de tamanho
                    if len(self.sync_history_log) > self.max_history_entries:
                        self.sync_history_log = self.sync_history_log[-self.max_history_entries:]

                    ntp_logger.info(f"Sincronizado com {server_address}. Offset: {offset_val*1000:.1f}ms.")
                    return True # Sincronização bem-sucedida

            ntp_logger.error(f"Falha ao sincronizar com todos os {len(servers_to_try)} servidores NTP listados.")
            return False
        # Fim do bloco with self._sync_lock

    async def _get_offset_from_ntp_server(self, server_address: str) -> Optional[float]: # Renomeado
        """Tenta obter o offset de tempo de um servidor NTP específico com retries."""
        for attempt in range(self.max_server_retries):
            try:
                # Executar a requisição NTP bloqueante em um thread pool executor
                loop = asyncio.get_event_loop()
                ntp_response: Optional[ntplib.NTPStats] = await loop.run_in_executor( # Adicionada tipagem
                    None,  # Default ThreadPoolExecutor
                    self._ntp_request_sync, # Renomeado
                    server_address
                )

                if ntp_response:
                    # O offset é a diferença que precisa ser ADICIONADA ao tempo local
                    # para obter o tempo NTP.
                    # response.offset já é calculado pela biblioteca ntplib
                    # como (t_server - t_local) ajustado por round-trip delay.
                    return ntp_response.offset # Este é o valor correto a ser usado

            except ntplib.NTPException as e_ntp_lib: # Renomeado
                ntp_logger.debug(f"NTPException com {server_address} (tentativa {attempt + 1}/{self.max_server_retries}): {e_ntp_lib}")
            except socket.timeout:
                ntp_logger.debug(f"Timeout ao conectar com {server_address} (tentativa {attempt + 1}/{self.max_server_retries}).")
            except socket.gaierror: # Erro de resolução de nome
                ntp_logger.warning(f"Erro ao resolver nome do servidor NTP {server_address}. Verifique a conectividade/DNS.")
                return None # Não tentar novamente para este servidor se o nome não resolve
            except Exception as e_sync_server: # Renomeado
                ntp_logger.error(f"Erro inesperado ao sincronizar com {server_address} (tentativa {attempt + 1}): {e_sync_server}", exc_info=False) # exc_info=False para não poluir com tracebacks de cada tentativa

            if attempt < self.max_server_retries - 1: # Se não for a última tentativa
                await asyncio.sleep(0.5 * (attempt + 1))  # Backoff simples
        
        ntp_logger.warning(f"Falha ao obter offset de {server_address} após {self.max_server_retries} tentativas.")
        return None


    def _ntp_request_sync(self, server_address: str) -> Optional[ntplib.NTPStats]: # Renomeado
        """Faz a requisição NTP (operação síncrona/bloqueante)."""
        # Este método é executado em um thread separado por run_in_executor.
        try:
            # A biblioteca ntplib pode ter sua própria lógica de timeout na request.
            # O timeout aqui é para a chamada de request.
            response = self.ntp_client.request(
                server_address,
                version=3, # NTPv3 é comum
                timeout=self.ntp_request_timeout
            )
            return response # Retorna o objeto NTPStats completo
        except (ntplib.NTPException, socket.timeout, socket.gaierror) as e_req: # Capturar gaierror aqui também
            # ntp_logger.debug(f"Falha na requisição NTP para {server_address}: {type(e_req).__name__} - {e_req}") # Logado no chamador
            raise # Relançar para ser pego por _get_offset_from_ntp_server
        except Exception as e_req_generic:
            ntp_logger.error(f"Erro genérico em _ntp_request_sync para {server_address}: {e_req_generic}")
            raise NTPException(f"Erro genérico na requisição NTP: {e_req_generic}") from e_req_generic



    async def start_periodic_sync(self, interval_seconds: int = 300): # Renomeado de periodic_sync
        """Inicia a tarefa de sincronização NTP periódica em background."""
        if interval_seconds <= 0:
            ntp_logger.warning("Intervalo de sincronização periódica inválido. Desabilitando.")
            return

        ntp_logger.info(f"Iniciando sincronização NTP periódica a cada {interval_seconds} segundos.")
        # É melhor que esta função retorne a Task para que possa ser gerenciada (ex: cancelada)
        # ou que a própria classe gerencie sua task interna.
        # self._periodic_sync_task = asyncio.create_task(self._run_sync_loop(interval_seconds))
        # await self._periodic_sync_task # Se quiser esperar aqui (não é o caso para background)
        # Se for para rodar como uma task de background, o chamador deve fazer o create_task.
        # A implementação original era um loop infinito, o que é correto para uma task.

        while True: # Este loop deve ser gerenciado por quem chama start_periodic_sync
            try:
                sync_successful = await self.sync_time_with_server_pool() # Renomeado

                if not sync_successful and self.last_successful_sync_time:
                    time_since_last_sync_sec = (datetime.now(timezone.utc) - self.last_successful_sync_time).total_seconds() # Usar UTC
                    if time_since_last_sync_sec > 3600:  # 1 hora sem sincronização
                        ntp_logger.warning(f"ALERTA: Mais de {time_since_last_sync_sec/3600:.1f} horas sem sincronização NTP bem-sucedida!")
                elif sync_successful:
                     ntp_logger.debug(f"Sincronização periódica NTP bem-sucedida. Próxima em {interval_seconds}s.")


                await asyncio.sleep(interval_seconds)

            except asyncio.CancelledError:
                ntp_logger.info("Tarefa de sincronização NTP periódica cancelada.")
                break # Sair do loop se cancelado
            except Exception as e_periodic: # Renomeado
                ntp_logger.exception(f"Erro inesperado na sincronização periódica NTP:") # Usar logger.exception
                await asyncio.sleep(60)  # Esperar 1 minuto antes de tentar novamente em caso de erro


    def get_current_accurate_time(self) -> datetime: # Renomeado
        """Retorna o tempo local atual ajustado pelo offset NTP. Sempre ciente do fuso horário (UTC)."""
        # Usar time.time() para tempo local em segundos desde epoch, depois converter para datetime UTC
        # e aplicar offset. Isso evita ambiguidades de fuso horário local.
        current_utc_timestamp = time.time() # Segundos desde epoch, é inerentemente UTC-like
        
        # Aplicar offset para obter o tempo NTP "real"
        accurate_utc_timestamp = current_utc_timestamp + self.time_offset_seconds
        
        # Converter para datetime object UTC
        accurate_time_utc = datetime.fromtimestamp(accurate_utc_timestamp, tz=timezone.utc)
        return accurate_time_utc


    def get_current_accurate_timestamp_seconds(self) -> float: # Renomeado
        """Retorna o timestamp Unix preciso (em segundos) considerando o offset NTP."""
        return time.time() + self.time_offset_seconds


    def get_current_accurate_timestamp_ms(self) -> int: # Renomeado
        """Retorna o timestamp Unix preciso (em milissegundos) considerando o offset NTP."""
        return int((time.time() + self.time_offset_seconds) * 1000)


    def get_current_offset_ms(self) -> float: # Renomeado
        """Retorna o offset de tempo atual em milissegundos."""
        return self.time_offset_seconds * 1000


    def get_current_sync_quality_stats(self) -> Dict[str, Any]: # Renomeado
        """Avalia e retorna estatísticas sobre a qualidade da sincronização NTP."""
        if not self.sync_history_log:
            return {
                'quality_rating': 'Desconhecida (sem histórico)', # Renomeado
                'last_sync_time_utc': None, # Renomeado
                'avg_offset_recent_ms': 0.0,
                'offset_stability_std_dev_ms': 0.0, # Renomeado
                'sync_attempts_in_history': 0 # Renomeado
            }

        # Calcular estatísticas dos últimos N offsets no histórico
        # O original usa os últimos 10, o que é razoável.
        recent_syncs_for_stats = self.sync_history_log[-10:] # Renomeado
        recent_offsets_sec = [s_entry['time_offset_seconds'] for s_entry in recent_syncs_for_stats if 'time_offset_seconds' in s_entry] # Renomeado

        if not recent_offsets_sec: # Se não houver offsets válidos nos recentes
             return {
                'quality_rating': 'Desconhecida (sem offsets recentes)',
                'last_sync_time_utc': self.last_successful_sync_time.isoformat() if self.last_successful_sync_time else None,
                'avg_offset_recent_ms': 0.0,
                'offset_stability_std_dev_ms': 0.0,
                'sync_attempts_in_history': len(self.sync_history_log)
            }


        avg_offset_val_sec = statistics.mean(recent_offsets_sec) # Renomeado
        std_dev_offset_sec = statistics.stdev(recent_offsets_sec) if len(recent_offsets_sec) > 1 else 0.0 # Renomeado

        # Determinar qualidade com base em limiares (estes podem ser configuráveis)
        # Offsets em milissegundos para comparação
        avg_offset_ms_abs = abs(avg_offset_val_sec * 1000)
        std_dev_offset_ms = std_dev_offset_sec * 1000

        quality_rating_str: str # Renomeado
        if avg_offset_ms_abs <= 10 and std_dev_offset_ms <= 5: quality_rating_str = 'Excelente'
        elif avg_offset_ms_abs <= 50 and std_dev_offset_ms <= 20: quality_rating_str = 'Boa'
        elif avg_offset_ms_abs <= 100 and std_dev_offset_ms <= 50: quality_rating_str = 'Razoável'
        else: quality_rating_str = 'Ruim'

        # Estabilidade (0-1, onde 1 é mais estável, baseado no desvio padrão)
        # Normalizar std_dev_offset_ms em relação a um valor "ruim" (ex: 50ms)
        stability_score = 1.0 - min(std_dev_offset_ms / 50.0, 1.0) # Renomeado

        return {
            'quality_rating': quality_rating_str,
            'last_sync_time_utc': self.last_successful_sync_time.isoformat() if self.last_successful_sync_time else None,
            'avg_offset_recent_ms': round(avg_offset_val_sec * 1000, 3),
            'offset_stability_std_dev_ms': round(std_dev_offset_ms, 3),
            'stability_score_0_to_1': round(stability_score, 3),
            'sync_events_in_history': len(self.sync_history_log), # Renomeado
            'recent_syncs_analyzed': len(recent_syncs_for_stats)
        }


    def is_currently_synchronized(self, max_age_seconds: int = 600) -> bool: # Renomeado e com arg
        """Verifica se o sistema está considerado sincronizado (última sync recente)."""
        if not self.last_successful_sync_time:
            return False

        time_since_last_sync = (datetime.now(timezone.utc) - self.last_successful_sync_time).total_seconds() # Usar UTC
        return time_since_last_sync < max_age_seconds


    async def benchmark_ntp_servers(self) -> List[Tuple[str, float, float]]: # Renomeado
        """Testa a latência e o offset de todos os servidores NTP configurados e os reordena."""
        ntp_logger.info("Iniciando benchmark de servidores NTP...")
        benchmark_results: List[Tuple[str, float, float]] = [] # (server, latency_ms, offset_ms)

        # Usar uma cópia da lista de servidores para evitar problemas se for modificada
        servers_to_benchmark = list(self.ntp_server_pool)

        for server_addr_bench in servers_to_benchmark: # Renomeado
            try:
                # Medir tempo total da tentativa de sincronização
                start_benchmark_time = time.perf_counter() # Usar perf_counter para medição de tempo
                # _get_offset_from_ntp_server já lida com retries e retorna o offset ou None
                offset_val_bench = await self._get_offset_from_ntp_server(server_addr_bench) # Renomeado
                latency_total_seconds = time.perf_counter() - start_benchmark_time # Em segundos

                if offset_val_bench is not None:
                    # A latência aqui é o tempo total da chamada _get_offset_from_ntp_server.
                    # A "latência NTP" real (round-trip delay / 2) é calculada por ntplib.
                    # Se _ntp_request_sync retornasse NTPStats, poderíamos usar response.delay.
                    # Por enquanto, latency_total_seconds é uma proxy.
                    latency_ms_proxy = latency_total_seconds * 1000
                    offset_ms_val = offset_val_bench * 1000 # Renomeado

                    benchmark_results.append((server_addr_bench, latency_ms_proxy, offset_ms_val))
                    ntp_logger.info(f"Benchmark {server_addr_bench}: Latência Estimada {latency_ms_proxy:.1f}ms, Offset {offset_ms_val:.1f}ms")
                else:
                    ntp_logger.warning(f"Benchmark {server_addr_bench}: Falha ao obter offset.")

            except Exception as e_bench: # Renomeado
                ntp_logger.error(f"Erro ao testar servidor NTP {server_addr_bench}: {e_bench}")

        if benchmark_results:
            # Ordenar resultados por latência (menor primeiro) e depois por menor offset absoluto
            benchmark_results.sort(key=lambda x_item: (x_item[1], abs(x_item[2]))) # Renomeado x para x_item

            # Reorganizar a lista de servidores principal com os mais performáticos primeiro
            new_server_order = [res[0] for res in benchmark_results] # Renomeado
            # Adicionar servidores que podem ter falhado no benchmark ao final da lista
            original_set = set(self.ntp_server_pool)
            benchmarked_set = set(new_server_order)
            failed_or_missed_servers = list(original_set - benchmarked_set)
            
            self.ntp_server_pool = new_server_order + failed_or_missed_servers
            ntp_logger.info(f"Servidores NTP reordenados por performance. Nova ordem: {self.ntp_server_pool[:5]}...") # Logar os 5 primeiros
        
        ntp_logger.info("Benchmark de servidores NTP concluído.")
        return benchmark_results

# === Singleton Global e Funções de Conveniência ===

_global_ntp_synchronizer_instance: Optional[NTPSynchronizer] = None # Renomeado
_ntp_init_lock = asyncio.Lock() # Lock para inicialização do singleton

async def get_ntp_synchronizer_instance() -> NTPSynchronizer: # Renomeado
    """Retorna a instância singleton do NTPSynchronizer, inicializando se necessário."""
    global _global_ntp_synchronizer_instance
    if _global_ntp_synchronizer_instance is None:
        async with _ntp_init_lock: # Garantir que apenas uma corrotina inicialize
            if _global_ntp_synchronizer_instance is None: # Dupla verificação (double-checked locking)
                _global_ntp_synchronizer_instance = NTPSynchronizer()
                # Considerar uma sincronização inicial aqui se for crítico ter o offset imediatamente
                # await _global_ntp_synchronizer_instance.sync_time_with_server_pool()
                ntp_logger.info("Instância singleton de NTPSynchronizer criada.")
    return _global_ntp_synchronizer_instance

async def get_accurate_utc_time() -> datetime: # Renomeado
    """Retorna o tempo UTC preciso, ajustado pelo NTP."""
    synchronizer = await get_ntp_synchronizer_instance()
    return synchronizer.get_current_accurate_time()

async def get_accurate_timestamp_ms_utc() -> int: # Renomeado
    """Retorna o timestamp Unix preciso em milissegundos (UTC), ajustado pelo NTP."""
    synchronizer = await get_ntp_synchronizer_instance()
    return synchronizer.get_current_accurate_timestamp_ms()

async def ensure_initial_ntp_sync(timeout_seconds: int = 10): # Renomeado e com timeout
    """
    Garante que uma tentativa de sincronização NTP seja feita na inicialização.
    Retorna True se sincronizado, False caso contrário.
    """
    synchronizer = await get_ntp_synchronizer_instance()
    if not synchronizer.is_currently_synchronized(max_age_seconds=3600): # Se não sincronizado na última hora
        ntp_logger.info("Tempo não sincronizado com NTP ou sincronização antiga. Tentando sincronizar agora...")
        try:
            # Tentar sincronizar com timeout para não bloquear a inicialização indefinidamente
            sync_success = await asyncio.wait_for(
                synchronizer.sync_time_with_server_pool(),
                timeout=timeout_seconds
            )
            if not sync_success:
                ntp_logger.warning("Falha na sincronização NTP inicial. O sistema operará com o relógio local.")
            return sync_success
        except asyncio.TimeoutError:
            ntp_logger.error(f"Timeout ({timeout_seconds}s) durante a sincronização NTP inicial.")
            return False
        except Exception as e_ensure_sync: # Renomeado
            ntp_logger.exception(f"Erro durante ensure_initial_ntp_sync: {e_ensure_sync}")
            return False
    ntp_logger.info("NTP já está sincronizado.")
    return True