# monitoring/metrics_collector.py
"""Coletor de métricas para Prometheus"""
from prometheus_client import Counter, Histogram, Gauge, Summary, Info
from prometheus_client import start_http_server, generate_latest, REGISTRY # Adicionado REGISTRY
import asyncio
from typing import Dict, Optional, Any # Adicionado Any
from datetime import datetime, timezone # Adicionado timezone
import psutil
import platform
import os # Para obter CPU count

from utils.logger import setup_logger
# from config.settings import CONFIG # Se precisar de alguma config global aqui

logger = setup_logger("metrics_collector")

class MetricsCollector:
    """Coletor de métricas para monitoramento via Prometheus"""

    def __init__(self, port: int = 9090, registry=REGISTRY): # Permitir passar um registry customizado
        self.port = port
        self.registry = registry # Usar o registry fornecido ou o global
        self.server_started = False
        self._system_metrics_task: Optional[asyncio.Task] = None # Para rastrear a tarefa

        # === Métricas de Sistema ===
        # Usar o registry para criar métricas evita problemas se a classe for instanciada múltiplas vezes
        self.system_info = Info('trading_bot_info', 'Informações do sistema', registry=self.registry)
        self.cpu_usage = Gauge('cpu_usage_percent', 'Uso de CPU em percentual', registry=self.registry)
        self.memory_usage = Gauge('memory_usage_percent', 'Uso de memória em percentual', registry=self.registry)
        self.memory_used_mb = Gauge('memory_used_mb', 'Memória usada em MB', registry=self.registry)
        self.process_cpu_usage = Gauge('process_cpu_usage_percent', 'Uso de CPU do processo do bot em percentual', registry=self.registry)
        self.process_memory_mb = Gauge('process_memory_mb', 'Memória usada pelo processo do bot em MB', registry=self.registry)
        self.thread_count = Gauge('process_thread_count', 'Número de threads do processo do bot', registry=self.registry)


        # === Métricas de Conexão ===
        self.connection_status = Gauge('connection_status', 'Status da conexão (1=conectado, 0=desconectado)', ['service'], registry=self.registry)
        self.websocket_latency_ms = Histogram('websocket_latency_ms', 'Latência WebSocket em ms',
                                          buckets=(1, 5, 10, 25, 50, 100, 250, 500, 1000, float('inf')), registry=self.registry) # Adicionado inf
        self.api_requests_total = Counter('api_requests_total', 'Total de requisições API', ['method', 'endpoint', 'status_code'], registry=self.registry) # Adicionado status_code
        # api_errors_total removido, usar api_requests_total com status_code >= 400

        # === Métricas de Trading ===
        self.trades_total = Counter('trades_total', 'Total de trades executados (fills)', ['strategy', 'symbol', 'side', 'result'], registry=self.registry) # Adicionado symbol
        self.orders_total = Counter('orders_total', 'Total de ordens criadas/atualizadas', ['strategy', 'symbol', 'type', 'status'], registry=self.registry) # Adicionado symbol
        self.positions_open_count = Gauge('positions_open_count', 'Número de posições abertas', ['strategy', 'symbol'], registry=self.registry) # Renomeado e adicionado symbol
        self.positions_open_volume = Gauge('positions_open_volume_lots', 'Volume total de posições abertas em lotes', ['strategy', 'symbol', 'side'], registry=self.registry) # Adicionado
        self.order_execution_time_ms = Histogram('order_execution_time_ms', 'Tempo de execução de ordem em ms (submissão ao fill)',
                                            buckets=(10, 25, 50, 100, 250, 500, 1000, 2500, 5000, float('inf')), registry=self.registry)
        self.slippage_pips = Histogram('order_slippage_pips', 'Slippage de ordens em pips',
                                    buckets=(-2, -1, -0.5, 0, 0.5, 1, 2, 5, float('inf')), registry=self.registry) # Buckets para slippage


        # === Métricas Financeiras ===
        self.account_balance = Gauge('account_balance_usd', 'Saldo da conta em USD', registry=self.registry) # Adicionado _usd
        self.account_equity = Gauge('account_equity_usd', 'Equity da conta em USD', registry=self.registry) # Adicionado _usd
        self.daily_pnl_usd = Gauge('daily_pnl_usd', 'PnL diário em USD', registry=self.registry) # Adicionado _usd
        self.daily_pnl_percent = Gauge('daily_pnl_percent', 'PnL diário em percentual', registry=self.registry)
        self.drawdown_current_percent = Gauge('drawdown_current_percent', 'Drawdown atual em percentual', registry=self.registry)
        self.margin_level_percent = Gauge('margin_level_percent', 'Nível de margem em percentual', registry=self.registry)
        self.margin_used_usd = Gauge('margin_used_usd', 'Margem utilizada em USD', registry=self.registry) # Adicionado
        self.free_margin_usd = Gauge('free_margin_usd', 'Margem livre em USD', registry=self.registry) # Adicionado


        # === Métricas de Estratégia ===
        self.strategy_score = Gauge('strategy_score_ratio', 'Score da estratégia (0-1)', ['strategy'], registry=self.registry) # Adicionado _ratio
        self.strategy_sharpe_ratio = Gauge('strategy_sharpe_ratio', 'Sharpe ratio da estratégia', ['strategy'], registry=self.registry)
        self.strategy_win_rate_percent = Gauge('strategy_win_rate_percent', 'Win rate da estratégia (%)', ['strategy'], registry=self.registry) # Adicionado _percent
        self.strategy_active_status = Gauge('strategy_active_status', 'Estratégia ativa (1) ou inativa (0)', ['strategy'], registry=self.registry) # Renomeado
        self.strategy_signal_count_total = Counter('strategy_signal_count_total', 'Contagem de sinais gerados por estratégia', ['strategy', 'side'], registry=self.registry) # Adicionado

        # === Métricas de Regime de Mercado ===
        self.market_regime_active = Gauge('market_regime_active', 'Regime de mercado atual (1 para ativo, 0 para outros)', ['regime_type'], registry=self.registry) # Renomeado e ajustado
        self.regime_confidence_ratio = Gauge('regime_confidence_ratio', 'Confiança na detecção de regime (0-1)', registry=self.registry) # Adicionado _ratio
        self.market_volatility_value = Gauge('market_volatility_value', 'Volatilidade atual do mercado (ex: ATR %)', registry=self.registry) # Renomeado
        self.market_spread_average_pips = Gauge('market_spread_average_pips', 'Spread médio do mercado em pips', ['symbol'], registry=self.registry) # Adicionado symbol

        # === Métricas de Risco ===
        self.risk_overall_score = Gauge('risk_overall_score_ratio', 'Score de risco geral (0-1)', registry=self.registry) # Renomeado e _ratio
        self.circuit_breaker_status_code = Gauge('circuit_breaker_status_code', 'Status do circuit breaker (0=closed, 1=open, 2=half-open)', registry=self.registry) # Renomeado
        self.risk_consecutive_losses_count = Gauge('risk_consecutive_losses_count', 'Perdas consecutivas atuais', registry=self.registry) # Renomeado
        self.risk_events_total = Counter('risk_events_total', 'Total de eventos de risco', ['type', 'severity'], registry=self.registry)

        # === Métricas de Performance Interna ===
        self.tick_processing_time_ms = Summary('tick_processing_time_ms', 'Tempo de processamento de tick em ms', registry=self.registry)
        self.optimization_duration_seconds = Histogram('optimization_duration_seconds', 'Duração da otimização em segundos',
                                             buckets=(60, 300, 600, 1800, 3600, 7200, float('inf')), registry=self.registry)
        self.backtest_speed_ticks_per_second = Gauge('backtest_speed_ticks_per_second', 'Velocidade do backtest em ticks/s', registry=self.registry)

        # === Métricas de Dados ===
        self.ticks_processed_total = Counter('ticks_processed_total', 'Total de ticks processados', ['symbol'], registry=self.registry)
        self.data_storage_size_mb = Gauge('data_storage_size_mb', 'Tamanho do armazenamento de dados em MB (ex: Parquet)', ['storage_type'], registry=self.registry) # Adicionado storage_type
        self.cache_hit_ratio = Gauge('cache_hit_ratio', 'Taxa de acerto do cache', ['cache_name'], registry=self.registry) # Adicionado cache_name
        self.data_latency_ms = Histogram('data_latency_ms', 'Latência de acesso a dados em ms (ex: DB, Redis)', ['source'], # Adicionado source
                                    buckets=(1, 5, 10, 25, 50, 100, 250, 500, float('inf')), registry=self.registry)

        self._bot_process = psutil.Process(os.getpid()) # Para métricas do processo
        self._register_system_info()

    def _register_system_info(self):
        """Registra informações estáticas do sistema e do bot."""
        # Informações do sistema (plataforma, python, etc.)
        self.system_info.info({
            'bot_version': getattr(CONFIG, 'BOT_VERSION', '1.0.0'), # Exemplo, viria de CONFIG
            'python_version': platform.python_version(),
            'platform_details': platform.platform(), # Renomeado de platform
            'processor_type': platform.processor(), # Renomeado de processor
            'hostname': platform.node(),
            'trading_symbol_main': CONFIG.SYMBOL # Adicionar símbolo principal
        })

    def start(self):
        """Inicia servidor HTTP para métricas e coleta de métricas de sistema."""
        if not self.server_started:
            try:
                start_http_server(self.port, registry=self.registry)
                self.server_started = True
                logger.info(f"Servidor de métricas Prometheus iniciado na porta {self.port}")

                # Iniciar coleta de métricas do sistema em uma tarefa separada
                if self._system_metrics_task is None or self._system_metrics_task.done():
                    self._system_metrics_task = asyncio.create_task(self._collect_system_metrics_periodically()) # Renomeado

            except OSError as e: # Capturar erro específico de porta em uso
                if e.errno == 98: # Address already in use
                    logger.error(f"Porta {self.port} já está em uso. Servidor de métricas não iniciado.")
                else:
                    logger.exception("Erro ao iniciar servidor de métricas Prometheus:")
            except Exception as e_start: # Renomeado
                logger.exception("Erro geral ao iniciar servidor de métricas Prometheus:")


    async def _collect_system_metrics_periodically(self): # Renomeado
        """Coleta métricas do sistema e do processo periodicamente."""
        logger.info("Coletor de métricas de sistema iniciado.")
        while self.server_started: # Continuar apenas se o servidor estiver rodando
            try:
                # CPU Global
                self.cpu_usage.set(psutil.cpu_percent(interval=None)) # Non-blocking, usar após primeira chamada com interval

                # Memória Global
                memory_info = psutil.virtual_memory() # Renomeado
                self.memory_usage.set(memory_info.percent)
                self.memory_used_mb.set(memory_info.used / (1024 * 1024))

                # Métricas do Processo do Bot
                self.process_cpu_usage.set(self._bot_process.cpu_percent(interval=None)) # Non-blocking
                self.process_memory_mb.set(self._bot_process.memory_info().rss / (1024 * 1024)) # RSS
                self.thread_count.set(self._bot_process.num_threads())


                await asyncio.sleep(10)  # Atualizar a cada 10 segundos

            except psutil.NoSuchProcess:
                logger.warning("Processo do bot não encontrado para coleta de métricas. Encerrando coleta de sistema.")
                break # Sair do loop se o processo morrer
            except asyncio.CancelledError:
                logger.info("Coleta de métricas de sistema cancelada.")
                break
            except Exception as e_collect: # Renomeado
                logger.error(f"Erro ao coletar métricas do sistema: {e_collect}")
                await asyncio.sleep(60) # Esperar mais em caso de erro
        logger.info("Coletor de métricas de sistema parado.")


    # === Métodos de Atualização de Métricas (mantendo os nomes originais) ===

    def update_connection_status(self, service: str, connected: bool): # Modificado para um serviço por vez
        """Atualiza status de uma conexão específica."""
        self.connection_status.labels(service=service).set(1 if connected else 0)

    def record_websocket_latency(self, latency_ms: float):
        self.websocket_latency.observe(latency_ms)

    def record_api_request(self, method: str, endpoint: str, status_code: int): # Adicionado status_code
        """Registra requisição API e seu status."""
        self.api_requests_total.labels(method=method, endpoint=endpoint, status_code=status_code).inc()
        # api_errors_total foi removido, pois pode ser derivado desta métrica (status_code >= 400)

    def record_trade(self, strategy: str, symbol:str, side: str, pnl: float): # Adicionado symbol
        """Registra trade executado (fill)."""
        result = 'win' if pnl > 0 else ('loss' if pnl < 0 else 'breakeven')
        self.trades_total.labels(strategy=strategy, symbol=symbol, side=side, result=result).inc()

    def record_order(self, strategy: str, symbol:str, order_type: str, status: str): # Adicionado symbol
        """Registra evento de ordem (criada, cancelada, rejeitada, etc.)."""
        self.orders_total.labels(
            strategy=strategy,
            symbol=symbol, # Adicionado symbol
            type=order_type,
            status=status
        ).inc()

    def update_open_positions_metrics(self, strategy: str, symbol: str, side: str, count: int, volume_lots: float): # Renomeado e mais granular
        """Atualiza contagem e volume de posições abertas."""
        self.positions_open_count.labels(strategy=strategy, symbol=symbol).set(count)
        self.positions_open_volume.labels(strategy=strategy, symbol=symbol, side=side).set(volume_lots)


    def record_order_execution_time(self, execution_time_ms: float):
        self.order_execution_time.observe(execution_time_ms)

    def record_order_slippage(self, slippage_in_pips: float): # Novo método para slippage
        self.slippage_pips.observe(slippage_in_pips)


    def update_account_metrics(self, balance: float, equity: float,
                             daily_pnl: float, daily_pnl_pct: float,
                             drawdown_pct: float, margin_level_pct: float, # Renomeados para clareza de unidade
                             margin_used: float, free_margin: float): # Adicionados
        """Atualiza métricas da conta."""
        self.account_balance.set(balance)
        self.account_equity.set(equity)
        self.daily_pnl_usd.set(daily_pnl)
        self.daily_pnl_percent.set(daily_pnl_pct * 100) # Multiplicar por 100 se for ratio
        self.drawdown_current_percent.set(drawdown_pct * 100)
        self.margin_level_percent.set(margin_level_pct * 100)
        self.margin_used_usd.set(margin_used)
        self.free_margin_usd.set(free_margin)


    def update_strategy_metrics(self, strategy: str, score: float, # Score é ratio 0-1
                              sharpe: float, win_rate_pct: float, active: bool): # Win rate em %
        """Atualiza métricas de estratégia."""
        self.strategy_score.labels(strategy=strategy).set(score)
        self.strategy_sharpe_ratio.labels(strategy=strategy).set(sharpe)
        self.strategy_win_rate_percent.labels(strategy=strategy).set(win_rate_pct)
        self.strategy_active_status.labels(strategy=strategy).set(1 if active else 0)

    def record_strategy_signal(self, strategy:str, side:str): # Novo método
        """Registra um sinal gerado por uma estratégia."""
        self.strategy_signal_count_total.labels(strategy=strategy, side=side).inc()


    def update_market_regime(self, active_regime_type: str, confidence_ratio: float): # Renomeado e ajustado
        """Atualiza regime de mercado ativo."""
        for r_type in MarketRegime.__dict__.values(): # Assumindo que MarketRegime tem os tipos como atributos
            if isinstance(r_type, str): # Checar se é string (valor do regime)
                 self.market_regime_active.labels(regime_type=r_type).set(1 if r_type == active_regime_type else 0)
        self.regime_confidence_ratio.set(confidence_ratio)


    def update_market_conditions(self, symbol: str, volatility_value: float, avg_spread_pips: float): # Renomeado, adicionado symbol
        """Atualiza condições de mercado."""
        self.market_volatility_value.set(volatility_value) # Esta métrica é global ou por símbolo? Se por símbolo, adicionar label.
        self.market_spread_average_pips.labels(symbol=symbol).set(avg_spread_pips)


    def update_risk_metrics(self, risk_score_ratio: float, circuit_breaker_code: int, # Renomeado
                          consecutive_losses_val: int): # Renomeado
        """Atualiza métricas de risco."""
        self.risk_overall_score.set(risk_score_ratio)
        self.circuit_breaker_status_code.set(circuit_breaker_code)
        self.risk_consecutive_losses_count.set(consecutive_losses_val)


    def record_risk_event(self, event_type: str, severity: str):
        self.risk_events_total.labels(type=event_type, severity=severity).inc()

    def record_tick_processing_time(self, processing_time_ms: float):
        self.tick_processing_time.observe(processing_time_ms)

    def record_optimization_duration(self, duration_seconds: float):
        self.optimization_duration.observe(duration_seconds)

    def update_backtest_speed(self, ticks_per_second: float):
        self.backtest_speed.set(ticks_per_second)

    def record_tick_processed(self, symbol: str):
        self.ticks_processed_total.labels(symbol=symbol).inc()

    def update_data_metrics(self, storage_type: str, storage_size_mb: float, cache_name: str, cache_hit_rt: float): # Renomeado e adicionado params
        """Atualiza métricas de dados."""
        self.data_storage_size_mb.labels(storage_type=storage_type).set(storage_size_mb)
        self.cache_hit_ratio.labels(cache_name=cache_name).set(cache_hit_rt)


    def record_data_latency(self, source:str, latency_ms: float): # Adicionado source
        """Registra latência de acesso a dados."""
        self.data_latency_ms.labels(source=source).observe(latency_ms)

    async def stop(self): # Transformado em async para poder cancelar a task
        """Para o servidor de métricas e a coleta de métricas de sistema."""
        logger.info("Parando MetricsCollector...")
        self.server_started = False # Sinaliza para a task de coleta parar

        if self._system_metrics_task and not self._system_metrics_task.done():
            self._system_metrics_task.cancel()
            try:
                await self._system_metrics_task
            except asyncio.CancelledError:
                logger.info("Tarefa de coleta de métricas de sistema cancelada durante o stop.")
            except Exception as e_stop_task: # Renomeado
                 logger.error(f"Erro ao aguardar cancelamento da tarefa de métricas de sistema: {e_stop_task}")


        # Prometheus não fornece método direto para parar o servidor HTTP.
        # Ele roda em um thread separado. O término do processo principal geralmente o encerra.
        # Se fosse um servidor asyncio (ex: aiohttp), poderíamos pará-lo aqui.
        logger.info("Servidor de métricas Prometheus (se iniciado) será parado com o processo principal.")


    def get_metrics_payload(self) -> bytes: # Renomeado de get_metrics
        """Retorna métricas em formato Prometheus para exposição manual se necessário."""
        return generate_latest(self.registry)


    async def health_check(self) -> Dict[str, Any]: # Usar Any
        """Verifica saúde do coletor de métricas."""
        return {
            'status': 'healthy' if self.server_started else 'stopped',
            'port': self.port,
            'system_metrics_task_running': self._system_metrics_task is not None and not self._system_metrics_task.done(),
            'last_collected_cpu_percent': self.cpu_usage._value if hasattr(self.cpu_usage, '_value') else 'N/A', # Acesso ao valor interno para health check
            'bot_start_time': getattr(CONFIG, 'BOT_START_TIME', datetime.now(timezone.utc).isoformat()) # Exemplo
        }