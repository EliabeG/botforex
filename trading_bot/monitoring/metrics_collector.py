# monitoring/metrics_collector.py
"""Coletor de métricas para Prometheus"""
from prometheus_client import Counter, Histogram, Gauge, Summary, Info
from prometheus_client import start_http_server, generate_latest
import asyncio
from typing import Dict, Optional
from datetime import datetime
import psutil
import platform

from utils.logger import setup_logger

logger = setup_logger("metrics_collector")

class MetricsCollector:
    """Coletor de métricas para monitoramento via Prometheus"""
    
    def __init__(self, port: int = 9090):
        self.port = port
        self.server_started = False
        
        # === Métricas de Sistema ===
        self.system_info = Info('trading_bot_info', 'Informações do sistema')
        self.cpu_usage = Gauge('cpu_usage_percent', 'Uso de CPU em percentual')
        self.memory_usage = Gauge('memory_usage_percent', 'Uso de memória em percentual')
        self.memory_used_mb = Gauge('memory_used_mb', 'Memória usada em MB')
        
        # === Métricas de Conexão ===
        self.connection_status = Gauge('connection_status', 'Status da conexão', ['service'])
        self.websocket_latency = Histogram('websocket_latency_ms', 'Latência WebSocket em ms', 
                                          buckets=(1, 5, 10, 25, 50, 100, 250, 500, 1000))
        self.api_requests_total = Counter('api_requests_total', 'Total de requisições API', ['method', 'endpoint'])
        self.api_errors_total = Counter('api_errors_total', 'Total de erros API', ['method', 'endpoint', 'error_type'])
        
        # === Métricas de Trading ===
        self.trades_total = Counter('trades_total', 'Total de trades executados', ['strategy', 'side', 'result'])
        self.orders_total = Counter('orders_total', 'Total de ordens criadas', ['strategy', 'type', 'status'])
        self.positions_open = Gauge('positions_open', 'Posições abertas', ['strategy'])
        self.order_execution_time = Histogram('order_execution_time_ms', 'Tempo de execução de ordem em ms',
                                            buckets=(10, 25, 50, 100, 250, 500, 1000, 2500, 5000))
        
        # === Métricas Financeiras ===
        self.account_balance = Gauge('account_balance', 'Saldo da conta')
        self.account_equity = Gauge('account_equity', 'Equity da conta')
        self.daily_pnl = Gauge('daily_pnl', 'PnL diário')
        self.daily_pnl_percent = Gauge('daily_pnl_percent', 'PnL diário em percentual')
        self.drawdown_current = Gauge('drawdown_current_percent', 'Drawdown atual em percentual')
        self.margin_level = Gauge('margin_level_percent', 'Nível de margem em percentual')
        
        # === Métricas de Estratégia ===
        self.strategy_score = Gauge('strategy_score', 'Score da estratégia', ['strategy'])
        self.strategy_sharpe_ratio = Gauge('strategy_sharpe_ratio', 'Sharpe ratio da estratégia', ['strategy'])
        self.strategy_win_rate = Gauge('strategy_win_rate', 'Win rate da estratégia', ['strategy'])
        self.strategy_active = Gauge('strategy_active', 'Estratégia ativa (1) ou inativa (0)', ['strategy'])
        
        # === Métricas de Regime de Mercado ===
        self.market_regime = Gauge('market_regime', 'Regime de mercado atual', ['regime'])
        self.regime_confidence = Gauge('regime_confidence', 'Confiança na detecção de regime')
        self.volatility_current = Gauge('volatility_current', 'Volatilidade atual')
        self.spread_average = Gauge('spread_average_pips', 'Spread médio em pips')
        
        # === Métricas de Risco ===
        self.risk_score = Gauge('risk_score', 'Score de risco geral (0-100)')
        self.circuit_breaker_status = Gauge('circuit_breaker_status', 'Status do circuit breaker (0=closed, 1=open, 2=half-open)')
        self.consecutive_losses = Gauge('consecutive_losses', 'Perdas consecutivas')
        self.risk_events_total = Counter('risk_events_total', 'Total de eventos de risco', ['type', 'severity'])
        
        # === Métricas de Performance ===
        self.tick_processing_time = Summary('tick_processing_time_ms', 'Tempo de processamento de tick em ms')
        self.optimization_duration = Histogram('optimization_duration_seconds', 'Duração da otimização em segundos',
                                             buckets=(60, 300, 600, 1800, 3600, 7200))
        self.backtest_speed = Gauge('backtest_speed_ticks_per_second', 'Velocidade do backtest em ticks/s')
        
        # === Métricas de Dados ===
        self.ticks_processed_total = Counter('ticks_processed_total', 'Total de ticks processados', ['symbol'])
        self.data_storage_size_mb = Gauge('data_storage_size_mb', 'Tamanho do armazenamento de dados em MB')
        self.redis_hit_rate = Gauge('redis_hit_rate', 'Taxa de acerto do cache Redis')
        self.data_latency = Histogram('data_latency_ms', 'Latência de acesso a dados em ms',
                                    buckets=(1, 5, 10, 25, 50, 100, 250))
        
        # Registrar informações do sistema
        self._register_system_info()
    
    def _register_system_info(self):
        """Registra informações estáticas do sistema"""
        self.system_info.info({
            'version': '1.0.0',
            'python_version': platform.python_version(),
            'platform': platform.platform(),
            'processor': platform.processor(),
            'hostname': platform.node()
        })
    
    def start(self):
        """Inicia servidor HTTP para métricas"""
        if not self.server_started:
            try:
                start_http_server(self.port)
                self.server_started = True
                logger.info(f"Servidor de métricas iniciado na porta {self.port}")
                
                # Iniciar coleta de métricas do sistema
                asyncio.create_task(self._collect_system_metrics())
                
            except Exception as e:
                logger.error(f"Erro ao iniciar servidor de métricas: {e}")
    
    async def _collect_system_metrics(self):
        """Coleta métricas do sistema periodicamente"""
        while True:
            try:
                # CPU
                self.cpu_usage.set(psutil.cpu_percent(interval=1))
                
                # Memória
                memory = psutil.virtual_memory()
                self.memory_usage.set(memory.percent)
                self.memory_used_mb.set(memory.used / (1024 * 1024))
                
                await asyncio.sleep(10)  # Atualizar a cada 10 segundos
                
            except Exception as e:
                logger.error(f"Erro ao coletar métricas do sistema: {e}")
                await asyncio.sleep(60)
    
    # === Métodos de Atualização de Métricas ===
    
    def update_connection_status(self, feed: bool, trade: bool):
        """Atualiza status das conexões"""
        self.connection_status.labels(service='feed').set(1 if feed else 0)
        self.connection_status.labels(service='trade').set(1 if trade else 0)
    
    def record_websocket_latency(self, latency_ms: float):
        """Registra latência do WebSocket"""
        self.websocket_latency.observe(latency_ms)
    
    def record_api_request(self, method: str, endpoint: str):
        """Registra requisição API"""
        self.api_requests_total.labels(method=method, endpoint=endpoint).inc()
    
    def record_api_error(self, method: str, endpoint: str, error_type: str):
        """Registra erro de API"""
        self.api_errors_total.labels(
            method=method, 
            endpoint=endpoint, 
            error_type=error_type
        ).inc()
    
    def record_trade(self, strategy: str, side: str, pnl: float):
        """Registra trade executado"""
        result = 'win' if pnl > 0 else 'loss'
        self.trades_total.labels(strategy=strategy, side=side, result=result).inc()
    
    def record_order(self, strategy: str, order_type: str, status: str):
        """Registra ordem"""
        self.orders_total.labels(
            strategy=strategy,
            type=order_type,
            status=status
        ).inc()
    
    def update_positions(self, positions_by_strategy: Dict[str, int]):
        """Atualiza contagem de posições abertas por estratégia"""
        for strategy, count in positions_by_strategy.items():
            self.positions_open.labels(strategy=strategy).set(count)
    
    def record_order_execution_time(self, execution_time_ms: float):
        """Registra tempo de execução de ordem"""
        self.order_execution_time.observe(execution_time_ms)
    
    def update_account_metrics(self, balance: float, equity: float, 
                             daily_pnl: float, daily_pnl_pct: float,
                             drawdown: float, margin_level: float):
        """Atualiza métricas da conta"""
        self.account_balance.set(balance)
        self.account_equity.set(equity)
        self.daily_pnl.set(daily_pnl)
        self.daily_pnl_percent.set(daily_pnl_pct * 100)
        self.drawdown_current.set(drawdown * 100)
        self.margin_level.set(margin_level * 100)
    
    def update_strategy_metrics(self, strategy: str, score: float,
                              sharpe: float, win_rate: float, active: bool):
        """Atualiza métricas de estratégia"""
        self.strategy_score.labels(strategy=strategy).set(score)
        self.strategy_sharpe_ratio.labels(strategy=strategy).set(sharpe)
        self.strategy_win_rate.labels(strategy=strategy).set(win_rate)
        self.strategy_active.labels(strategy=strategy).set(1 if active else 0)
    
    def update_market_regime(self, regime: str, confidence: float):
        """Atualiza regime de mercado"""
        # Resetar todos os regimes para 0
        for r in ['trend', 'range', 'high_volatility', 'low_volume']:
            self.market_regime.labels(regime=r).set(0)
        
        # Setar regime atual para 1
        self.market_regime.labels(regime=regime).set(1)
        self.regime_confidence.set(confidence)
    
    def update_market_conditions(self, volatility: float, avg_spread_pips: float):
        """Atualiza condições de mercado"""
        self.volatility_current.set(volatility)
        self.spread_average.set(avg_spread_pips)
    
    def update_risk_metrics(self, risk_score: float, circuit_breaker_state: int,
                          consecutive_losses: int):
        """Atualiza métricas de risco"""
        self.risk_score.set(risk_score)
        self.circuit_breaker_status.set(circuit_breaker_state)
        self.consecutive_losses.set(consecutive_losses)
    
    def record_risk_event(self, event_type: str, severity: str):
        """Registra evento de risco"""
        self.risk_events_total.labels(type=event_type, severity=severity).inc()
    
    def record_tick_processing_time(self, processing_time_ms: float):
        """Registra tempo de processamento de tick"""
        self.tick_processing_time.observe(processing_time_ms)
    
    def record_optimization_duration(self, duration_seconds: float):
        """Registra duração de otimização"""
        self.optimization_duration.observe(duration_seconds)
    
    def update_backtest_speed(self, ticks_per_second: float):
        """Atualiza velocidade do backtest"""
        self.backtest_speed.set(ticks_per_second)
    
    def record_tick_processed(self, symbol: str):
        """Registra tick processado"""
        self.ticks_processed_total.labels(symbol=symbol).inc()
    
    def update_data_metrics(self, storage_size_mb: float, redis_hit_rate: float):
        """Atualiza métricas de dados"""
        self.data_storage_size_mb.set(storage_size_mb)
        self.redis_hit_rate.set(redis_hit_rate)
    
    def record_data_latency(self, latency_ms: float):
        """Registra latência de acesso a dados"""
        self.data_latency.observe(latency_ms)
    
    def stop(self):
        """Para o servidor de métricas"""
        # Prometheus não fornece método direto para parar o servidor
        # Em produção, geralmente o processo inteiro é terminado
        self.server_started = False
        logger.info("Servidor de métricas parado")
    
    def get_metrics(self) -> bytes:
        """Retorna métricas em formato Prometheus"""
        return generate_latest()
    
    async def health_check(self) -> Dict:
        """Verifica saúde do coletor de métricas"""
        return {
            'status': 'healthy' if self.server_started else 'stopped',
            'port': self.port,
            'uptime': datetime.now().isoformat()
        }