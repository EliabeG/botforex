# utils/logger.py
"""Sistema de logging configurável e robusto"""
import logging
import logging.handlers
import sys
import os
from datetime import datetime
from pathlib import Path
import json
import colorlog
from typing import Optional, Dict, Any

class TradingFormatter(logging.Formatter):
    """Formatter customizado para logs de trading"""
    
    def format(self, record):
        # Adicionar informações extras se disponíveis
        if hasattr(record, 'strategy'):
            record.msg = f"[{record.strategy}] {record.msg}"
        
        if hasattr(record, 'trade_id'):
            record.msg = f"[Trade:{record.trade_id}] {record.msg}"
        
        return super().format(record)

class JSONFormatter(logging.Formatter):
    """Formatter para logs em JSON (para análise posterior)"""
    
    def format(self, record):
        log_obj = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Adicionar campos extras
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'created', 'filename', 
                          'funcName', 'levelname', 'levelno', 'lineno',
                          'module', 'msecs', 'pathname', 'process',
                          'processName', 'relativeCreated', 'thread',
                          'threadName', 'exc_info', 'exc_text', 'stack_info']:
                log_obj[key] = value
        
        if record.exc_info:
            log_obj['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_obj)

def setup_logger(name: str, 
                level: str = "INFO",
                log_file: Optional[str] = None,
                console: bool = True,
                json_logs: bool = False) -> logging.Logger:
    """
    Configura logger para módulo específico
    
    Args:
        name: Nome do logger
        level: Nível de log (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Arquivo para salvar logs
        console: Se deve logar no console
        json_logs: Se deve usar formato JSON
    
    Returns:
        Logger configurado
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Evitar duplicação de handlers
    if logger.handlers:
        return logger
    
    # Console handler com cores
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        
        if json_logs:
            console_formatter = JSONFormatter()
        else:
            # Formato colorido para console
            console_formatter = colorlog.ColoredFormatter(
                '%(log_color)s%(asctime)s [%(levelname)-8s] %(name)-20s %(message)s%(reset)s',
                datefmt='%Y-%m-%d %H:%M:%S',
                log_colors={
                    'DEBUG': 'cyan',
                    'INFO': 'green',
                    'WARNING': 'yellow',
                    'ERROR': 'red',
                    'CRITICAL': 'red,bg_white',
                }
            )
        
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        # Criar diretório se não existir
        log_dir = Path(log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Rotating file handler (max 100MB, mantém 10 arquivos)
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=100 * 1024 * 1024,  # 100MB
            backupCount=10,
            encoding='utf-8'
        )
        
        if json_logs:
            file_formatter = JSONFormatter()
        else:
            file_formatter = TradingFormatter(
                '%(asctime)s [%(levelname)-8s] %(name)-20s %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    # Handler específico para erros críticos
    error_log_file = f"logs/errors_{datetime.now().strftime('%Y%m%d')}.log"
    error_dir = Path(error_log_file).parent
    error_dir.mkdir(parents=True, exist_ok=True)
    
    error_handler = logging.FileHandler(error_log_file, encoding='utf-8')
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(JSONFormatter())
    logger.addHandler(error_handler)
    
    return logger

class TradingLogger:
    """Logger especializado para operações de trading"""
    
    def __init__(self, name: str):
        self.logger = setup_logger(name)
        self.trade_logs = []
        
    def log_trade_open(self, trade_id: str, strategy: str, 
                      side: str, size: float, price: float, **kwargs):
        """Log de abertura de trade"""
        extra = {
            'trade_id': trade_id,
            'strategy': strategy,
            'action': 'OPEN',
            'side': side,
            'size': size,
            'price': price,
            **kwargs
        }
        
        self.logger.info(
            f"Trade aberto: {side.upper()} {size} @ {price}",
            extra=extra
        )
        
        self.trade_logs.append({
            'timestamp': datetime.utcnow(),
            'event': 'open',
            **extra
        })
    
    def log_trade_close(self, trade_id: str, exit_price: float, 
                       pnl: float, reason: str, **kwargs):
        """Log de fechamento de trade"""
        extra = {
            'trade_id': trade_id,
            'action': 'CLOSE',
            'exit_price': exit_price,
            'pnl': pnl,
            'reason': reason,
            **kwargs
        }
        
        level = logging.INFO if pnl >= 0 else logging.WARNING
        self.logger.log(
            level,
            f"Trade fechado: PnL ${pnl:.2f} @ {exit_price} ({reason})",
            extra=extra
        )
        
        self.trade_logs.append({
            'timestamp': datetime.utcnow(),
            'event': 'close',
            **extra
        })
    
    def log_risk_event(self, event_type: str, severity: str, 
                      description: str, **kwargs):
        """Log de evento de risco"""
        extra = {
            'event_type': event_type,
            'severity': severity,
            'risk_event': True,
            **kwargs
        }
        
        level = {
            'low': logging.INFO,
            'medium': logging.WARNING,
            'high': logging.ERROR,
            'critical': logging.CRITICAL
        }.get(severity, logging.WARNING)
        
        self.logger.log(
            level,
            f"[RISCO-{severity.upper()}] {description}",
            extra=extra
        )
    
    def log_performance_update(self, metrics: Dict[str, Any]):
        """Log de atualização de performance"""
        self.logger.info(
            f"Performance: Win Rate {metrics.get('win_rate', 0):.1%} | "
            f"Sharpe {metrics.get('sharpe_ratio', 0):.2f} | "
            f"DD {metrics.get('drawdown', 0):.1%}",
            extra={'performance_metrics': metrics}
        )
    
    def get_trade_history(self, trade_id: Optional[str] = None) -> list:
        """Retorna histórico de trades"""
        if trade_id:
            return [log for log in self.trade_logs if log.get('trade_id') == trade_id]
        return self.trade_logs

# Logger para métricas (formato específico para Prometheus)
class MetricsLogger:
    """Logger para métricas do sistema"""
    
    def __init__(self):
        self.logger = setup_logger("metrics", json_logs=True)
        
    def log_latency(self, operation: str, latency_ms: float):
        """Log de latência"""
        self.logger.info(
            f"Latency: {operation}",
            extra={
                'metric_type': 'latency',
                'operation': operation,
                'value': latency_ms,
                'unit': 'ms'
            }
        )
    
    def log_execution(self, strategy: str, success: bool, 
                     slippage: float = 0):
        """Log de execução"""
        self.logger.info(
            f"Execution: {strategy}",
            extra={
                'metric_type': 'execution',
                'strategy': strategy,
                'success': success,
                'slippage': slippage
            }
        )
    
    def log_system_health(self, cpu: float, memory: float, 
                         connections: int):
        """Log de saúde do sistema"""
        self.logger.info(
            "System health",
            extra={
                'metric_type': 'health',
                'cpu_percent': cpu,
                'memory_percent': memory,
                'active_connections': connections
            }
        )

# Configurar logger raiz
def configure_root_logger(level: str = "INFO", 
                         log_dir: str = "logs",
                         json_logs: bool = False):
    """Configura o logger raiz da aplicação"""
    
    # Criar diretório de logs
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # Configurar logger raiz
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))
    
    # Limpar handlers existentes
    root_logger.handlers.clear()
    
    # Handler para arquivo principal
    main_log = f"{log_dir}/trading_bot_{datetime.now().strftime('%Y%m%d')}.log"
    file_handler = logging.handlers.TimedRotatingFileHandler(
        main_log,
        when='midnight',
        interval=1,
        backupCount=30,
        encoding='utf-8'
    )
    
    if json_logs:
        file_handler.setFormatter(JSONFormatter())
    else:
        file_handler.setFormatter(TradingFormatter(
            '%(asctime)s [%(levelname)-8s] %(name)-20s %(message)s'
        ))
    
    root_logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(colorlog.ColoredFormatter(
        '%(log_color)s%(asctime)s [%(levelname)-8s] %(message)s%(reset)s',
        datefmt='%H:%M:%S'
    ))
    root_logger.addHandler(console_handler)
    
    # Suprimir logs verbosos de bibliotecas
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('websockets').setLevel(logging.WARNING)
    logging.getLogger('asyncio').setLevel(logging.WARNING)
    
    return root_logger

# Função auxiliar para log estruturado
def log_event(logger: logging.Logger, event_type: str, 
              message: str, **data):
    """Helper para log estruturado de eventos"""
    logger.info(
        message,
        extra={
            'event_type': event_type,
            'event_data': data,
            'timestamp': datetime.utcnow().isoformat()
        }
    )