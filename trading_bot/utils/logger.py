# utils/logger.py
"""Sistema de logging configurável e robusto para o trading bot."""
import logging
import logging.handlers
import sys
import os
from datetime import datetime, timezone # Adicionado timezone
from pathlib import Path
import json
import colorlog # colorlog já está nos requirements
from typing import Optional, Dict, Any, Union, TextIO # Adicionado Union, TextIO

# Importar CONFIG para caminhos de log e nível de log padrão
from config.settings import CONFIG

class TradingFormatter(logging.Formatter):
    """
    Formatter customizado para logs de trading.
    Adiciona informações extras como 'strategy' e 'trade_id' à mensagem se presentes no record.
    """
    def __init__(self, fmt: Optional[str] = None, datefmt: Optional[str] = None, style: str = '%', validate: bool = True):
        # Definir um formato padrão se nenhum for fornecido
        default_fmt = '%(asctime)s [%(levelname)-8s] %(name)-20s %(message)s'
        super().__init__(fmt or default_fmt, datefmt, style, validate)


    def format(self, record: logging.LogRecord) -> str: # Adicionada tipagem explícita
        # Adicionar informações extras se disponíveis
        # Usar getattr para acesso seguro
        strategy_name = getattr(record, 'strategy', None)
        trade_id_val = getattr(record, 'trade_id', None) # Renomeado

        # Salvar mensagem original para não modificar o record permanentemente entre handlers
        original_msg = record.msg
        prefix = ""

        if strategy_name:
            prefix += f"[{strategy_name}] "
        if trade_id_val:
            prefix += f"[Trade:{trade_id_val}] "
        
        if prefix:
            record.msg = f"{prefix}{original_msg}"
        
        formatted_msg = super().format(record)
        record.msg = original_msg # Restaurar mensagem original no record
        return formatted_msg


class JSONFormatter(logging.Formatter):
    """Formatter para logs em formato JSON, útil para análise e ingestão por sistemas de log."""
    def __init__(self, *args: Any, **kwargs: Any): # Aceitar args e kwargs
        super().__init__(*args, **kwargs) # Passar para o construtor base
        self.default_time_format = '%Y-%m-%dT%H:%M:%S.%fZ' # Formato ISO 8601 com Z para UTC


    def format(self, record: logging.LogRecord) -> str:
        log_obj: Dict[str, Any] = { # Tipagem para log_obj
            'timestamp': datetime.now(timezone.utc).strftime(self.default_time_format), # Usar UTC e formato padrão
            'level': record.levelname,
            'logger_name': record.name, # Renomeado de 'logger' para evitar conflito com o próprio logger
            'message': record.getMessage(), # Usar getMessage() para formatar placeholders
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'thread_id': record.thread, # Adicionado
            'thread_name': record.threadName, # Adicionado
            'process_id': record.process, # Adicionado
        }

        # Adicionar campos extras do record.__dict__ que não são padrão
        # Evitar sobrescrever chaves já definidas ou adicionar dados redundantes/internos do logging.
        standard_keys = {
            'name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 'filename',
            'module', 'lineno', 'funcName', 'created', 'asctime', 'msecs',
            'relativeCreated', 'thread', 'threadName', 'process', 'message',
            'exc_info', 'exc_text', 'stack_info', 'taskName' # taskName para asyncio
        }
        # Adicionar também chaves que já foram explicitamente incluídas em log_obj
        standard_keys.update(log_obj.keys())


        for key, value in record.__dict__.items():
            if key not in standard_keys and not key.startswith('_'): # Ignorar atributos privados
                # Tentar serializar o valor de forma segura
                try:
                    json.dumps(value) # Testar se é serializável
                    log_obj[key] = value
                except TypeError:
                    log_obj[key] = repr(value) # Usar repr() como fallback

        if record.exc_info:
            # self.formatException já retorna uma string formatada do traceback
            log_obj['exception_info'] = self.formatException(record.exc_info) # Renomeado de 'exception'
        if record.stack_info: # Adicionar stack_info se presente
            log_obj['stack_info'] = self.formatStack(record.stack_info)


        return json.dumps(log_obj, default=str) # default=str para lidar com tipos não serializáveis


def setup_logger(name: str,
                level_str: str = CONFIG.LOG_LEVEL, # Usar de CONFIG como default # Renomeado level
                log_to_file: Union[str, bool] = True, # Permitir bool para ligar/desligar file logging com path padrão # Renomeado log_file
                log_to_console: bool = True, # Renomeado console
                use_json_logs: bool = False, # Renomeado json_logs
                log_dir: str = CONFIG.LOG_PATH) -> logging.Logger: # Usar de CONFIG
    """
    Configura e retorna um logger para um módulo específico.

    Args:
        name: Nome do logger (geralmente __name__ do módulo).
        level_str: Nível de log em string (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_to_file: Se True, loga para um arquivo padrão no log_dir. Se string, usa como caminho do arquivo. Se False, desabilita.
        log_to_console: Se True, loga para o console (stdout).
        use_json_logs: Se True, usa formato JSON para todos os handlers.
        log_dir: Diretório base para arquivos de log.

    Returns:
        Logger configurado.
    """
    logger_instance = logging.getLogger(name) # Renomeado logger para logger_instance
    
    # Definir nível do logger. Se já tiver handlers, não mudar nível globalmente, apenas para novos handlers.
    try:
        log_level_attr = getattr(logging, level_str.upper()) # Renomeado level para log_level_attr
    except AttributeError:
        log_level_attr = logging.INFO # Default para INFO se nível inválido
        logger_instance.warning(f"Nível de log '{level_str}' inválido. Usando INFO como padrão.")
    
    logger_instance.setLevel(log_level_attr)

    # Evitar duplicação de handlers se o logger já foi configurado
    # (útil se esta função for chamada múltiplas vezes para o mesmo nome de logger)
    if logger_instance.handlers:
        # Se os handlers existentes já estiverem no nível desejado ou mais baixo, não reconfigurar.
        # Ou, limpar handlers e reconfigurar (mais simples para garantir consistência):
        # logger_instance.handlers.clear()
        # logger_instance.propagate = False # Evitar propagar para root se configurando individualmente
        # A lógica original retorna o logger se já tiver handlers. Manteremos isso por enquanto.
        # No entanto, isso pode impedir a mudança dinâmica de formato de log (JSON vs texto)
        # ou nível se chamado novamente com parâmetros diferentes.
        # Uma opção melhor seria checar se os handlers são do tipo esperado.
        # Por ora, vamos simplificar: se chamado de novo, pode adicionar handlers.
        # Para evitar duplicação, quem chama deve ser responsável ou usar um gerenciador de loggers.
        # O código original tem `if logger.handlers: return logger`. Isso é mantido.
        if len(logger_instance.handlers) > 0 and any(isinstance(h, (logging.StreamHandler, logging.FileHandler)) for h in logger_instance.handlers):
            # Assume-se que se já tem handlers de stream/file, está configurado.
            # Poderia verificar os formatters e níveis dos handlers existentes se necessário.
            # logger_instance.debug(f"Logger '{name}' já possui handlers. Retornando instância existente.")
            return logger_instance


    # Console Handler (stdout) com cores
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout) # Usar sys.stdout explicitamente
        console_handler.setLevel(log_level_attr) # Nível do handler pode ser diferente do logger

        if use_json_logs:
            console_formatter = JSONFormatter()
        else:
            console_formatter = colorlog.ColoredFormatter(
                fmt='%(log_color)s%(asctime)s [%(levelname)-8s] %(name)-25s %(message)s%(reset)s', # Ajustado width do nome
                datefmt='%Y-%m-%d %H:%M:%S', # Formato de data/hora
                log_colors={
                    'DEBUG': 'cyan', 'INFO': 'green',
                    'WARNING': 'yellow', 'ERROR': 'red',
                    'CRITICAL': 'bold_red,bg_white', # CRITICAL mais destacado
                }
            )
        console_handler.setFormatter(console_formatter)
        logger_instance.addHandler(console_handler)

    # File Handler (rotativo)
    actual_log_file_path: Optional[str] = None # Renomeado
    if log_to_file is True: # Usar nome de arquivo padrão
        # Garantir que log_dir seja um Path
        log_directory = Path(log_dir) # Renomeado
        log_directory.mkdir(parents=True, exist_ok=True) # Criar diretório se não existir
        # Nome do arquivo de log baseado no nome do logger, com fallback para 'app'
        log_file_name_base = name.replace('.', '_') if name != "__main__" else "app" # Renomeado
        actual_log_file_path = str(log_directory / f"{log_file_name_base}.log")
    elif isinstance(log_to_file, str): # Usar caminho fornecido
        actual_log_file_path = log_to_file
        log_directory = Path(actual_log_file_path).parent
        log_directory.mkdir(parents=True, exist_ok=True)


    if actual_log_file_path:
        # Rotating file handler (ex: 10MB por arquivo, mantém 5 backups)
        # Usar TimedRotatingFileHandler para rotação baseada em tempo (diária) é comum
        # RotatingFileHandler é baseado em tamanho.
        # O original usava RotatingFileHandler, vamos mantê-lo mas ajustar os params.
        max_bytes_val = getattr(CONFIG, 'LOG_MAX_BYTES', 100 * 1024 * 1024) # 100MB # Renomeado
        backup_count_val = getattr(CONFIG, 'LOG_BACKUP_COUNT', 10) # Renomeado

        file_handler = logging.handlers.RotatingFileHandler(
            filename=actual_log_file_path,
            maxBytes=max_bytes_val,
            backupCount=backup_count_val,
            encoding='utf-8'
        )
        file_handler.setLevel(log_level_attr)

        if use_json_logs:
            file_formatter = JSONFormatter()
        else:
            # Usar TradingFormatter para logs de arquivo se não for JSON
            file_formatter = TradingFormatter( # Formato ligeiramente diferente para arquivo
                fmt='%(asctime)s [%(levelname)-8s] %(name)-25s (%(module)s:%(lineno)d) %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S,%f' # Adicionar milissegundos
            )
        file_handler.setFormatter(file_formatter)
        logger_instance.addHandler(file_handler)

    # Handler específico para erros críticos (opcional, mas boa prática)
    # O código original adicionava um error_handler para CADA logger.
    # É mais comum ter um error_handler no logger RAIZ ou em um logger central de erros.
    # Se for para cada logger, o nome do arquivo deve ser único ou eles vão sobrescrever.
    # Por ora, mantendo a lógica de um error_log por logger, mas com nome de arquivo único.
    # Se CONFIG.ENABLE_CRITICAL_ERROR_LOG for True:
    if getattr(CONFIG, 'ENABLE_DEDICATED_ERROR_LOG', True): # Exemplo de flag de config
        error_log_base_dir = Path(log_dir) / "errors" # Subdiretório para logs de erro
        error_log_base_dir.mkdir(parents=True, exist_ok=True)
        error_file_path = error_log_base_dir / f"{name.replace('.', '_')}_error.log" # Nome único

        # Usar um handler diferente, ex: WatchedFileHandler ou apenas FileHandler se rotação não for crítica para erros
        error_file_handler = logging.FileHandler(error_file_path, encoding='utf-8') # Renomeado
        error_file_handler.setLevel(logging.ERROR) # Logar apenas ERROR e CRITICAL
        error_file_handler.setFormatter(JSONFormatter()) # Erros sempre em JSON para fácil parsing
        logger_instance.addHandler(error_file_handler)

    # Evitar que o logger propague para o logger raiz se já tiver handlers próprios
    # (a menos que o logger raiz também esteja configurado de forma desejada)
    # logger_instance.propagate = False # Descomentar se quiser isolar este logger

    return logger_instance


class TradingLogger:
    """Logger especializado para operações de trading, com métodos estruturados."""
    def __init__(self, strategy_name: str, base_logger_name: Optional[str] = None): # Renomeado
        # O nome do logger pode incluir a estratégia para fácil filtragem
        logger_name_tl = base_logger_name or f"trading.{strategy_name}" # Renomeado
        # Usar o setup_logger para consistência
        self.logger = setup_logger(logger_name_tl, use_json_logs=getattr(CONFIG, 'LOG_TRADES_AS_JSON', False)) # Exemplo de config
        self.trade_log_buffer: List[Dict[str, Any]] = [] # Renomeado e tipado (para buffer, se necessário)
        self.strategy_name_tl = strategy_name # Renomeado

    def log_trade_open(self, trade_id: str, symbol: str, # Adicionado symbol
                      side: str, size: float, entry_price: float, # Renomeado price
                      stop_loss: Optional[float] = None, take_profit: Optional[float] = None, # Adicionado SL/TP
                      **kwargs: Any): # Adicionada tipagem para kwargs
        """Log de abertura de uma nova trade."""
        extra_data = { # Renomeado
            'trade_event': 'OPEN', # Adicionado tipo de evento
            'trade_id': trade_id,
            'strategy': self.strategy_name_tl, # Usar o nome da estratégia da instância
            'symbol': symbol,
            'side': side.upper(),
            'size_lots': size, # Renomeado
            'entry_price': entry_price,
            'stop_loss_price': stop_loss, # Renomeado
            'take_profit_price': take_profit, # Renomeado
            **kwargs # Adicionar quaisquer outros metadados
        }
        # Mensagem de log concisa
        log_msg = f"Trade ABERTO: {extra_data['side']} {extra_data['size_lots']} {extra_data['symbol']} @ {extra_data['entry_price']:.5f}"
        if stop_loss: log_msg += f" SL: {stop_loss:.5f}"
        if take_profit: log_msg += f" TP: {take_profit:.5f}"

        self.logger.info(log_msg, extra=extra_data)
        
        # Opcional: adicionar ao buffer/histórico interno se este logger mantiver estado
        # self.trade_log_buffer.append({
        #     'timestamp_utc': datetime.now(timezone.utc).isoformat(), # Usar UTC
        #     **extra_data
        # })


    def log_trade_close(self, trade_id: str, symbol: str, # Adicionado symbol
                       exit_price_val: float, # Renomeado
                       pnl_value: float, # Renomeado
                       exit_reason: str, # Renomeado
                       duration_seconds: Optional[float] = None, # Adicionado
                       **kwargs: Any):
        """Log de fechamento de uma trade."""
        extra_data = {
            'trade_event': 'CLOSE',
            'trade_id': trade_id,
            'strategy': self.strategy_name_tl,
            'symbol': symbol,
            'exit_price': exit_price_val,
            'pnl_currency': pnl_value, # Adicionar unidade se não for óbvio (ex: pnl_usd)
            'exit_reason': exit_reason,
            'duration_seconds': duration_seconds,
            **kwargs
        }
        log_level = logging.INFO if pnl_value >= 0 else logging.WARNING # Usar WARNING para perdas
        
        log_msg = f"Trade FECHADO: ID {trade_id}, PnL ${pnl_value:.2f}, Preço Saída {exit_price_val:.5f}. Razão: {exit_reason}"
        if duration_seconds is not None:
            log_msg += f" (Duração: {format_duration_to_readable_str(duration_seconds)})" # Usar helper


        self.logger.log(log_level, log_msg, extra=extra_data)
        # self.trade_log_buffer.append(...) # Adicionar ao buffer se necessário


    def log_risk_management_event(self, event_type_str: str, severity_str: str, # Renomeados
                      description_text: str, **kwargs: Any):
        """Log de um evento de gestão de risco (ex: circuit breaker, stop out)."""
        # Mapear severity_str para níveis de log Python
        level_map = {
            'low': logging.INFO, 'medium': logging.WARNING,
            'high': logging.ERROR, 'critical': logging.CRITICAL
        }
        log_level_risk = level_map.get(severity_str.lower(), logging.WARNING) # Renomeado

        extra_data = {
            'risk_event_type': event_type_str, # Renomeado
            'severity': severity_str.upper(),
            # 'is_risk_event': True, # Implícito pelo nome do método # Removido
            **kwargs
        }
        self.logger.log(
            log_level_risk,
            f"[RISCO-{severity_str.upper()}] {event_type_str}: {description_text}",
            extra=extra_data
        )

    def log_strategy_performance_update(self, metrics_dict: Dict[str, Any]): # Renomeado
        """Log de atualização de métricas de performance da estratégia."""
        # Formatar uma mensagem concisa com as principais métricas
        # Exemplo: PnL, Win Rate, Sharpe, Trades
        pnl = metrics_dict.get('total_pnl', metrics_dict.get('net_pnl', 0.0))
        win_rate = metrics_dict.get('win_rate', 0.0) * 100
        sharpe = metrics_dict.get('sharpe_ratio', 0.0)
        trades = metrics_dict.get('total_trades', 0)

        log_msg = (f"Atualização Performance ({self.strategy_name_tl}): "
                   f"PnL Total ${pnl:.2f}, WinRate {win_rate:.1f}%, "
                   f"Sharpe {sharpe:.2f}, Trades {trades}")
        
        self.logger.info(log_msg, extra={'performance_metrics_update': metrics_dict}) # Chave mais específica


    def get_internal_trade_history(self, trade_id_filter: Optional[str] = None) -> list: # Renomeado
        """Retorna histórico de trades logados por esta instância (se bufferizado)."""
        if not self.trade_log_buffer: # Se não estiver usando buffer interno
            # logger.info("Histórico de trades não está sendo bufferizado internamente pelo TradingLogger.")
            return []
        
        if trade_id_filter:
            return [log_entry for log_entry in self.trade_log_buffer if log_entry.get('trade_id') == trade_id_filter] # Renomeado
        return list(self.trade_log_buffer) # Retornar cópia


# Logger para métricas (formato específico para Prometheus ou outros sistemas de métricas)
class MetricsLogger:
    """Logger especializado para métricas do sistema, geralmente em formato estruturado (JSON)."""
    def __init__(self, logger_name: str = "system_metrics"): # Renomeado
        # Métricas geralmente são logadas em JSON para fácil parsing por sistemas como Fluentd/ELK
        # ou para serem consumidas por um coletor Prometheus se não usar o prometheus_client diretamente.
        self.logger = setup_logger(logger_name, use_json_logs=True, log_to_console=False) # JSON, sem console por padrão

    def log_metric_event(self, metric_name: str, value: Any, unit: Optional[str] = None, # Renomeado
                         tags: Optional[Dict[str, Any]] = None, **extra_data: Any):
        """Loga um evento de métrica genérico."""
        metric_payload = { # Renomeado
            'metric_type': 'generic_metric', # Tipo geral
            'metric_name': metric_name,
            'value': value,
            **(tags or {}), # Adicionar tags como campos de alto nível
            **extra_data   # Outros dados específicos
        }
        if unit: metric_payload['unit'] = unit
        
        # INFO é geralmente usado para métricas, pois não são erros.
        self.logger.info(f"Métrica: {metric_name} = {value} {unit or ''}", extra=metric_payload)


    def log_latency_metric(self, operation_name: str, latency_value_ms: float, # Renomeados todos
                          success: bool = True, endpoint: Optional[str] = None):
        """Log de latência para uma operação específica."""
        payload = {
            'metric_type': 'latency',
            'operation': operation_name,
            'value_ms': latency_value_ms, # Renomeado e unidade clara
            'unit': 'ms',
            'successful_operation': success # Renomeado
        }
        if endpoint: payload['endpoint_target'] = endpoint # Renomeado
        self.logger.info(f"Latência: {operation_name} = {latency_value_ms:.2f}ms, Sucesso: {success}", extra=payload)


    def log_order_execution_metric(self, strategy_name: str, symbol: str, side: str, # Renomeados todos
                                 is_successful: bool, slippage_pips_val: float = 0.0,
                                 fill_time_ms: Optional[float] = None):
        """Log de métrica de execução de ordem."""
        payload = {
            'metric_type': 'order_execution',
            'strategy': strategy_name,
            'symbol': symbol,
            'side': side.upper(),
            'execution_successful': is_successful, # Renomeado
            'slippage_pips': slippage_pips_val
        }
        if fill_time_ms is not None: payload['fill_time_ms'] = fill_time_ms
        self.logger.info(
            f"Execução Ordem: Strat={strategy_name}, Symbol={symbol}, Side={side.upper()}, "
            f"Sucesso={is_successful}, Slippage={slippage_pips_val:.1f} pips",
            extra=payload
        )


    def log_system_health_metrics(self, cpu_percent_val: float, memory_percent_val: float, # Renomeados todos
                         active_connections_val: int, disk_usage_percent_val: Optional[float] = None):
        """Log de métricas de saúde do sistema."""
        payload = {
            'metric_type': 'system_health',
            'cpu_usage_percent': cpu_percent_val, # Renomeado
            'memory_usage_percent': memory_percent_val, # Renomeado
            'network_active_connections': active_connections_val, # Renomeado
        }
        if disk_usage_percent_val is not None: payload['disk_usage_percent_main'] = disk_usage_percent_val # Renomeado
        
        self.logger.info(
            f"Saúde Sistema: CPU {cpu_percent_val:.1f}%, Mem {memory_percent_val:.1f}%, Conexões {active_connections_val}",
            extra=payload
        )


# Configurar logger raiz da aplicação (chamado uma vez no início do bot)
def configure_application_root_logger(app_log_level_str: str = CONFIG.LOG_LEVEL, # Renomeado
                         app_log_dir: str = CONFIG.LOG_PATH, # Renomeado
                         use_json_for_root: bool = False): # Renomeado
    """Configura o logger raiz da aplicação."""

    log_directory_root = Path(app_log_dir) # Renomeado
    log_directory_root.mkdir(parents=True, exist_ok=True) # Criar diretório se não existir

    root_logger_instance = logging.getLogger() # Renomeado
    try:
        root_log_level = getattr(logging, app_log_level_str.upper()) # Renomeado
    except AttributeError:
        root_log_level = logging.INFO
    root_logger_instance.setLevel(root_log_level)

    # Limpar handlers existentes do root logger para evitar duplicação se chamado múltiplas vezes
    # (cuidado se outras bibliotecas adicionarem handlers ao root)
    # root_logger_instance.handlers.clear() # Comentado, pois pode ser destrutivo

    # Verificar se já existem handlers de arquivo/console para não duplicar
    has_file_handler = any(isinstance(h, logging.handlers.TimedRotatingFileHandler) for h in root_logger_instance.handlers)
    has_console_handler = any(isinstance(h, logging.StreamHandler) and h.stream in [sys.stdout, sys.stderr] for h in root_logger_instance.handlers)


    # Handler para arquivo principal (rotacionado por tempo, ex: meia-noite)
    if not has_file_handler:
        main_log_filename = log_directory_root / f"trading_bot_main_{datetime.now(timezone.utc).strftime('%Y%m%d')}.log" # Renomeado
        # Usar TimedRotatingFileHandler para rotação diária
        file_handler_root = logging.handlers.TimedRotatingFileHandler( # Renomeado
            filename=main_log_filename,
            when='midnight', # Rotacionar à meia-noite
            interval=1,      # Diariamente
            backupCount=getattr(CONFIG, 'LOG_RETENTION_DAYS', 30), # Usar de CONFIG
            encoding='utf-8',
            utc=True # Usar UTC para nomes de arquivo e rotação
        )
        file_handler_root.setLevel(root_log_level) # Nível do handler

        if use_json_for_root:
            file_handler_root.setFormatter(JSONFormatter())
        else:
            file_handler_root.setFormatter(TradingFormatter( # Usar TradingFormatter
                fmt='%(asctime)s [%(levelname)-8s] %(name)-25s (%(module)s:%(lineno)d) %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S,%f'
            ))
        root_logger_instance.addHandler(file_handler_root)

    # Console handler (colorido para texto, JSON simples se json_logs for True)
    if not has_console_handler:
        console_handler_root = logging.StreamHandler(sys.stdout) # Renomeado
        console_handler_root.setLevel(root_log_level) # Nível do handler

        if use_json_for_root:
            console_handler_root.setFormatter(JSONFormatter())
        else:
            console_handler_root.setFormatter(colorlog.ColoredFormatter(
                fmt='%(log_color)s%(asctime)s [%(levelname)-8s] %(name)-20s %(message)s%(reset)s', # Ajustado width e nome
                datefmt='%H:%M:%S' # Formato mais curto para console
            ))
        root_logger_instance.addHandler(console_handler_root)

    # Suprimir logs verbosos de bibliotecas de terceiros (ajustar níveis conforme necessidade)
    libraries_to_silence = { # Renomeado
        'urllib3': logging.WARNING,
        'websockets': logging.INFO, # INFO pode ser útil para conexões, WARNING para menos verboso
        'asyncio': logging.INFO,    # Logs de asyncio podem ser úteis para debug de tasks
        'aioredis': logging.WARNING,
        'httpx': logging.WARNING,
        'optuna': logging.INFO, # Optuna pode ser verboso em DEBUG
        'matplotlib': logging.WARNING, # Matplotlib pode ser verboso
    }
    for lib_name, lib_level in libraries_to_silence.items(): # Renomeado
        logging.getLogger(lib_name).setLevel(lib_level)
    
    logger.info(f"Logger raiz configurado. Nível: {app_log_level_str.upper()}. Console: {not has_console_handler}, Arquivo: {not has_file_handler}")
    return root_logger_instance


# Função auxiliar para log estruturado de eventos genéricos
def log_event(target_logger: logging.Logger, event_name: str, # Renomeado
              event_message: str, # Renomeado
              log_level: int = logging.INFO, # Permitir especificar nível
              **event_data_kwargs: Any): # Renomeado
    """Helper para log estruturado de eventos com dados adicionais."""
    # Construir o campo 'extra' para o logger
    # A chave 'event_type' foi usada no código original, manter consistência ou renomear.
    extra_payload = { # Renomeado
        'event_name': event_name, # Usar event_name em vez de event_type para evitar conflito com loggers que já usam event_type
        'event_specific_data': event_data_kwargs, # Renomeado
        'log_timestamp_utc': datetime.now(timezone.utc).isoformat() # Adicionar timestamp explícito ao payload
    }
    target_logger.log(log_level, event_message, extra=extra_payload)