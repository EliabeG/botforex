# utils/__init__.py
"""Utilitários gerais para o trading bot""" # Descrição ajustada
from .logger import setup_logger, TradingLogger, MetricsLogger, log_event, JSONFormatter, TradingFormatter # Exportar mais componentes úteis
from .ntp_sync import NTPSynchronizer, get_ntp_synchronizer, get_accurate_time, get_accurate_timestamp_ms, ensure_time_sync # Exportar funções de conveniência
from .helpers import ( # Quebrar import longo para melhor legibilidade
    calculate_pip_value, format_price, round_to_tick,
    calculate_position_size, calculate_risk_reward_ratio,
    pips_to_price, price_to_pips, calculate_spread_cost,
    get_trading_session, calculate_lot_from_risk, is_market_open,
    calculate_compound_returns, calculate_sharpe_ratio, calculate_max_drawdown,
    normalize_symbol, is_major_pair, calculate_correlation,
    time_until_session, validate_stops, format_duration,
    calculate_expectancy, round_to_broker_precision,
    # Constantes também podem ser exportadas se desejado
    # SECONDS_IN_DAY, TRADING_DAYS_YEAR
)


__all__ = [
    # Logger components
    'setup_logger',
    'TradingLogger',
    'MetricsLogger',
    'log_event',
    'JSONFormatter',
    'TradingFormatter',

    # NTP Sync components
    'NTPSynchronizer',
    'get_ntp_synchronizer',
    'get_accurate_time',
    'get_accurate_timestamp_ms',
    'ensure_time_sync',

    # Helper functions
    'calculate_pip_value',
    'format_price',
    'round_to_tick',
    'calculate_position_size',
    'calculate_risk_reward_ratio',
    'pips_to_price',
    'price_to_pips',
    'calculate_spread_cost',
    'get_trading_session',
    'calculate_lot_from_risk',
    'is_market_open',
    'calculate_compound_returns',
    'calculate_sharpe_ratio',
    'calculate_max_drawdown',
    'normalize_symbol',
    'is_major_pair',
    'calculate_correlation',
    'time_until_session',
    'validate_stops',
    'format_duration',
    'calculate_expectancy',
    'round_to_broker_precision',
]

# ===================================