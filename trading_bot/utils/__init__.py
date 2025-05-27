# utils/__init__.py
"""Utilitarios gerais para o trading bot"""
from .logger import setup_logger, TradingLogger, MetricsLogger, log_event, JSONFormatter, TradingFormatter
from .ntp_sync import NTPSynchronizer, get_ntp_synchronizer_instance as get_ntp_synchronizer, get_accurate_utc_time as get_accurate_time, get_accurate_timestamp_ms_utc as get_accurate_timestamp_ms, ensure_initial_ntp_sync as ensure_time_sync
from .helpers import (
    calculate_pip_value, format_price, round_to_tick_size as round_to_tick,
    calculate_forex_position_size as calculate_position_size, calculate_risk_reward_ratio,
    convert_pips_to_price_diff as pips_to_price, convert_price_diff_to_pips as price_to_pips, calculate_spread_cost_in_account_ccy as calculate_spread_cost,
    get_forex_trading_session as get_trading_session, calculate_lot_size_from_risk_pct as calculate_lot_from_risk, is_forex_market_open as is_market_open,
    calculate_compound_returns_on_balance as calculate_compound_returns, calculate_sharpe_ratio_simple as calculate_sharpe_ratio, calculate_max_drawdown_from_equity as calculate_max_drawdown,
    normalize_forex_symbol as normalize_symbol, is_major_forex_pair as is_major_pair, calculate_pearson_correlation as calculate_correlation,
    time_until_next_session_starts as time_until_session, validate_stop_loss_take_profit as validate_stops, format_duration_to_readable_str as format_duration,
    calculate_trade_expectancy as calculate_expectancy, round_value_to_broker_precision,
    # Constantes tambem podem ser exportadas se desejado
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