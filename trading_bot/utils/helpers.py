# utils/helpers.py
"""Funcoes auxiliares diversas para o trading bot."""
import math
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from typing import Union, Optional, Tuple, List, Any, TYPE_CHECKING
from datetime import datetime, timedelta, timezone, date

import pytz
import numpy as np

from .logger import setup_logger
logger = setup_logger("utils_helpers_logger")

# Importar CONFIG e RISK_PARAMS diretamente
from config.settings import CONFIG as GlobalTradingConfig # Renomear para evitar conflito com nome de variavel local
from config.risk_config import RISK_PARAMS as GlobalRiskParams # Renomear para evitar conflito

if TYPE_CHECKING:
    from config.settings import TradingConfig
    from config.risk_config import RiskParameters # Adicionar tipo para GlobalRiskParams
    CONFIG_TYPE = TradingConfig
    RISK_PARAMS_TYPE = RiskParameters
else:
    CONFIG_TYPE = Any
    RISK_PARAMS_TYPE = Any


_CONFIG_INSTANCE: Optional[CONFIG_TYPE] = None
_RISK_PARAMS_INSTANCE: Optional[RISK_PARAMS_TYPE] = None # Adicionado para RISK_PARAMS

def get_config_helpers() -> CONFIG_TYPE: # Esta funcao retorna apenas CONFIG
    global _CONFIG_INSTANCE
    if _CONFIG_INSTANCE is None:
        _CONFIG_INSTANCE = GlobalTradingConfig
    return _CONFIG_INSTANCE

def get_risk_params_helpers() -> RISK_PARAMS_TYPE: # Nova funcao para RISK_PARAMS
    global _RISK_PARAMS_INSTANCE
    if _RISK_PARAMS_INSTANCE is None:
        _RISK_PARAMS_INSTANCE = GlobalRiskParams
    return _RISK_PARAMS_INSTANCE


def calculate_pip_value(symbol: str,
                       lot_size: float = 1.0,
                       account_currency: str = "USD",
                       current_exchange_rate_quote_to_acct_currency: Optional[float] = None,
                       quote_currency_inferred: Optional[str] = None) -> float:
    """
    Calcula o valor do pip para um par de moedas na moeda da conta.
    """
    if lot_size <= 0: return 0.0

    cfg = get_config_helpers()
    contract_size = getattr(cfg, 'CONTRACT_SIZE', 100000.0)

    normalized_symbol_hlp = symbol.upper().replace("/", "")

    if quote_currency_inferred is None:
        if len(normalized_symbol_hlp) == 6:
            quote_currency_inferred = normalized_symbol_hlp[3:]
        elif len(normalized_symbol_hlp) == 7 and normalized_symbol_hlp.endswith("M"):
             quote_currency_inferred = normalized_symbol_hlp[3:6]
        else:
            logger.warning(f"Nao foi possivel inferir a moeda de cotacao de '{symbol}' para valor do pip. Retornando 0.")
            return 0.0

    pip_size_in_quote_ccy_hlp: float
    if "JPY" in quote_currency_inferred.upper():
        pip_size_in_quote_ccy_hlp = 0.01
    elif any(p in normalized_symbol_hlp for p in ["XAU", "XAG", "BTC", "ETH"]):
        if "XAU" in normalized_symbol_hlp: pip_size_in_quote_ccy_hlp = 0.01
        else: pip_size_in_quote_ccy_hlp = 0.0001
    else:
        pip_size_in_quote_ccy_hlp = 0.0001

    pip_value_one_lot_in_quote_ccy = pip_size_in_quote_ccy_hlp * contract_size
    pip_value_total_lots_in_quote_ccy = pip_value_one_lot_in_quote_ccy * lot_size

    if quote_currency_inferred.upper() == account_currency.upper():
        return pip_value_total_lots_in_quote_ccy
    else:
        if current_exchange_rate_quote_to_acct_currency is None or current_exchange_rate_quote_to_acct_currency <= 0:
            logger.error(f"Taxa de cambio ({quote_currency_inferred}/{account_currency}) nao fornecida ou invalida para {symbol}. "
                           f"Nao e possivel converter o valor do pip para a moeda da conta.")
            return 0.0

        return pip_value_total_lots_in_quote_ccy * current_exchange_rate_quote_to_acct_currency


def format_price(price: Union[float, int, Decimal], symbol: str = "EURUSD", digits: Optional[int] = None) -> str:
    """
    Formata o preco com a precisao correta para o simbolo ou com 'digits' especificado.
    """
    if not isinstance(price, (float, int, Decimal, np.number)):
        try:
            price_val = float(price)
        except (ValueError, TypeError):
            logger.warning(f"Preco invalido '{price}' para formatacao. Retornando string 'N/A'.")
            return "N/A"
    else:
        price_val = float(price)

    if digits is not None:
        if not isinstance(digits, int) or digits < 0:
            logger.warning(f"Numero de digitos invalido '{digits}'. Usando formatacao padrao do simbolo.")
            digits = None
        else:
            return f"{price_val:.{digits}f}"

    normalized_symbol_fmt = symbol.upper().replace("/", "")
    if "JPY" in normalized_symbol_fmt:
        return f"{price_val:.3f}"
    elif any(p in normalized_symbol_fmt for p in ["XAU", "XAG"]):
        return f"{price_val:.2f}"
    else:
        return f"{price_val:.5f}"


def round_to_tick_size(price: float, tick_size: float = 0.00001) -> float:
    """
    Arredonda o preco para o multiplo mais proximo do tamanho do tick.
    """
    if not isinstance(price, (float, int, np.number)) or not isinstance(tick_size, (float, int, np.number)):
         logger.warning(f"Tipos invalidos para round_to_tick_size: price={type(price)}, tick_size={type(tick_size)}. Retornando preco original.")
         return float(price) if isinstance(price, (float, int, np.number)) else 0.0

    if tick_size <= 1e-9:
        return price
    try:
        price_decimal = Decimal(str(price))
        tick_size_decimal = Decimal(str(tick_size))
        
        if price_decimal.is_nan() or price_decimal.is_infinite():
            logger.warning(f"Preco NaN ou Inf ({price}) nao pode ser arredondado. Retornando como esta.")
            return price

        quotient = (price_decimal / tick_size_decimal)
        rounded_quotient = quotient.quantize(Decimal('1'), rounding=ROUND_HALF_UP)
        final_rounded_price = rounded_quotient * tick_size_decimal
        return float(final_rounded_price)

    except InvalidOperation:
        logger.warning(f"Operacao Decimal invalida ao arredondar preco {price} com tick_size {tick_size}. Usando math.round.")
        if tick_size == 0: return price
        return round(price / tick_size) * tick_size
    except Exception as e_round_tick:
        logger.exception(f"Erro inesperado em round_to_tick_size para preco {price}, tick {tick_size}:")
        return price


def calculate_forex_position_size(account_balance: float,
                          risk_percent_of_balance: float, 
                          stop_loss_pips_val: float,
                          pip_value_per_lot_in_acct_ccy: float, 
                          min_lot_size: float = 0.01,
                          lot_step: float = 0.01,
                          max_lot_size: float = 100.0) -> float:
    """
    Calcula o tamanho da posicao em lotes para Forex, baseado no risco percentual do saldo.
    """
    if not (account_balance > 0 and risk_percent_of_balance > 0 and
            stop_loss_pips_val > 0 and pip_value_per_lot_in_acct_ccy > 0):
        logger.warning(f"Parametros invalidos para calculate_forex_position_size: "
                       f"Balanco={account_balance}, Risco%={risk_percent_of_balance}, "
                       f"SLPips={stop_loss_pips_val}, PipVal={pip_value_per_lot_in_acct_ccy}. Retornando lote minimo.")
        return min_lot_size

    risk_amount_in_account_ccy = account_balance * risk_percent_of_balance
    risk_per_lot_in_acct_ccy = stop_loss_pips_val * pip_value_per_lot_in_acct_ccy

    if risk_per_lot_in_acct_ccy <= 1e-9:
        logger.warning("Risco por lote e zero ou negativo. Nao e possivel calcular tamanho da posicao. Retornando lote minimo.")
        return min_lot_size

    calculated_lot_size = risk_amount_in_account_ccy / risk_per_lot_in_acct_ccy

    if lot_step > 0:
        calculated_lot_size = math.floor(calculated_lot_size / lot_step) * lot_step

    final_lot_size = round(max(min_lot_size, min(calculated_lot_size, max_lot_size)), 2)
    return final_lot_size


def calculate_risk_reward_ratio(entry_price_val: float,
                               stop_loss_price_val: float,
                               take_profit_price_val: float) -> float:
    """Calcula a relacao Risco/Recompensa."""
    if not all(isinstance(p, (float, int, np.number)) for p in [entry_price_val, stop_loss_price_val, take_profit_price_val]):
        logger.warning("Valores nao numericos em calculate_risk_reward_ratio.")
        return 0.0

    potential_risk_abs_val = abs(entry_price_val - stop_loss_price_val)
    potential_reward_abs_val = abs(take_profit_price_val - entry_price_val)

    if potential_risk_abs_val < 1e-9:
        return float('inf') if potential_reward_abs_val > 1e-9 else 0.0
    
    return potential_reward_abs_val / potential_risk_abs_val


def convert_pips_to_price_diff(pips_value: float, symbol_string: str = "EURUSD") -> float:
    """Converte um valor em pips para a diferenca de preco correspondente."""
    normalized_symbol_val = normalize_forex_symbol(symbol_string)
    if "JPY" in normalized_symbol_val:
        return pips_value * 0.01
    elif "XAU" in normalized_symbol_val or "XAG" in normalized_symbol_val:
        return pips_value * 0.01
    else:
        return pips_value * 0.0001


def convert_price_diff_to_pips(price_diff_val: float, symbol_string: str = "EURUSD") -> float:
    """Converte uma diferenca de preco para o valor correspondente em pips."""
    normalized_symbol_val = normalize_forex_symbol(symbol_string)
    pip_decimal_unit: float
    if "JPY" in normalized_symbol_val:
        pip_decimal_unit = 0.01
    elif "XAU" in normalized_symbol_val or "XAG" in normalized_symbol_val:
        pip_decimal_unit = 0.01
    else:
        pip_decimal_unit = 0.0001

    if abs(pip_decimal_unit) < 1e-9:
        logger.warning(f"Unidade de pip e zero para {symbol_string}. Retornando 0 pips.")
        return 0.0
    
    return round(price_diff_val / pip_decimal_unit, 1)


def calculate_spread_cost_in_account_ccy(spread_in_pips_val: float,
                         lot_size_val: float,
                         pip_value_per_lot_in_acct_ccy_val: float) -> float:
    """Calcula o custo do spread na moeda da conta."""
    if not (spread_in_pips_val >= 0 and lot_size_val >= 0 and pip_value_per_lot_in_acct_ccy_val >= 0):
        logger.warning("Valores negativos em calculate_spread_cost. Retornando 0.")
        return 0.0
    return spread_in_pips_val * lot_size_val * pip_value_per_lot_in_acct_ccy_val


def get_forex_trading_session(timestamp_to_check_utc: Optional[datetime] = None) -> str:
    """
    Determina a sessao de trading Forex (Asia, London, NewYork, Overlap_*, Closed_Hours)
    baseado no timestamp UTC.
    """
    cfg = get_config_helpers()
    
    if timestamp_to_check_utc is None:
        current_utc_val = datetime.now(timezone.utc)
    else:
        if timestamp_to_check_utc.tzinfo is None:
            current_utc_val = timestamp_to_check_utc.replace(tzinfo=timezone.utc)
        else:
            current_utc_val = timestamp_to_check_utc.astimezone(timezone.utc)

    current_hour_val = current_utc_val.hour

    asia_opens_utc = getattr(cfg.SESSION_CONFIG.get('ASIA',{}), 'start_hour', 23) if hasattr(cfg, 'SESSION_CONFIG') else 23
    asia_closes_utc = getattr(cfg.SESSION_CONFIG.get('ASIA',{}), 'end_hour', 8) if hasattr(cfg, 'SESSION_CONFIG') else 8
    london_opens_utc = getattr(cfg.SESSION_CONFIG.get('LONDON',{}), 'start_hour', 7) if hasattr(cfg, 'SESSION_CONFIG') else 7
    london_closes_utc = getattr(cfg.SESSION_CONFIG.get('LONDON',{}), 'end_hour', 16) if hasattr(cfg, 'SESSION_CONFIG') else 16
    newyork_opens_utc = getattr(cfg.SESSION_CONFIG.get('NEWYORK',{}), 'start_hour', 13) if hasattr(cfg, 'SESSION_CONFIG') else 13
    newyork_closes_utc = getattr(cfg.SESSION_CONFIG.get('NEWYORK',{}), 'end_hour', 22) if hasattr(cfg, 'SESSION_CONFIG') else 22


    is_asia = (current_hour_val >= asia_opens_utc or current_hour_val < asia_closes_utc)
    is_london = (london_opens_utc <= current_hour_val < london_closes_utc)
    is_newyork = (newyork_opens_utc <= current_hour_val < newyork_closes_utc)

    if is_london and is_newyork: return 'Overlap_London_NY'
    if is_asia and is_london: return 'Overlap_Asia_London'
    if is_london: return 'London'
    if is_newyork: return 'NewYork'
    if is_asia: return 'Asia'
    
    return 'Closed_Session_Hours'


def calculate_lot_size_from_risk_pct(balance: float,
                           risk_percentage_val: float, 
                           stop_loss_pips_val: float,
                           symbol_str: str = "EURUSD",
                           account_currency_val: str = "USD",
                           current_exchange_rate_quote_to_acct: Optional[float] = None, 
                           quote_currency_for_pip_val: Optional[str] = None) -> float:
    """Calcula o tamanho do lote (em lotes padrao) baseado no risco percentual do saldo."""
    if not (balance > 0 and risk_percentage_val > 0 and stop_loss_pips_val > 0):
        logger.warning("Parametros invalidos para calculate_lot_size_from_risk_pct. Retornando 0.01.")
        return 0.01

    risk_amount_val_total = balance * (risk_percentage_val / 100.0)

    pip_value_for_one_std_lot_in_acct_ccy = calculate_pip_value(
        symbol=symbol_str,
        lot_size=1.0,
        account_currency=account_currency_val,
        current_exchange_rate_quote_to_acct_currency=current_exchange_rate_quote_to_acct,
        quote_currency_inferred=quote_currency_for_pip_val
    )

    if pip_value_for_one_std_lot_in_acct_ccy <= 1e-9:
        logger.error(f"Valor do pip para {symbol_str} em {account_currency_val} e invalido ({pip_value_for_one_std_lot_in_acct_ccy}). "
                       "Nao e possivel calcular o tamanho do lote. Verifique taxas de cambio e moedas.")
        return 0.01

    calculated_lot_size_val = risk_amount_val_total / (stop_loss_pips_val * pip_value_for_one_std_lot_in_acct_ccy)
    return max(0.01, round(calculated_lot_size_val, 2))


def is_forex_market_open(current_time_to_check_utc: Optional[datetime] = None) -> bool:
    """Verifica se o mercado Forex global esta aberto no timestamp UTC fornecido."""
    cfg = get_config_helpers()
    if current_time_to_check_utc is None:
        now_to_check = datetime.now(timezone.utc)
    else:
        if current_time_to_check_utc.tzinfo is None:
            now_to_check = current_time_to_check_utc.replace(tzinfo=timezone.utc)
        else:
            now_to_check = current_time_to_check_utc.astimezone(timezone.utc)

    weekday_check = now_to_check.weekday()
    hour_check = now_to_check.hour

    market_close_hour_friday = getattr(cfg, 'FOREX_MARKET_CLOSE_HOUR_FRIDAY_UTC', 21)
    market_open_hour_sunday = getattr(cfg, 'FOREX_MARKET_OPEN_HOUR_SUNDAY_UTC', 21)

    if weekday_check == 4: 
        return hour_check < market_close_hour_friday
    elif weekday_check == 5: 
        return False
    elif weekday_check == 6: 
        return hour_check >= market_open_hour_sunday
    else: 
        return True


def calculate_compound_returns_on_balance(initial_balance_val: float,
                             returns_list_decimal: List[float]) -> float:
    """
    Calcula o saldo final apos uma serie de retornos decimais compostos.
    """
    current_balance_val = initial_balance_val
    for ret_decimal in returns_list_decimal:
        current_balance_val *= (1.0 + ret_decimal)
    return current_balance_val


def calculate_sharpe_ratio_simple(returns_series_val: List[float],
                          risk_free_rate_annual_val: float = 0.02,
                          periods_in_year_val: int = 252) -> float:
    """Calcula o Sharpe Ratio (simplificado) a partir de uma serie de retornos de periodo."""
    if not returns_series_val or len(returns_series_val) < 2: return 0.0
    returns_arr_sr = np.array(returns_series_val, dtype=float)

    std_dev_returns_sr = np.std(returns_arr_sr)
    if std_dev_returns_sr < 1e-9:
        avg_return_period_sr = np.mean(returns_arr_sr)
        risk_free_per_period_sr = risk_free_rate_annual_val / periods_in_year_val
        return 10.0 if avg_return_period_sr > risk_free_per_period_sr else 0.0

    avg_return_sr = np.mean(returns_arr_sr)
    annualized_avg_return_sr = avg_return_sr * periods_in_year_val
    annualized_std_dev_sr = std_dev_returns_sr * np.sqrt(periods_in_year_val)

    if annualized_std_dev_sr == 0: return 0.0
    sharpe_ratio_val = (annualized_avg_return_sr - risk_free_rate_annual_val) / annualized_std_dev_sr
    return sharpe_ratio_val if not np.isnan(sharpe_ratio_val) else 0.0


def calculate_max_drawdown_from_equity(equity_values_list: List[float]) -> Tuple[float, int, int]:
    """Calcula o drawdown maximo percentual de uma curva de equity."""
    if not equity_values_list or len(equity_values_list) < 2: return 0.0, 0, 0

    peak_equity = equity_values_list[0]
    peak_equity_idx = 0
    max_dd_percent_val = 0.0
    dd_peak_idx_val = 0
    dd_trough_idx_val = 0

    for i_dd, current_equity_dd in enumerate(equity_values_list):
        if current_equity_dd > peak_equity:
            peak_equity = current_equity_dd
            peak_equity_idx = i_dd
        
        current_dd_val_pct = 0.0
        if peak_equity > 1e-9:
            current_dd_val_pct = (peak_equity - current_equity_dd) / peak_equity
        
        if current_dd_val_pct > max_dd_percent_val:
            max_dd_percent_val = current_dd_val_pct
            dd_peak_idx_val = peak_equity_idx
            dd_trough_idx_val = i_dd
            
    return max_dd_percent_val, dd_peak_idx_val, dd_trough_idx_val


def normalize_forex_symbol(symbol_input: str) -> str:
    """Normaliza um simbolo de par de moedas para um formato padrao (ex: "EURUSD")."""
    if not isinstance(symbol_input, str): return ""
    normalized_s = symbol_input.strip().upper().replace("/", "").replace("-", "").replace(".", "").replace("_", "")
    return normalized_s


def is_major_forex_pair(symbol_to_eval: str) -> bool:
    """Verifica se o simbolo e um dos principais pares de Forex."""
    MAJOR_PAIRS_SET = {
        "EURUSD", "GBPUSD", "USDJPY", "USDCHF",
        "AUDUSD", "USDCAD", "NZDUSD"
    }
    return normalize_forex_symbol(symbol_to_eval) in MAJOR_PAIRS_SET


def calculate_pearson_correlation(series_a: List[float], series_b: List[float]) -> float:
    """Calcula o coeficiente de correlacao de Pearson entre duas series de dados."""
    if len(series_a) != len(series_b) or len(series_a) < 2: return 0.0

    s1_arr_corr = np.array(series_a, dtype=float)
    s2_arr_corr = np.array(series_b, dtype=float)
    valid_mask_corr = ~np.isnan(s1_arr_corr) & ~np.isnan(s2_arr_corr)
    s1_clean_corr, s2_clean_corr = s1_arr_corr[valid_mask_corr], s2_arr_corr[valid_mask_corr]

    if len(s1_clean_corr) < 2: return 0.0
    if np.std(s1_clean_corr) < 1e-9 or np.std(s2_clean_corr) < 1e-9: return 0.0

    corr_matrix = np.corrcoef(s1_clean_corr, s2_clean_corr)
    correlation_val = corr_matrix[0, 1]
    return correlation_val if not np.isnan(correlation_val) else 0.0


def time_until_next_session_starts(target_session_name_str: str,
                                  current_time_val_utc: Optional[datetime] = None) -> timedelta:
    """Calcula o tempo restante ate o inicio da proxima sessao de trading especificada."""
    cfg = get_config_helpers()
    
    if current_time_val_utc is None:
        now_utc_session = datetime.now(timezone.utc)
    else:
        now_utc_session = current_time_val_utc.astimezone(timezone.utc) if current_time_val_utc.tzinfo else current_time_val_utc.replace(tzinfo=timezone.utc)

    session_start_hours_map_utc: Dict[str, int] = {
        'ASIA': getattr(cfg.SESSION_CONFIG.get('ASIA',{}), 'start_hour', 23) if hasattr(cfg, 'SESSION_CONFIG') else 23,
        'LONDON': getattr(cfg.SESSION_CONFIG.get('LONDON',{}), 'start_hour', 7) if hasattr(cfg, 'SESSION_CONFIG') else 7,
        'NEWYORK': getattr(cfg.SESSION_CONFIG.get('NEWYORK',{}), 'start_hour', 13) if hasattr(cfg, 'SESSION_CONFIG') else 13
    }
    
    session_name_upper = target_session_name_str.upper()
    if session_name_upper not in session_start_hours_map_utc:
        logger.warning(f"Nome de sessao '{target_session_name_str}' desconhecido em time_until_next_session_starts.")
        return timedelta(0)

    target_start_hour = session_start_hours_map_utc[session_name_upper]

    next_target_dt = now_utc_session.replace(hour=target_start_hour, minute=0, second=0, microsecond=0)
    if next_target_dt <= now_utc_session:
        next_target_dt += timedelta(days=1)

    market_open_sun_hour_cfg = getattr(cfg, 'FOREX_MARKET_OPEN_HOUR_SUNDAY_UTC', 21)

    while not is_forex_market_open(next_target_dt) or \
          (next_target_dt.weekday() == 6 and next_target_dt.hour < market_open_sun_hour_cfg and target_start_hour < market_open_sun_hour_cfg) or \
          (next_target_dt.weekday() == 5) :
        next_target_dt += timedelta(days=1)
        next_target_dt = next_target_dt.replace(hour=target_start_hour, minute=0, second=0, microsecond=0)
        if next_target_dt.weekday() == 6 and session_name_upper == 'ASIA' and target_start_hour >= market_open_sun_hour_cfg:
            break

    time_difference_val = next_target_dt - now_utc_session
    return time_difference_val if time_difference_val.total_seconds() > 0 else timedelta(0)


def validate_stop_loss_take_profit(entry_price_val: float, stop_loss_val: float,
                  take_profit_val: float, trade_side_str: str,
                  min_stop_dist_pips_val_param: Optional[float] = None, # Renomeado para evitar conflito
                  min_tp_dist_pips_val_param: Optional[float] = None,   # Renomeado para evitar conflito
                  symbol_for_pips_val_param: Optional[str] = None) -> Tuple[bool, str]: # Renomeado para evitar conflito
    """Valida SL e TP em relacao a entrada, lado, e distancias minimas."""
    cfg = get_config_helpers()
    risk_params = get_risk_params_helpers() # Obter RISK_PARAMS

    # Usar valores de RISK_PARAMS se disponiveis, senao os parametros da funcao, senao defaults internos
    min_stop_dist_pips = min_stop_dist_pips_val_param if min_stop_dist_pips_val_param is not None else getattr(risk_params, 'MIN_STOP_DISTANCE_PIPS', 5.0)
    min_tp_dist_pips = min_tp_dist_pips_val_param if min_tp_dist_pips_val_param is not None else getattr(risk_params, 'MIN_STOP_DISTANCE_PIPS', 5.0) # Usar o mesmo ou um especifico para TP
    symbol_for_pips = symbol_for_pips_val_param if symbol_for_pips_val_param is not None else getattr(cfg, 'SYMBOL', "EURUSD")


    if not all(isinstance(v, (float, int, np.number)) for v in [entry_price_val, stop_loss_val, take_profit_val]):
        return False, "Valores de preco invalidos (nao numericos)."

    pip_unit_validate = 0.01 if "JPY" in symbol_for_pips.upper() else 0.0001
    min_stop_price_diff_val = min_stop_dist_pips * pip_unit_validate
    min_tp_price_diff_val = min_tp_dist_pips * pip_unit_validate

    side_lower = trade_side_str.lower()
    if side_lower == 'buy':
        if stop_loss_val >= entry_price_val: return False, "SL (Compra) deve ser < Preco de Entrada."
        if take_profit_val <= entry_price_val: return False, "TP (Compra) deve ser > Preco de Entrada."
        if abs(entry_price_val - stop_loss_val) < min_stop_price_diff_val - 1e-7: 
            return False, f"SL (Compra) muito proximo. Min: {min_stop_dist_pips} pips."
        if abs(take_profit_val - entry_price_val) < min_tp_price_diff_val - 1e-7:
            return False, f"TP (Compra) muito proximo. Min: {min_tp_dist_pips} pips."
    elif side_lower == 'sell':
        if stop_loss_val <= entry_price_val: return False, "SL (Venda) deve ser > Preco de Entrada."
        if take_profit_val >= entry_price_val: return False, "TP (Venda) deve ser < Preco de Entrada."
        if abs(stop_loss_val - entry_price_val) < min_stop_price_diff_val - 1e-7:
            return False, f"SL (Venda) muito proximo. Min: {min_stop_dist_pips} pips."
        if abs(entry_price_val - take_profit_val) < min_tp_price_diff_val - 1e-7:
            return False, f"TP (Venda) muito proximo. Min: {min_tp_dist_pips} pips."
    else:
        return False, f"Lado da trade desconhecido: '{trade_side_str}'. Use 'buy' ou 'sell'."
    return True, "SL/TP Validos."


def format_duration_to_readable_str(total_seconds_val: float) -> str:
    """Formata uma duracao em segundos para uma string legivel (ex: "1d 2h 30m 15s")."""
    if not isinstance(total_seconds_val, (int, float)) or total_seconds_val < 0: return "0s"
    
    total_seconds_int_fmt = int(total_seconds_val)

    days_fmt, rem_secs_fmt = divmod(total_seconds_int_fmt, SECONDS_IN_DAY)
    hours_fmt, rem_secs_fmt = divmod(rem_secs_fmt, SECONDS_IN_HOUR)
    minutes_fmt, secs_fmt = divmod(rem_secs_fmt, SECONDS_IN_MINUTE)

    parts_str_list: List[str] = []
    if days_fmt > 0: parts_str_list.append(f"{days_fmt}d")
    if hours_fmt > 0: parts_str_list.append(f"{hours_fmt}h")
    if minutes_fmt > 0: parts_str_list.append(f"{minutes_fmt}m")
    if secs_fmt > 0 or not parts_str_list: 
        parts_str_list.append(f"{secs_fmt}s")
    
    return " ".join(parts_str_list) if parts_str_list else "0s"


def calculate_trade_expectancy(win_amounts_list: List[float],
                               loss_amounts_positive_list: List[float]) -> float:
    """Calcula a Expectancy (Esperanca Matematica) de uma estrategia."""
    num_total_wins = len(win_amounts_list)
    num_total_losses = len(loss_amounts_positive_list)
    total_trades_exp = num_total_wins + num_total_losses

    if total_trades_exp == 0: return 0.0

    win_rate_exp = num_total_wins / total_trades_exp
    avg_win_amt_exp = np.mean(win_amounts_list) if num_total_wins > 0 else 0.0
    avg_loss_amt_exp = np.mean(loss_amounts_positive_list) if num_total_losses > 0 else 0.0
    
    expectancy_value = (win_rate_exp * avg_win_amt_exp) - ((1.0 - win_rate_exp) * avg_loss_amt_exp)
    return expectancy_value


def round_value_to_broker_precision(value_to_round_prec: Union[float, int, Decimal, str],
                                   decimal_precision_val: int = 2) -> float:
    """Arredonda um valor para a precisao decimal especificada, usando Decimal."""
    if not isinstance(decimal_precision_val, int) or decimal_precision_val < 0:
        logger.warning(f"Precisao decimal invalida ({decimal_precision_val}). Usando 0.")
        decimal_precision_val = 0
    
    try:
        value_str_prec = str(value_to_round_prec)
        value_decimal_prec = Decimal(value_str_prec)

        if value_decimal_prec.is_nan() or value_decimal_prec.is_infinite():
            logger.warning(f"Valor NaN ou Inf ({value_to_round_prec}) nao pode ser arredondado. Retornando como float.")
            return float(value_to_round_prec)

        quantizer_format_str: str
        if decimal_precision_val > 0:
            quantizer_format_str = '0.' + '0' * (decimal_precision_val -1) + '1'
        else: 
            quantizer_format_str = '1'

        quantizer_decimal_obj = Decimal(quantizer_format_str)
        rounded_decimal_val_prec = value_decimal_prec.quantize(quantizer_decimal_obj, rounding=ROUND_HALF_UP)
        return float(rounded_decimal_val_prec)

    except (InvalidOperation, ValueError, TypeError) as e_dec_round:
        logger.warning(f"Erro Decimal ao arredondar '{value_to_round_prec}' para {decimal_precision_val} casas: {e_dec_round}. Usando round() padrao.")
        try:
            return round(float(value_to_round_prec), decimal_precision_val)
        except Exception as e_round_fallback:
             logger.error(f"Erro no fallback de round() para '{value_to_round_prec}': {e_round_fallback}")
             return float(value_to_round_prec) if isinstance(value_to_round_prec, (float, int, np.number)) else 0.0


SECONDS_IN_MINUTE: int = 60
SECONDS_IN_HOUR: int = SECONDS_IN_MINUTE * 60
SECONDS_IN_DAY: int = SECONDS_IN_HOUR * 24
TRADING_DAYS_PER_YEAR: int = 252