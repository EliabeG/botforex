# utils/helpers.py
"""Funções auxiliares diversas para o trading bot."""
import math
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation # Adicionado InvalidOperation
from typing import Union, Optional, Tuple, List, Any, TYPE_CHECKING # Adicionado Any, TYPE_CHECKING
from datetime import datetime, timedelta, timezone, date # Adicionado timezone, date

import pytz # Para manipulação de fusos horários em get_trading_session
import numpy as np

# Logger pode ser útil para avisos em helpers, se necessário
from .logger import setup_logger # Importar setup_logger
logger = setup_logger("utils_helpers_logger") # Nome específico para este logger

if TYPE_CHECKING: # Evitar dependência circular em tempo de execução para type hinting
    from config.settings import TradingConfig # Para CONFIG, se usado aqui
    CONFIG_TYPE = TradingConfig
else:
    CONFIG_TYPE = Any


# Global CONFIG (carregado de forma segura)
_CONFIG_INSTANCE: Optional[CONFIG_TYPE] = None

def get_config_helpers() -> CONFIG_TYPE:
    global _CONFIG_INSTANCE
    if _CONFIG_INSTANCE is None:
        from config.settings import CONFIG as cfg_settings
        _CONFIG_INSTANCE = cfg_settings
    return _CONFIG_INSTANCE


def calculate_pip_value(symbol: str,
                       lot_size: float = 1.0,
                       account_currency: str = "USD",
                       current_exchange_rate_quote_to_acct_currency: Optional[float] = None,
                       quote_currency_inferred: Optional[str] = None) -> float: # Renomeado quote_currency
    """
    Calcula o valor do pip para um par de moedas na moeda da conta.

    Args:
        symbol: Par de moedas (ex: "EURUSD").
        lot_size: Tamanho do lote (ex: 1.0 para 1 lote padrão).
        account_currency: Moeda da conta (ex: "USD").
        current_exchange_rate_quote_to_acct_currency: Taxa de câmbio ATUAL da moeda de cotação
                                                 para a moeda da conta. Essencial se a moeda de cotação
                                                 não for a moeda da conta.
                                                 Ex: Para USDJPY em conta EUR, seria JPY/EUR.
                                                 Ex: Para EURUSD em conta USD, quote_currency=USD, rate=1.0.
        quote_currency_inferred: A moeda de cotação do par (ex: USD para EURUSD, JPY para USDJPY).
                        Se não fornecido, tenta inferir do símbolo.

    Returns:
        Valor do pip na moeda da conta. Retorna 0.0 se dados insuficientes.
    """
    if lot_size <= 0: return 0.0

    cfg = get_config_helpers()
    contract_size = getattr(cfg, 'CONTRACT_SIZE', 100000.0)

    normalized_symbol_hlp = symbol.upper().replace("/", "") # Renomeado

    if quote_currency_inferred is None:
        if len(normalized_symbol_hlp) == 6:
            quote_currency_inferred = normalized_symbol_hlp[3:]
        elif len(normalized_symbol_hlp) == 7 and normalized_symbol_hlp.endswith("M"): # Ex: EURUSDM
             quote_currency_inferred = normalized_symbol_hlp[3:6]
        else:
            logger.warning(f"Não foi possível inferir a moeda de cotação de '{symbol}' para valor do pip. Retornando 0.")
            return 0.0 # Não é possível prosseguir sem moeda de cotação

    pip_size_in_quote_ccy_hlp: float # Renomeado
    if "JPY" in quote_currency_inferred.upper():
        pip_size_in_quote_ccy_hlp = 0.01
    elif any(p in normalized_symbol_hlp for p in ["XAU", "XAG", "BTC", "ETH"]): # Metais e Criptos podem ter precisão diferente
        # Esta parte é específica do broker e do ativo. Exemplo:
        if "XAU" in normalized_symbol_hlp: pip_size_in_quote_ccy_hlp = 0.01 # Ouro geralmente 2 casas
        else: pip_size_in_quote_ccy_hlp = 0.0001 # Default para outros
    else:
        pip_size_in_quote_ccy_hlp = 0.0001

    # Valor de um pip, para um lote, na moeda de cotação
    pip_value_one_lot_in_quote_ccy = pip_size_in_quote_ccy_hlp * contract_size # Valor para 1 lote

    # Valor do pip para o lot_size dado, na moeda de cotação
    pip_value_total_lots_in_quote_ccy = pip_value_one_lot_in_quote_ccy * lot_size

    if quote_currency_inferred.upper() == account_currency.upper():
        return pip_value_total_lots_in_quote_ccy
    else:
        # Precisamos da taxa de câmbio para converter da moeda de cotação para a moeda da conta.
        # current_exchange_rate_quote_to_acct_currency deve ser: (1 unidade de quote_currency) = X unidades de account_currency
        # Ex: Quote=JPY, Acct=USD. Precisa de JPY/USD (ou 1/USDJPY_price).
        # Ex: Quote=EUR, Acct=USD. Precisa de EUR/USD (ou EURUSD_price).
        # Ex: Quote=USD, Acct=EUR. Precisa de USD/EUR (ou 1/EURUSD_price).
        if current_exchange_rate_quote_to_acct_currency is None or current_exchange_rate_quote_to_acct_currency <= 0:
            logger.error(f"Taxa de câmbio ({quote_currency_inferred}/{account_currency}) não fornecida ou inválida para {symbol}. "
                           f"Não é possível converter o valor do pip para a moeda da conta.")
            # Não se deve retornar o valor na moeda de cotação se o chamador espera na moeda da conta.
            # É melhor levantar um erro ou retornar um valor que indique falha (ex: 0 ou NaN).
            return 0.0 # Ou raise ValueError(...)

        return pip_value_total_lots_in_quote_ccy * current_exchange_rate_quote_to_acct_currency


def format_price(price: Union[float, int, Decimal], symbol: str = "EURUSD", digits: Optional[int] = None) -> str:
    """
    Formata o preço com a precisão correta para o símbolo ou com 'digits' especificado.
    """
    if not isinstance(price, (float, int, Decimal, np.number)): # np.number para tipos numpy
        try:
            price_val = float(price) # Tentar converter, renomeado
        except (ValueError, TypeError):
            logger.warning(f"Preço inválido '{price}' para formatação. Retornando string 'N/A'.")
            return "N/A" # Retornar algo que indique erro
    else:
        price_val = float(price) # Converter Decimal ou int para float para formatação f-string


    if digits is not None:
        if not isinstance(digits, int) or digits < 0:
            logger.warning(f"Número de dígitos inválido '{digits}'. Usando formatação padrão do símbolo.")
            digits = None # Resetar para usar lógica do símbolo
        else:
            return f"{price_val:.{digits}f}"

    normalized_symbol_fmt = symbol.upper().replace("/", "") # Renomeado
    if "JPY" in normalized_symbol_fmt:
        return f"{price_val:.3f}"
    elif any(p in normalized_symbol_fmt for p in ["XAU", "XAG"]): # Ouro e Prata geralmente 2 casas
        return f"{price_val:.2f}"
    # Adicionar mais lógicas para outros tipos de ativos (índices, criptos) se necessário
    else: # Maioria dos pares Forex
        return f"{price_val:.5f}"


def round_to_tick_size(price: float, tick_size: float = 0.00001) -> float:
    """
    Arredonda o preço para o múltiplo mais próximo do tamanho do tick.
    """
    if not isinstance(price, (float, int, np.number)) or not isinstance(tick_size, (float, int, np.number)):
         logger.warning(f"Tipos inválidos para round_to_tick_size: price={type(price)}, tick_size={type(tick_size)}. Retornando preço original.")
         return float(price) if isinstance(price, (float, int, np.number)) else 0.0


    if tick_size <= 1e-9: # Tick size muito pequeno ou zero
        # logger.debug(f"Tamanho do tick inválido ou muito pequeno ({tick_size}). Retornando preço original.")
        return price
    try:
        price_decimal = Decimal(str(price))
        tick_size_decimal = Decimal(str(tick_size))
        # Arredondar para o número de casas decimais do tick_size
        num_decimals_tick = abs(tick_size_decimal.as_tuple().exponent)

        # (Preço / TickSize) arredondado para inteiro * TickSize
        # Usar quantize para arredondar para o número de casas decimais do tick_size após a divisão
        # ou diretamente com a lógica de múltiplo
        # Ex: price=1.23456, tick_size=0.00005. Rounded = floor(1.23456 / 0.00005 + 0.5) * 0.00005
        # ou (price_decimal / tick_size_decimal).quantize(Decimal('1'), rounding=ROUND_HALF_UP) * tick_size_decimal

        if price_decimal.is_nan() or price_decimal.is_infinite(): # Lidar com NaN/Inf
            logger.warning(f"Preço NaN ou Inf ({price}) não pode ser arredondado. Retornando como está.")
            return price


        # Multiplicador para arredondamento: 1 / tick_size
        # multiplier = 1 / tick_size
        # return round(price * multiplier) / multiplier
        # A abordagem com Decimal é mais robusta para evitar erros de ponto flutuante cumulativos.
        quotient = (price_decimal / tick_size_decimal)
        # Arredondar o quociente para o inteiro mais próximo
        rounded_quotient = quotient.quantize(Decimal('1'), rounding=ROUND_HALF_UP)
        final_rounded_price = rounded_quotient * tick_size_decimal
        return float(final_rounded_price)

    except InvalidOperation:
        logger.warning(f"Operação Decimal inválida ao arredondar preço {price} com tick_size {tick_size}. Usando math.round.")
        # Fallback para método original se Decimal falhar (improvável com inputs float válidos)
        if tick_size == 0: return price # Evitar divisão por zero
        return round(price / tick_size) * tick_size
    except Exception as e_round_tick:
        logger.exception(f"Erro inesperado em round_to_tick_size para preço {price}, tick {tick_size}:")
        return price # Retornar original em caso de erro inesperado


def calculate_forex_position_size(account_balance: float,
                          risk_percent_of_balance: float, # Ex: 0.01 para 1%
                          stop_loss_pips_val: float,
                          pip_value_per_lot_in_acct_ccy: float, # Calculado externamente
                          min_lot_size: float = 0.01,
                          lot_step: float = 0.01,
                          max_lot_size: float = 100.0) -> float: # Adicionado max_lot_size
    """
    Calcula o tamanho da posição em lotes para Forex, baseado no risco percentual do saldo.
    """
    if not (account_balance > 0 and risk_percent_of_balance > 0 and
            stop_loss_pips_val > 0 and pip_value_per_lot_in_acct_ccy > 0):
        logger.warning(f"Parâmetros inválidos para calculate_forex_position_size: "
                       f"Balanço={account_balance}, Risco%={risk_percent_of_balance}, "
                       f"SLPips={stop_loss_pips_val}, PipVal={pip_value_per_lot_in_acct_ccy}. Retornando lote mínimo.")
        return min_lot_size

    risk_amount_in_account_ccy = account_balance * risk_percent_of_balance # Renomeado

    # Risco total em moeda da conta se usasse 1 lote: stop_loss_pips * pip_value_per_lot
    risk_per_lot_in_acct_ccy = stop_loss_pips_val * pip_value_per_lot_in_acct_ccy # Renomeado

    if risk_per_lot_in_acct_ccy <= 1e-9: # Risco por lote é zero ou negativo
        logger.warning("Risco por lote é zero ou negativo. Não é possível calcular tamanho da posição. Retornando lote mínimo.")
        return min_lot_size

    calculated_lot_size = risk_amount_in_account_ccy / risk_per_lot_in_acct_ccy # Renomeado

    # Arredondar para baixo para o múltiplo de lot_step mais próximo
    if lot_step > 0:
        # Ex: lots=0.127, step=0.01 -> floor(0.127/0.01)*0.01 = floor(12.7)*0.01 = 12*0.01 = 0.12
        # Ex: lots=0.127, step=0.1  -> floor(0.127/0.1)*0.1  = floor(1.27)*0.1  = 1*0.1  = 0.1
        calculated_lot_size = math.floor(calculated_lot_size / lot_step) * lot_step

    # Aplicar limites de lote mínimo e máximo, e arredondar para 2 casas decimais (comum em Forex)
    final_lot_size = round(max(min_lot_size, min(calculated_lot_size, max_lot_size)), 2) # Renomeado
    return final_lot_size


def calculate_risk_reward_ratio(entry_price_val: float,
                               stop_loss_price_val: float,
                               take_profit_price_val: float) -> float:
    """Calcula a relação Risco/Recompensa."""
    if not all(isinstance(p, (float, int, np.number)) for p in [entry_price_val, stop_loss_price_val, take_profit_price_val]):
        logger.warning("Valores não numéricos em calculate_risk_reward_ratio.")
        return 0.0

    potential_risk_abs_val = abs(entry_price_val - stop_loss_price_val) # Renomeado
    potential_reward_abs_val = abs(take_profit_price_val - entry_price_val) # Renomeado

    if potential_risk_abs_val < 1e-9: # Usar tolerância pequena para risco zero
        # Se risco é zero e recompensa é positiva, R:R é "infinito"
        # Se ambos são zero, R:R é indefinido, retornar 0.
        return float('inf') if potential_reward_abs_val > 1e-9 else 0.0
    
    return potential_reward_abs_val / potential_risk_abs_val


def convert_pips_to_price_diff(pips_value: float, symbol_string: str = "EURUSD") -> float: # Renomeado
    """Converte um valor em pips para a diferença de preço correspondente."""
    normalized_symbol_val = normalize_forex_symbol(symbol_string) # Renomeado
    if "JPY" in normalized_symbol_val: # Geralmente para pares com JPY
        return pips_value * 0.01
    # Adicionar lógica para outros tipos de instrumentos (ex: XAUUSD, Índices) se necessário
    # XAUUSD (Ouro) geralmente tem 2 casas decimais, então um "pip" pode ser 0.01
    elif "XAU" in normalized_symbol_val or "XAG" in normalized_symbol_val:
        return pips_value * 0.01 # Assumindo que 1 pip = 0.01 para ouro/prata
    else: # Maioria dos pares Forex
        return pips_value * 0.0001


def convert_price_diff_to_pips(price_diff_val: float, symbol_string: str = "EURUSD") -> float: # Renomeado
    """Converte uma diferença de preço para o valor correspondente em pips."""
    normalized_symbol_val = normalize_forex_symbol(symbol_string)
    pip_decimal_unit: float # Renomeado
    if "JPY" in normalized_symbol_val:
        pip_decimal_unit = 0.01
    elif "XAU" in normalized_symbol_val or "XAG" in normalized_symbol_val:
        pip_decimal_unit = 0.01
    else:
        pip_decimal_unit = 0.0001

    if abs(pip_decimal_unit) < 1e-9: # Evitar divisão por zero
        logger.warning(f"Unidade de pip é zero para {symbol_string}. Retornando 0 pips.")
        return 0.0
    
    return round(price_diff_val / pip_decimal_unit, 1) # Arredondar pips para 1 casa decimal


def calculate_spread_cost_in_account_ccy(spread_in_pips_val: float, # Renomeado
                         lot_size_val: float,
                         pip_value_per_lot_in_acct_ccy_val: float) -> float: # Renomeado
    """Calcula o custo do spread na moeda da conta."""
    if not (spread_in_pips_val >= 0 and lot_size_val >= 0 and pip_value_per_lot_in_acct_ccy_val >= 0):
        logger.warning("Valores negativos em calculate_spread_cost. Retornando 0.")
        return 0.0
    return spread_in_pips_val * lot_size_val * pip_value_per_lot_in_acct_ccy_val


def get_forex_trading_session(timestamp_to_check_utc: Optional[datetime] = None) -> str: # Renomeado
    """
    Determina a sessão de trading Forex (Asia, London, NewYork, Overlap_*, Closed_Hours)
    baseado no timestamp UTC.
    Os horários são aproximados e podem variar com DST e brokers.
    """
    cfg = get_config_helpers() # Para acessar SESSION_CONFIG se movido para lá
    
    if timestamp_to_check_utc is None:
        current_utc_val = datetime.now(timezone.utc) # Renomeado
    else:
        if timestamp_to_check_utc.tzinfo is None: # Se for naive, assumir UTC
            current_utc_val = timestamp_to_check_utc.replace(tzinfo=timezone.utc)
        else: # Converter para UTC se já for aware
            current_utc_val = timestamp_to_check_utc.astimezone(timezone.utc)

    current_hour_val = current_utc_val.hour # Renomeado

    # Horários de referência em UTC (podem vir de CONFIG.SESSION_CONFIG)
    # Exemplo de como poderia ser usado com config:
    # asia_cfg = getattr(cfg, 'SESSION_CONFIG', {}).get('ASIA', {'start_hour': 23, 'end_hour': 8})
    # london_cfg = getattr(cfg, 'SESSION_CONFIG', {}).get('LONDON', {'start_hour': 7, 'end_hour': 16})
    # ny_cfg = getattr(cfg, 'SESSION_CONFIG', {}).get('NEWYORK', {'start_hour': 13, 'end_hour': 22})
    # Usando hardcoded como no original, mas com nomes mais claros:
    asia_opens_utc, asia_closes_utc = 23, 8 # Sydney/Tóquio (genérico)
    london_opens_utc, london_closes_utc = 7, 16
    newyork_opens_utc, newyork_closes_utc = 13, 22


    is_asia = (current_hour_val >= asia_opens_utc or current_hour_val < asia_closes_utc)
    is_london = (london_opens_utc <= current_hour_val < london_closes_utc)
    is_newyork = (newyork_opens_utc <= current_hour_val < newyork_closes_utc)

    if is_london and is_newyork: return 'Overlap_London_NY'
    if is_asia and is_london: return 'Overlap_Asia_London' # Menos comum, mas possível
    if is_london: return 'London'
    if is_newyork: return 'NewYork'
    if is_asia: return 'Asia'
    
    return 'Closed_Session_Hours' # Fora dos horários principais


def calculate_lot_size_from_risk_pct(balance: float,
                           risk_percentage_val: float, # Ex: 1.0 para 1%
                           stop_loss_pips_val: float,
                           symbol_str: str = "EURUSD",
                           account_currency_val: str = "USD",
                           current_exchange_rate_quote_to_acct: Optional[float] = None, # Para conversão do valor do pip
                           quote_currency_for_pip_val: Optional[str] = None) -> float: # Renomeado
    """Calcula o tamanho do lote (em lotes padrão) baseado no risco percentual do saldo."""
    if not (balance > 0 and risk_percentage_val > 0 and stop_loss_pips_val > 0):
        logger.warning("Parâmetros inválidos para calculate_lot_size_from_risk_pct. Retornando 0.01.")
        return 0.01

    risk_amount_val_total = balance * (risk_percentage_val / 100.0) # Renomeado

    pip_value_for_one_std_lot_in_acct_ccy = calculate_pip_value( # Renomeado
        symbol=symbol_str,
        lot_size=1.0, # Para 1 lote padrão
        account_currency=account_currency_val,
        current_exchange_rate_to_acct_currency=current_exchange_rate_quote_to_acct,
        quote_currency_inferred=quote_currency_for_pip_val
    )

    if pip_value_for_one_std_lot_in_acct_ccy <= 1e-9: # Quase zero ou negativo
        logger.error(f"Valor do pip para {symbol_str} em {account_currency_val} é inválido ({pip_value_for_one_std_lot_in_acct_ccy}). "
                       "Não é possível calcular o tamanho do lote. Verifique taxas de câmbio e moedas.")
        return 0.01 # Fallback

    calculated_lot_size_val = risk_amount_val_total / (stop_loss_pips_val * pip_value_for_one_std_lot_in_acct_ccy) # Renomeado
    return max(0.01, round(calculated_lot_size_val, 2)) # Arredondar para 2 casas e garantir mínimo


def is_forex_market_open(current_time_to_check_utc: Optional[datetime] = None) -> bool: # Renomeado
    """Verifica se o mercado Forex global está aberto no timestamp UTC fornecido."""
    cfg = get_config_helpers()
    if current_time_to_check_utc is None:
        now_to_check = datetime.now(timezone.utc) # Renomeado
    else:
        if current_time_to_check_utc.tzinfo is None:
            now_to_check = current_time_to_check_utc.replace(tzinfo=timezone.utc)
        else:
            now_to_check = current_time_to_check_utc.astimezone(timezone.utc)

    weekday_check = now_to_check.weekday() # Renomeado
    hour_check = now_to_check.hour # Renomeado

    # Buscar horários de CONFIG ou usar defaults
    # Estes devem ser os horários GLOBAIS de fechamento/abertura do mercado Forex.
    market_close_hour_friday = getattr(cfg, 'FOREX_MARKET_CLOSE_HOUR_FRIDAY_UTC', 21)
    market_open_hour_sunday = getattr(cfg, 'FOREX_MARKET_OPEN_HOUR_SUNDAY_UTC', 21)

    if weekday_check == 4:  # Sexta-feira
        return hour_check < market_close_hour_friday
    elif weekday_check == 5:  # Sábado
        return False
    elif weekday_check == 6:  # Domingo
        return hour_check >= market_open_hour_sunday
    else:  # Segunda a Quinta
        return True


def calculate_compound_returns_on_balance(initial_balance_val: float,
                             returns_list_decimal: List[float]) -> float: # Renomeado e tipo de returns
    """
    Calcula o saldo final após uma série de retornos decimais compostos.
    `returns_list_decimal` ex: [0.01, -0.005, 0.02] para +1%, -0.5%, +2%.
    """
    current_balance_val = initial_balance_val # Renomeado
    for ret_decimal in returns_list_decimal: # Renomeado
        current_balance_val *= (1.0 + ret_decimal)
    return current_balance_val


def calculate_sharpe_ratio_simple(returns_series_val: List[float], # Renomeado
                          risk_free_rate_annual_val: float = 0.02, # Renomeado
                          periods_in_year_val: int = 252) -> float: # Renomeado
    """Calcula o Sharpe Ratio (simplificado) a partir de uma série de retornos de período."""
    if not returns_series_val or len(returns_series_val) < 2: return 0.0
    returns_arr_sr = np.array(returns_series_val, dtype=float) # Renomeado

    std_dev_returns_sr = np.std(returns_arr_sr) # Renomeado
    if std_dev_returns_sr < 1e-9: # Quase sem variação
        avg_return_period_sr = np.mean(returns_arr_sr) # Renomeado
        risk_free_per_period_sr = risk_free_rate_annual_val / periods_in_year_val # Renomeado
        return 10.0 if avg_return_period_sr > risk_free_per_period_sr else 0.0

    avg_return_sr = np.mean(returns_arr_sr) # Renomeado
    annualized_avg_return_sr = avg_return_sr * periods_in_year_val # Renomeado
    annualized_std_dev_sr = std_dev_returns_sr * np.sqrt(periods_in_year_val) # Renomeado

    if annualized_std_dev_sr == 0: return 0.0
    sharpe_ratio_val = (annualized_avg_return_sr - risk_free_rate_annual_val) / annualized_std_dev_sr # Renomeado
    return sharpe_ratio_val if not np.isnan(sharpe_ratio_val) else 0.0


def calculate_max_drawdown_from_equity(equity_values_list: List[float]) -> Tuple[float, int, int]: # Renomeado
    """Calcula o drawdown máximo percentual de uma curva de equity."""
    if not equity_values_list or len(equity_values_list) < 2: return 0.0, 0, 0

    peak_equity = equity_values_list[0] # Renomeado
    peak_equity_idx = 0 # Renomeado
    max_dd_percent_val = 0.0 # Renomeado
    dd_peak_idx_val = 0 # Renomeado
    dd_trough_idx_val = 0 # Renomeado

    for i_dd, current_equity_dd in enumerate(equity_values_list): # Renomeado
        if current_equity_dd > peak_equity:
            peak_equity = current_equity_dd
            peak_equity_idx = i_dd
        
        current_dd_val_pct = 0.0 # Renomeado
        if peak_equity > 1e-9: # Evitar divisão por zero ou DDs muito grandes se equity for quase zero
            current_dd_val_pct = (peak_equity - current_equity_dd) / peak_equity
        
        if current_dd_val_pct > max_dd_percent_val:
            max_dd_percent_val = current_dd_val_pct
            dd_peak_idx_val = peak_equity_idx
            dd_trough_idx_val = i_dd
            
    return max_dd_percent_val, dd_peak_idx_val, dd_trough_idx_val


def normalize_forex_symbol(symbol_input: str) -> str: # Renomeado
    """Normaliza um símbolo de par de moedas para um formato padrão (ex: "EURUSD")."""
    if not isinstance(symbol_input, str): return ""
    normalized_s = symbol_input.strip().upper().replace("/", "").replace("-", "").replace(".", "").replace("_", "") # Renomeado
    return normalized_s


def is_major_forex_pair(symbol_to_eval: str) -> bool: # Renomeado
    """Verifica se o símbolo é um dos principais pares de Forex."""
    MAJOR_PAIRS_SET = { # Definido como constante local
        "EURUSD", "GBPUSD", "USDJPY", "USDCHF",
        "AUDUSD", "USDCAD", "NZDUSD"
    }
    return normalize_forex_symbol(symbol_to_eval) in MAJOR_PAIRS_SET


def calculate_pearson_correlation(series_a: List[float], series_b: List[float]) -> float: # Renomeado
    """Calcula o coeficiente de correlação de Pearson entre duas séries de dados."""
    if len(series_a) != len(series_b) or len(series_a) < 2: return 0.0

    s1_arr_corr = np.array(series_a, dtype=float) # Renomeado
    s2_arr_corr = np.array(series_b, dtype=float) # Renomeado
    valid_mask_corr = ~np.isnan(s1_arr_corr) & ~np.isnan(s2_arr_corr) # Renomeado
    s1_clean_corr, s2_clean_corr = s1_arr_corr[valid_mask_corr], s2_arr_corr[valid_mask_corr] # Renomeado

    if len(s1_clean_corr) < 2: return 0.0
    if np.std(s1_clean_corr) < 1e-9 or np.std(s2_clean_corr) < 1e-9: return 0.0

    corr_matrix = np.corrcoef(s1_clean_corr, s2_clean_corr)
    correlation_val = corr_matrix[0, 1] # Renomeado
    return correlation_val if not np.isnan(correlation_val) else 0.0


def time_until_next_session_starts(target_session_name_str: str, # Renomeado
                                  current_time_val_utc: Optional[datetime] = None) -> timedelta:
    """Calcula o tempo restante até o início da próxima sessão de trading especificada."""
    cfg = get_config_helpers() # Para acessar SESSION_CONFIG
    
    if current_time_val_utc is None:
        now_utc_session = datetime.now(timezone.utc) # Renomeado
    else:
        now_utc_session = current_time_val_utc.astimezone(timezone.utc) if current_time_val_utc.tzinfo else current_time_val_utc.replace(tzinfo=timezone.utc)

    # Usar SESSION_CONFIG de strategies_config.py para horários de sessão
    # Exemplo: session_details = StrategyConfig.SESSION_CONFIG.get(target_session_name_str.upper())
    # Por enquanto, usando a lógica hardcoded do original, mas com nomes de sessão consistentes
    session_start_hours_map_utc: Dict[str, int] = { # Renomeado
        'ASIA': getattr(cfg.SESSION_CONFIG.get('ASIA',{}), 'start_hour', 23),
        'LONDON': getattr(cfg.SESSION_CONFIG.get('LONDON',{}), 'start_hour', 7),
        'NEWYORK': getattr(cfg.SESSION_CONFIG.get('NEWYORK',{}), 'start_hour', 13)
    }
    
    session_name_upper = target_session_name_str.upper() # Renomeado
    if session_name_upper not in session_start_hours_map_utc:
        logger.warning(f"Nome de sessão '{target_session_name_str}' desconhecido em time_until_next_session_starts.")
        return timedelta(0)

    target_start_hour = session_start_hours_map_utc[session_name_upper] # Renomeado

    next_target_dt = now_utc_session.replace(hour=target_start_hour, minute=0, second=0, microsecond=0) # Renomeado
    if next_target_dt <= now_utc_session:
        next_target_dt += timedelta(days=1)

    # Ajustar para fins de semana
    # Se o próximo início calculado é Sábado (5) ou Domingo (6) ANTES da abertura do mercado
    market_open_sun_hour_cfg = getattr(cfg, 'FOREX_MARKET_OPEN_HOUR_SUNDAY_UTC', 21) # Renomeado

    # Loop para garantir que a data de início da sessão seja um dia de semana válido
    while not is_forex_market_open(next_target_dt) or \
          (next_target_dt.weekday() == 6 and next_target_dt.hour < market_open_sun_hour_cfg and target_start_hour < market_open_sun_hour_cfg) or \
          (next_target_dt.weekday() == 5) : # Se for Sábado, ou Domingo antes da abertura para sessões que não sejam Ásia pós-abertura
        # Se cair no Sábado, ou no Domingo antes da abertura para sessões que não sejam a de abertura (Asia)
        next_target_dt += timedelta(days=1)
        next_target_dt = next_target_dt.replace(hour=target_start_hour, minute=0, second=0, microsecond=0)
        # Casos especiais para Domingo: se a sessão alvo é Ásia e já passou da hora de abertura do mercado no Domingo, está ok.
        if next_target_dt.weekday() == 6 and session_name_upper == 'ASIA' and target_start_hour >= market_open_sun_hour_cfg:
            break # Ásia no Domingo após abertura do mercado é válida


    time_difference_val = next_target_dt - now_utc_session # Renomeado
    return time_difference_val if time_difference_val.total_seconds() > 0 else timedelta(0)


def validate_stop_loss_take_profit(entry_price_val: float, stop_loss_val: float,
                  take_profit_val: float, trade_side_str: str,
                  min_stop_dist_pips_val: float = RISK_PARAMS.MIN_STOP_DISTANCE_PIPS, # Usar de config # Renomeado
                  min_tp_dist_pips_val: float = RISK_PARAMS.MIN_STOP_DISTANCE_PIPS, # TP também precisa de distância mínima # Renomeado
                  symbol_for_pips_val: str = CONFIG.SYMBOL) -> Tuple[bool, str]: # Renomeado
    """Valida SL e TP em relação à entrada, lado, e distâncias mínimas."""
    # Garantir que os valores são numéricos
    if not all(isinstance(v, (float, int, np.number)) for v in [entry_price_val, stop_loss_val, take_profit_val]):
        return False, "Valores de preço inválidos (não numéricos)."

    pip_unit_validate = 0.01 if "JPY" in symbol_for_pips_val.upper() else 0.0001 # Renomeado
    min_stop_price_diff_val = min_stop_dist_pips_val * pip_unit_validate # Renomeado
    min_tp_price_diff_val = min_tp_dist_pips_val * pip_unit_validate # Renomeado

    side_lower = trade_side_str.lower() # Renomeado
    if side_lower == 'buy':
        if stop_loss_val >= entry_price_val: return False, "SL (Compra) deve ser < Preço de Entrada."
        if take_profit_val <= entry_price_val: return False, "TP (Compra) deve ser > Preço de Entrada."
        if abs(entry_price_val - stop_loss_val) < min_stop_price_diff_val - 1e-7: # Adicionar tolerância para float
            return False, f"SL (Compra) muito próximo. Mín: {min_stop_dist_pips_val} pips."
        if abs(take_profit_val - entry_price_val) < min_tp_price_diff_val - 1e-7:
            return False, f"TP (Compra) muito próximo. Mín: {min_tp_dist_pips_val} pips."
    elif side_lower == 'sell':
        if stop_loss_val <= entry_price_val: return False, "SL (Venda) deve ser > Preço de Entrada."
        if take_profit_val >= entry_price_val: return False, "TP (Venda) deve ser < Preço de Entrada."
        if abs(stop_loss_val - entry_price_val) < min_stop_price_diff_val - 1e-7:
            return False, f"SL (Venda) muito próximo. Mín: {min_stop_dist_pips_val} pips."
        if abs(entry_price_val - take_profit_val) < min_tp_price_diff_val - 1e-7:
            return False, f"TP (Venda) muito próximo. Mín: {min_tp_dist_pips_val} pips."
    else:
        return False, f"Lado da trade desconhecido: '{trade_side_str}'. Use 'buy' ou 'sell'."
    return True, "SL/TP Válidos."


def format_duration_to_readable_str(total_seconds_val: float) -> str:
    """Formata uma duração em segundos para uma string legível (ex: "1d 2h 30m 15s")."""
    if not isinstance(total_seconds_val, (int, float)) or total_seconds_val < 0: return "0s"
    
    total_seconds_int_fmt = int(total_seconds_val) # Renomeado

    days_fmt, rem_secs_fmt = divmod(total_seconds_int_fmt, SECONDS_IN_DAY) # Renomeado
    hours_fmt, rem_secs_fmt = divmod(rem_secs_fmt, SECONDS_IN_HOUR) # Renomeado
    minutes_fmt, secs_fmt = divmod(rem_secs_fmt, SECONDS_IN_MINUTE) # Renomeado

    parts_str_list: List[str] = [] # Renomeado
    if days_fmt > 0: parts_str_list.append(f"{days_fmt}d")
    if hours_fmt > 0: parts_str_list.append(f"{hours_fmt}h")
    if minutes_fmt > 0: parts_str_list.append(f"{minutes_fmt}m")
    if secs_fmt > 0 or not parts_str_list: # Mostrar segundos se for a única unidade ou se as outras forem zero
        parts_str_list.append(f"{secs_fmt}s")
    
    return " ".join(parts_str_list) if parts_str_list else "0s"


def calculate_trade_expectancy(win_amounts_list: List[float], # Renomeado
                               loss_amounts_positive_list: List[float]) -> float: # Renomeado
    """Calcula a Expectancy (Esperança Matemática) de uma estratégia."""
    num_total_wins = len(win_amounts_list) # Renomeado
    num_total_losses = len(loss_amounts_positive_list) # Renomeado
    total_trades_exp = num_total_wins + num_total_losses # Renomeado

    if total_trades_exp == 0: return 0.0

    win_rate_exp = num_total_wins / total_trades_exp # Renomeado
    # loss_rate_exp = num_total_losses / total_trades_exp # Não usado diretamente na fórmula comum

    avg_win_amt_exp = np.mean(win_amounts_list) if num_total_wins > 0 else 0.0 # Renomeado
    avg_loss_amt_exp = np.mean(loss_amounts_positive_list) if num_total_losses > 0 else 0.0 # Renomeado
    
    # Expectancy = (WinRate * AvgWin) - (LossRate * AvgLoss)
    expectancy_value = (win_rate_exp * avg_win_amt_exp) - ((1.0 - win_rate_exp) * avg_loss_amt_exp) # Renomeado
    return expectancy_value


def round_value_to_broker_precision(value_to_round_prec: Union[float, int, Decimal, str], # Renomeado e tipo expandido
                                   decimal_precision_val: int = 2) -> float: # Renomeado
    """Arredonda um valor para a precisão decimal especificada, usando Decimal."""
    if not isinstance(decimal_precision_val, int) or decimal_precision_val < 0:
        logger.warning(f"Precisão decimal inválida ({decimal_precision_val}). Usando 0.")
        decimal_precision_val = 0
    
    try:
        # Tentar converter para string para Decimal lidar com floats de forma precisa
        value_str_prec = str(value_to_round_prec) # Renomeado
        value_decimal_prec = Decimal(value_str_prec) # Renomeado

        if value_decimal_prec.is_nan() or value_decimal_prec.is_infinite():
            logger.warning(f"Valor NaN ou Inf ({value_to_round_prec}) não pode ser arredondado. Retornando como float.")
            return float(value_to_round_prec)


        quantizer_format_str = '1.' + '0' * decimal_precision_val if decimal_precision_val > 0 else '1' # Renomeado
        # Correção: para 2 casas, '0.01'; para 0 casas, '1'
        if decimal_precision_val > 0:
            quantizer_format_str = '0.' + '0' * (decimal_precision_val -1) + '1'
        else: # precisão 0
            quantizer_format_str = '1'


        quantizer_decimal_obj = Decimal(quantizer_format_str) # Renomeado
        rounded_decimal_val_prec = value_decimal_prec.quantize(quantizer_decimal_obj, rounding=ROUND_HALF_UP) # Renomeado
        return float(rounded_decimal_val_prec)

    except (InvalidOperation, ValueError, TypeError) as e_dec_round: # Renomeado
        logger.warning(f"Erro Decimal ao arredondar '{value_to_round_prec}' para {decimal_precision_val} casas: {e_dec_round}. Usando round() padrão.")
        try:
            return round(float(value_to_round_prec), decimal_precision_val)
        except Exception as e_round_fallback: # Renomeado
             logger.error(f"Erro no fallback de round() para '{value_to_round_prec}': {e_round_fallback}")
             return float(value_to_round_prec) if isinstance(value_to_round_prec, (float, int, np.number)) else 0.0



# Constantes úteis
SECONDS_IN_MINUTE: int = 60
SECONDS_IN_HOUR: int = SECONDS_IN_MINUTE * 60
SECONDS_IN_DAY: int = SECONDS_IN_HOUR * 24
TRADING_DAYS_PER_YEAR: int = 252 # Aproximação para mercados de ações/forex (excluindo feriados)
# PIPS_IN_POINT removido por ambiguidade.