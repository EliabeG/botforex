# utils/helpers.py
"""Funções auxiliares para o trading bot"""
import math
from decimal import Decimal, ROUND_HALF_UP
from typing import Union, Optional, Tuple, List
from datetime import datetime, timedelta
import pytz
import numpy as np

def calculate_pip_value(symbol: str = "EURUSD", 
                       lot_size: float = 1.0,
                       account_currency: str = "USD") -> float:
    """
    Calcula o valor do pip para um par de moedas
    
    Args:
        symbol: Par de moedas
        lot_size: Tamanho do lote
        account_currency: Moeda da conta
    
    Returns:
        Valor do pip
    """
    # Para pares XXX/USD com conta em USD
    if symbol.endswith("USD") and account_currency == "USD":
        # 1 lote padrão = 100,000 unidades
        # 1 pip = 0.0001 para maioria dos pares (exceto JPY)
        return lot_size * 100000 * 0.0001
    
    # Para pares com JPY
    elif "JPY" in symbol:
        return lot_size * 100000 * 0.01
    
    # Para outros pares (simplificado - em produção, usar taxa de câmbio atual)
    else:
        # Assumir valor padrão
        return lot_size * 10

def format_price(price: float, symbol: str = "EURUSD") -> str:
    """
    Formata preço com precisão correta
    
    Args:
        price: Preço a formatar
        symbol: Símbolo para determinar precisão
    
    Returns:
        Preço formatado como string
    """
    if "JPY" in symbol:
        return f"{price:.3f}"
    else:
        return f"{price:.5f}"

def round_to_tick(price: float, tick_size: float = 0.00001) -> float:
    """
    Arredonda preço para o tick mais próximo
    
    Args:
        price: Preço a arredondar
        tick_size: Tamanho do tick
    
    Returns:
        Preço arredondado
    """
    return round(price / tick_size) * tick_size

def calculate_position_size(account_balance: float,
                          risk_percent: float,
                          stop_loss_pips: float,
                          pip_value: float = 10) -> float:
    """
    Calcula tamanho da posição baseado em risco
    
    Args:
        account_balance: Saldo da conta
        risk_percent: Percentual de risco (0.01 = 1%)
        stop_loss_pips: Distância do stop em pips
        pip_value: Valor do pip por lote
    
    Returns:
        Tamanho da posição em lotes
    """
    if stop_loss_pips <= 0:
        return 0.01  # Retornar mínimo
    
    risk_amount = account_balance * risk_percent
    position_size = risk_amount / (stop_loss_pips * pip_value)
    
    # Arredondar para 2 casas decimais (precisão do broker)
    return round(position_size, 2)

def calculate_risk_reward_ratio(entry: float, 
                               stop_loss: float,
                               take_profit: float) -> float:
    """
    Calcula relação risco/recompensa
    
    Args:
        entry: Preço de entrada
        stop_loss: Preço do stop loss
        take_profit: Preço do take profit
    
    Returns:
        Relação R:R
    """
    risk = abs(entry - stop_loss)
    reward = abs(take_profit - entry)
    
    if risk == 0:
        return 0
    
    return reward / risk

def pips_to_price(pips: float, symbol: str = "EURUSD") -> float:
    """
    Converte pips para diferença de preço
    
    Args:
        pips: Número de pips
        symbol: Símbolo do par
    
    Returns:
        Diferença de preço
    """
    if "JPY" in symbol:
        return pips * 0.01
    else:
        return pips * 0.0001

def price_to_pips(price_diff: float, symbol: str = "EURUSD") -> float:
    """
    Converte diferença de preço para pips
    
    Args:
        price_diff: Diferença de preço
        symbol: Símbolo do par
    
    Returns:
        Número de pips
    """
    if "JPY" in symbol:
        return price_diff / 0.01
    else:
        return price_diff / 0.0001

def calculate_spread_cost(spread_pips: float, 
                         lot_size: float,
                         pip_value: float = 10) -> float:
    """
    Calcula custo do spread
    
    Args:
        spread_pips: Spread em pips
        lot_size: Tamanho da posição
        pip_value: Valor do pip
    
    Returns:
        Custo do spread em dinheiro
    """
    return spread_pips * lot_size * pip_value

def get_trading_session(timestamp: datetime = None) -> str:
    """
    Determina sessão de trading atual
    
    Args:
        timestamp: Timestamp para verificar (default: agora)
    
    Returns:
        Nome da sessão (Asia, London, NewYork, Overlap)
    """
    if timestamp is None:
        timestamp = datetime.now(pytz.UTC)
    
    # Converter para UTC se necessário
    if timestamp.tzinfo is None:
        timestamp = pytz.UTC.localize(timestamp)
    
    hour = timestamp.hour
    
    # Definir sessões em UTC
    sessions = {
        'Asia': [(23, 8)],  # 23:00 - 08:00 UTC
        'London': [(7, 16)],  # 07:00 - 16:00 UTC
        'NewYork': [(13, 22)],  # 13:00 - 22:00 UTC
    }
    
    active_sessions = []
    
    for session, times in sessions.items():
        for start, end in times:
            if start > end:  # Sessão atravessa meia-noite
                if hour >= start or hour < end:
                    active_sessions.append(session)
            else:
                if start <= hour < end:
                    active_sessions.append(session)
    
    # Verificar overlap
    if 'London' in active_sessions and 'NewYork' in active_sessions:
        return 'Overlap'
    elif active_sessions:
        return active_sessions[0]
    else:
        return 'Closed'

def calculate_lot_from_risk(balance: float,
                           risk_pct: float,
                           stop_pips: float,
                           symbol: str = "EURUSD") -> float:
    """
    Calcula tamanho do lote baseado em risco percentual
    
    Args:
        balance: Saldo da conta
        risk_pct: Risco percentual (1 = 1%)
        stop_pips: Stop loss em pips
        symbol: Símbolo do par
    
    Returns:
        Tamanho do lote
    """
    risk_amount = balance * (risk_pct / 100)
    pip_value = calculate_pip_value(symbol, 1.0)
    
    if stop_pips > 0:
        lot_size = risk_amount / (stop_pips * pip_value)
        return round(lot_size, 2)
    
    return 0.01

def is_market_open() -> bool:
    """
    Verifica se o mercado forex está aberto
    
    Returns:
        True se mercado aberto, False caso contrário
    """
    now = datetime.now(pytz.UTC)
    weekday = now.weekday()
    
    # Forex fecha sexta 21:00 UTC e abre domingo 21:00 UTC
    if weekday == 4:  # Sexta
        return now.hour < 21
    elif weekday == 5:  # Sábado
        return False
    elif weekday == 6:  # Domingo
        return now.hour >= 21
    else:  # Segunda a Quinta
        return True

def calculate_compound_returns(initial_balance: float,
                             returns: List[float]) -> float:
    """
    Calcula retorno composto
    
    Args:
        initial_balance: Saldo inicial
        returns: Lista de retornos (0.01 = 1%)
    
    Returns:
        Saldo final
    """
    balance = initial_balance
    
    for ret in returns:
        balance *= (1 + ret)
    
    return balance

def calculate_sharpe_ratio(returns: List[float],
                          risk_free_rate: float = 0.02,
                          periods_per_year: int = 252) -> float:
    """
    Calcula Sharpe Ratio
    
    Args:
        returns: Lista de retornos
        risk_free_rate: Taxa livre de risco anual
        periods_per_year: Períodos por ano (252 para daily)
    
    Returns:
        Sharpe Ratio
    """
    if not returns or len(returns) < 2:
        return 0
    
    returns_array = np.array(returns)
    
    # Retorno médio
    avg_return = np.mean(returns_array)
    
    # Desvio padrão
    std_return = np.std(returns_array)
    
    if std_return == 0:
        return 0
    
    # Anualizar
    annual_return = avg_return * periods_per_year
    annual_std = std_return * np.sqrt(periods_per_year)
    
    # Sharpe Ratio
    sharpe = (annual_return - risk_free_rate) / annual_std
    
    return sharpe

def calculate_max_drawdown(equity_curve: List[float]) -> Tuple[float, int, int]:
    """
    Calcula drawdown máximo
    
    Args:
        equity_curve: Lista de valores de equity
    
    Returns:
        Tupla (max_drawdown, peak_idx, trough_idx)
    """
    if not equity_curve:
        return 0, 0, 0
    
    peak = equity_curve[0]
    peak_idx = 0
    max_dd = 0
    max_dd_peak_idx = 0
    max_dd_trough_idx = 0
    
    for i, value in enumerate(equity_curve):
        if value > peak:
            peak = value
            peak_idx = i
        
        dd = (peak - value) / peak if peak > 0 else 0
        
        if dd > max_dd:
            max_dd = dd
            max_dd_peak_idx = peak_idx
            max_dd_trough_idx = i
    
    return max_dd, max_dd_peak_idx, max_dd_trough_idx

def normalize_symbol(symbol: str) -> str:
    """
    Normaliza símbolo para formato padrão
    
    Args:
        symbol: Símbolo a normalizar
    
    Returns:
        Símbolo normalizado
    """
    # Remover espaços e converter para maiúsculas
    symbol = symbol.strip().upper()
    
    # Remover caracteres especiais comuns
    symbol = symbol.replace("/", "")
    symbol = symbol.replace("-", "")
    symbol = symbol.replace(".", "")
    
    return symbol

def is_major_pair(symbol: str) -> bool:
    """
    Verifica se é um par major
    
    Args:
        symbol: Símbolo do par
    
    Returns:
        True se for major
    """
    major_pairs = [
        "EURUSD", "GBPUSD", "USDJPY", "USDCHF",
        "AUDUSD", "USDCAD", "NZDUSD"
    ]
    
    return normalize_symbol(symbol) in major_pairs

def calculate_correlation(series1: List[float], 
                         series2: List[float]) -> float:
    """
    Calcula correlação entre duas séries
    
    Args:
        series1: Primeira série de dados
        series2: Segunda série de dados
    
    Returns:
        Coeficiente de correlação (-1 a 1)
    """
    if len(series1) != len(series2) or len(series1) < 2:
        return 0
    
    # Converter para numpy arrays
    s1 = np.array(series1)
    s2 = np.array(series2)
    
    # Calcular correlação
    correlation_matrix = np.corrcoef(s1, s2)
    correlation = correlation_matrix[0, 1]
    
    return correlation

def time_until_session(session: str) -> timedelta:
    """
    Calcula tempo até próxima sessão
    
    Args:
        session: Nome da sessão (Asia, London, NewYork)
    
    Returns:
        Timedelta até início da sessão
    """
    now = datetime.now(pytz.UTC)
    
    session_times = {
        'Asia': 23,     # 23:00 UTC
        'London': 7,    # 07:00 UTC
        'NewYork': 13   # 13:00 UTC
    }
    
    if session not in session_times:
        return timedelta(0)
    
    target_hour = session_times[session]
    
    # Calcular próxima ocorrência
    target = now.replace(hour=target_hour, minute=0, second=0, microsecond=0)
    
    if target <= now:
        # Sessão já passou hoje, calcular para amanhã
        target += timedelta(days=1)
    
    # Considerar fim de semana
    if target.weekday() == 5:  # Sábado
        target += timedelta(days=2)  # Pular para segunda
    elif target.weekday() == 6:  # Domingo
        if target.hour < 21:  # Antes da abertura
            target = target.replace(hour=21)
        else:
            target += timedelta(days=1)  # Segunda
    
    return target - now

def validate_stops(entry: float, 
                  stop_loss: float,
                  take_profit: float,
                  side: str,
                  min_distance: float = 0.0005) -> Tuple[bool, str]:
    """
    Valida stop loss e take profit
    
    Args:
        entry: Preço de entrada
        stop_loss: Stop loss
        take_profit: Take profit
        side: 'buy' ou 'sell'
        min_distance: Distância mínima em preço
    
    Returns:
        Tupla (válido, mensagem_erro)
    """
    if side == 'buy':
        # Para compra: SL < Entry < TP
        if stop_loss >= entry:
            return False, "Stop loss deve ser menor que entrada para compra"
        
        if take_profit <= entry:
            return False, "Take profit deve ser maior que entrada para compra"
        
        if entry - stop_loss < min_distance:
            return False, f"Stop loss muito próximo (mínimo {min_distance})"
        
        if take_profit - entry < min_distance:
            return False, f"Take profit muito próximo (mínimo {min_distance})"
    
    else:  # sell
        # Para venda: TP < Entry < SL
        if stop_loss <= entry:
            return False, "Stop loss deve ser maior que entrada para venda"
        
        if take_profit >= entry:
            return False, "Take profit deve ser menor que entrada para venda"
        
        if stop_loss - entry < min_distance:
            return False, f"Stop loss muito próximo (mínimo {min_distance})"
        
        if entry - take_profit < min_distance:
            return False, f"Take profit muito próximo (mínimo {min_distance})"
    
    return True, "OK"

def format_duration(seconds: float) -> str:
    """
    Formata duração em formato legível
    
    Args:
        seconds: Duração em segundos
    
    Returns:
        String formatada (ex: "2h 30m 15s")
    """
    if seconds < 60:
        return f"{int(seconds)}s"
    
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if secs > 0 or not parts:
        parts.append(f"{secs}s")
    
    return " ".join(parts)

def calculate_expectancy(wins: List[float], losses: List[float]) -> float:
    """
    Calcula expectancy (esperança matemática)
    
    Args:
        wins: Lista de valores ganhos
        losses: Lista de valores perdidos (positivos)
    
    Returns:
        Expectancy
    """
    if not wins and not losses:
        return 0
    
    total_trades = len(wins) + len(losses)
    
    if total_trades == 0:
        return 0
    
    win_rate = len(wins) / total_trades
    loss_rate = len(losses) / total_trades
    
    avg_win = np.mean(wins) if wins else 0
    avg_loss = np.mean(losses) if losses else 0
    
    expectancy = (win_rate * avg_win) - (loss_rate * avg_loss)
    
    return expectancy

def round_to_broker_precision(value: float, precision: int = 2) -> float:
    """
    Arredonda para precisão do broker
    
    Args:
        value: Valor a arredondar
        precision: Número de casas decimais
    
    Returns:
        Valor arredondado
    """
    return float(Decimal(str(value)).quantize(
        Decimal(f'0.{"0" * precision}'),
        rounding=ROUND_HALF_UP
    ))

# Constantes úteis
SECONDS_IN_DAY = 86400
SECONDS_IN_HOUR = 3600
SECONDS_IN_MINUTE = 60
TRADING_DAYS_YEAR = 252
PIPS_IN_POINT = 10