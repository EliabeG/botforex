# config/risk_config.py
"""Configurações de gestão de risco"""
from dataclasses import dataclass
from typing import Dict, List, Tuple

@dataclass
class RiskLimits:
    """Limites de risco do sistema"""
    # Limites por trade
    MAX_RISK_PER_TRADE: float = 0.01        # 1% máximo por trade
    MIN_RISK_PER_TRADE: float = 0.0025      # 0.25% mínimo
    DEFAULT_RISK_PER_TRADE: float = 0.005   # 0.5% padrão
    
    # Limites diários
    DAILY_LOSS_LIMIT: float = 0.03          # 3% perda máxima diária
    DAILY_PROFIT_TARGET: float = 0.05       # 5% meta diária
    DAILY_TRADE_LIMIT: int = 20             # Máximo de trades por dia
    
    # Limites de drawdown
    MAX_DRAWDOWN: float = 0.20              # 20% drawdown máximo
    WARNING_DRAWDOWN: float = 0.10          # 10% drawdown de aviso
    INTRADAY_DRAWDOWN: float = 0.05         # 5% drawdown intraday
    
    # Limites de exposição
    MAX_EXPOSURE: float = 0.50              # 50% exposição máxima
    MAX_CORRELATED_EXPOSURE: float = 0.30   # 30% em trades correlacionados
    MAX_POSITIONS: int = 5                  # Máximo de posições simultâneas
    MAX_POSITIONS_PER_STRATEGY: int = 2     # Máximo por estratégia
    
    # Circuit breaker
    CIRCUIT_BREAKER_CONSECUTIVE_LOSSES: int = 3
    CIRCUIT_BREAKER_PAUSE_HOURS: int = 24
    CIRCUIT_BREAKER_LOSS_THRESHOLD: float = 0.10  # 10% em 5 trades
    
    # Margem e alavancagem
    MAX_LEVERAGE: int = 500
    MARGIN_CALL_LEVEL: float = 0.50         # 50% de margem
    MARGIN_STOP_OUT: float = 0.20           # 20% stop out
    MIN_FREE_MARGIN: float = 0.30           # 30% margem livre mínima

@dataclass
class RiskParameters:
    """Parâmetros de cálculo de risco"""
    # Volatilidade
    VOLATILITY_LOOKBACK: int = 20           # Períodos para calcular volatilidade
    VOLATILITY_MULTIPLIER: float = 1.5      # Multiplicador de volatilidade
    HIGH_VOLATILITY_THRESHOLD: float = 0.02 # 2% é considerado alta volatilidade
    
    # Correlação
    CORRELATION_LOOKBACK: int = 100         # Períodos para correlação
    HIGH_CORRELATION_THRESHOLD: float = 0.7 # Correlação alta
    
    # Slippage e custos
    EXPECTED_SLIPPAGE: float = 0.0001       # 1 pip
    MAX_ALLOWED_SLIPPAGE: float = 0.0003    # 3 pips
    COMMISSION_PER_LOT: float = 7.0         # $7 por lote (ida e volta)
    
    # Stop loss
    MIN_STOP_DISTANCE: float = 0.0005       # 5 pips mínimo
    MAX_STOP_DISTANCE: float = 0.0050       # 50 pips máximo
    TRAILING_STOP_ACTIVATION: float = 0.0010 # Ativar após 10 pips de lucro
    TRAILING_STOP_DISTANCE: float = 0.0008   # 8 pips de distância
    
    # Take profit
    MIN_RR_RATIO: float = 1.5               # Risk/reward mínimo
    DEFAULT_RR_RATIO: float = 2.0           # Risk/reward padrão
    PARTIAL_TP_LEVELS: List[Tuple[float, float]] = [
        (0.5, 0.5),  # 50% na metade do TP
        (0.8, 0.3),  # 30% em 80% do TP
        (1.0, 0.2)   # 20% no TP final
    ]

class RiskScoreWeights:
    """Pesos para cálculo de score de risco"""
    MARKET_VOLATILITY: float = 0.25
    CORRELATION_RISK: float = 0.20
    DRAWDOWN_LEVEL: float = 0.20
    WIN_RATE: float = 0.15
    EXPOSURE_LEVEL: float = 0.10
    TIME_OF_DAY: float = 0.10

class RiskAdjustments:
    """Ajustes dinâmicos de risco baseados em condições"""
    
    # Ajustes por performance
    PERFORMANCE_ADJUSTMENTS = {
        'winning_streak_3': 1.1,    # Aumenta 10% após 3 vitórias
        'winning_streak_5': 1.2,    # Aumenta 20% após 5 vitórias
        'losing_streak_2': 0.75,    # Reduz 25% após 2 perdas
        'losing_streak_3': 0.5,     # Reduz 50% após 3 perdas
        'daily_profit_50pct': 0.8,  # Reduz 20% se já lucrou 50% da meta
        'daily_profit_80pct': 0.5,  # Reduz 50% se já lucrou 80% da meta
    }
    
    # Ajustes por volatilidade
    VOLATILITY_ADJUSTMENTS = {
        'low': 1.2,      # Volatilidade < 0.5%
        'normal': 1.0,   # Volatilidade 0.5-1.5%
        'high': 0.7,     # Volatilidade 1.5-2.5%
        'extreme': 0.3   # Volatilidade > 2.5%
    }
    
    # Ajustes por sessão
    SESSION_ADJUSTMENTS = {
        'asia': 0.8,           # Sessão asiática
        'london_open': 0.9,    # Abertura de Londres
        'london': 1.0,         # Sessão de Londres
        'ny_open': 0.9,        # Abertura de NY
        'newyork': 1.0,        # Sessão de NY
        'overlap': 1.1,        # Overlap Londres/NY
        'close': 0.7           # Fechamento
    }
    
    # Ajustes por dia da semana
    WEEKDAY_ADJUSTMENTS = {
        0: 0.9,  # Segunda - cautela após weekend
        1: 1.0,  # Terça
        2: 1.0,  # Quarta
        3: 1.0,  # Quinta
        4: 0.8   # Sexta - reduzir antes do weekend
    }
    
    # Ajustes por eventos
    EVENT_ADJUSTMENTS = {
        'high_impact_news_30min': 0.3,
        'medium_impact_news_15min': 0.7,
        'central_bank_day': 0.5,
        'nfp_day': 0.6,
        'holiday_thin_liquidity': 0.5,
        'month_end': 0.8,
        'quarter_end': 0.7
    }

class PositionSizingMethods:
    """Métodos de dimensionamento de posição"""
    
    FIXED_LOT = "fixed_lot"
    FIXED_RISK = "fixed_risk"
    KELLY_CRITERION = "kelly"
    VOLATILITY_BASED = "volatility"
    EQUITY_CURVE = "equity_curve"
    MARTINGALE = "martingale"        # Não recomendado!
    ANTI_MARTINGALE = "anti_martingale"
    
    # Configurações por método
    METHOD_CONFIG = {
        FIXED_LOT: {
            'lot_size': 0.01,
            'max_lots': 1.0
        },
        FIXED_RISK: {
            'risk_percent': 0.01,
            'round_to': 0.01
        },
        KELLY_CRITERION: {
            'kelly_fraction': 0.25,  # Kelly conservador
            'max_kelly': 0.02,       # Máximo 2%
            'lookback_trades': 100
        },
        VOLATILITY_BASED: {
            'base_risk': 0.01,
            'vol_adjustment': True,
            'vol_lookback': 20
        },
        EQUITY_CURVE: {
            'base_risk': 0.01,
            'increase_on_winning': 1.1,
            'decrease_on_losing': 0.9,
            'equity_ma_period': 20
        }
    }

# Risk matrix para decisões rápidas
RISK_MATRIX = {
    # (volatility, drawdown, correlation) -> risk_multiplier
    ('low', 'low', 'low'): 1.2,
    ('low', 'low', 'high'): 0.9,
    ('low', 'medium', 'low'): 0.8,
    ('low', 'medium', 'high'): 0.6,
    ('low', 'high', 'any'): 0.3,
    
    ('normal', 'low', 'low'): 1.0,
    ('normal', 'low', 'high'): 0.8,
    ('normal', 'medium', 'low'): 0.7,
    ('normal', 'medium', 'high'): 0.5,
    ('normal', 'high', 'any'): 0.2,
    
    ('high', 'low', 'low'): 0.7,
    ('high', 'low', 'high'): 0.5,
    ('high', 'medium', 'any'): 0.3,
    ('high', 'high', 'any'): 0.0,  # Não operar
    
    ('extreme', 'any', 'any'): 0.0  # Não operar
}

# Instâncias globais
RISK_LIMITS = RiskLimits()
RISK_PARAMS = RiskParameters()
RISK_SCORE_WEIGHTS = RiskScoreWeights()
RISK_ADJUSTMENTS = RiskAdjustments()
POSITION_SIZING = PositionSizingMethods()