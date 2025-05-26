# config/risk_config.py
"""Configurações de gestão de risco"""
from dataclasses import dataclass, field # Adicionado field para default_factory
from typing import Dict, List, Tuple, Any # Adicionado Any

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
    CIRCUIT_BREAKER_PAUSE_HOURS: int = 24 # Era 24, mantido
    CIRCUIT_BREAKER_LOSS_THRESHOLD: float = 0.10  # 10% em X trades (definido no circuit_breaker.py)

    # Margem e alavancagem
    MAX_LEVERAGE: int = 500 # Este valor geralmente vem da conta, mas pode ser um limite superior
    MARGIN_CALL_LEVEL: float = 0.50         # 50% de margem
    MARGIN_STOP_OUT: float = 0.20           # 20% stop out
    MIN_FREE_MARGIN_PERCENT: float = 0.30   # 30% margem livre mínima (renomeado para clareza)

@dataclass
class RiskParameters:
    """Parâmetros de cálculo de risco"""
    # Volatilidade
    VOLATILITY_LOOKBACK: int = 20           # Períodos para calcular volatilidade
    VOLATILITY_MULTIPLIER: float = 1.5      # Multiplicador de volatilidade
    HIGH_VOLATILITY_THRESHOLD: float = 0.02 # 2% é considerado alta volatilidade (ex: ATR/Preço)

    # Correlação
    CORRELATION_LOOKBACK: int = 100         # Períodos para correlação
    HIGH_CORRELATION_THRESHOLD: float = 0.7 # Correlação alta

    # Slippage e custos
    EXPECTED_SLIPPAGE_PIPS: float = 0.1       # 0.1 pip (ajustado do original 0.0001 que seria 1 pip para 5 casas decimais)
    MAX_ALLOWED_SLIPPAGE_PIPS: float = 0.3    # 0.3 pips
    COMMISSION_PER_LOT: float = 7.0         # $7 por lote (ida e volta)

    # Stop loss
    MIN_STOP_DISTANCE_PIPS: float = 5.0       # 5 pips mínimo (ajustado do original)
    MAX_STOP_DISTANCE_PIPS: float = 50.0      # 50 pips máximo
    TRAILING_STOP_ACTIVATION_PIPS: float = 10.0 # Ativar após 10 pips de lucro (ajustado)
    TRAILING_STOP_DISTANCE_PIPS: float = 8.0   # 8 pips de distância

    # Take profit
    MIN_RR_RATIO: float = 1.5               # Risk/reward mínimo
    DEFAULT_RR_RATIO: float = 2.0           # Risk/reward padrão
    PARTIAL_TP_LEVELS: List[Tuple[float, float]] = field(default_factory=lambda: [
        (0.5, 0.5),  # 50% do volume na metade do caminho para o TP original (baseado em R)
        (0.8, 0.3),  # 30% do volume restante em 80% do caminho para o TP
        (1.0, 1.0)   # 100% do volume restante no TP final (ou o que sobrou)
        # O segundo valor da tupla deve ser a % DO VOLUME ATUAL a ser fechada.
        # Ex: (R_level, percent_to_close_of_remaining_position)
        # (0.5 R, 0.5) -> fecha 50% da posição quando atinge 0.5R de lucro
        # (1.0 R, 0.5) -> fecha mais 50% (do que sobrou) quando atinge 1.0R
        # (1.5 R, 1.0) -> fecha o restante (100%) quando atinge 1.5R
    ])


@dataclass
class RiskScoreWeights:
    """Pesos para cálculo de score de risco"""
    MARKET_VOLATILITY: float = 0.25
    CORRELATION_RISK: float = 0.20
    DRAWDOWN_LEVEL: float = 0.20
    WIN_RATE: float = 0.15 # Win rate do sistema/estratégia
    EXPOSURE_LEVEL: float = 0.10
    TIME_OF_DAY: float = 0.10 # Risco associado ao horário

@dataclass
class RiskAdjustments:
    """Ajustes dinâmicos de risco baseados em condições"""

    PERFORMANCE_ADJUSTMENTS: Dict[str, float] = field(default_factory=lambda: {
        'winning_streak_3': 1.1,    # Aumenta 10% risco após 3 vitórias
        'winning_streak_5': 1.2,    # Aumenta 20% risco após 5 vitórias
        'losing_streak_2': 0.75,    # Reduz 25% risco após 2 perdas
        'losing_streak_3': 0.5,     # Reduz 50% risco após 3 perdas
        'daily_profit_target_50pct_reached': 0.8,  # Reduz 20% risco se já lucrou 50% da meta diária
        'daily_profit_target_80pct_reached': 0.5,  # Reduz 50% risco se já lucrou 80% da meta diária
    })

    VOLATILITY_ADJUSTMENTS: Dict[str, float] = field(default_factory=lambda: { # Chaves podem ser 'low', 'normal', 'high', 'extreme'
        'low': 1.2,      # Ex: Volatilidade < 0.5% (definido por HIGH_VOLATILITY_THRESHOLD etc.)
        'normal': 1.0,
        'high': 0.7,
        'extreme': 0.3
    })

    SESSION_ADJUSTMENTS: Dict[str, float] = field(default_factory=lambda: {
        'asia': 0.8,
        'london_open': 0.9,
        'london': 1.0,
        'ny_open': 0.9,
        'newyork': 1.0,
        'overlap': 1.1, # Overlap Londres/NY
        'market_close_period': 0.7 # Próximo ao fechamento semanal/diário
    })

    WEEKDAY_ADJUSTMENTS: Dict[int, float] = field(default_factory=lambda: { # 0=Segunda, 6=Domingo
        0: 0.9,  # Segunda
        1: 1.0,  # Terça
        2: 1.0,  # Quarta
        3: 1.0,  # Quinta
        4: 0.8,  # Sexta
        5: 0.5,  # Sábado (mercado geralmente fechado, mas para referência)
        6: 0.7   # Domingo (abertura)
    })

    EVENT_ADJUSTMENTS: Dict[str, float] = field(default_factory=lambda: {
        'high_impact_news_imminent_30min': 0.3, # Risco reduzido 30min antes de notícia alto impacto
        'medium_impact_news_imminent_15min': 0.7,
        'central_bank_decision_day': 0.5,
        'nfp_day_around_release': 0.4, # Durante o NFP
        'major_holiday_thin_liquidity': 0.5,
        'market_month_end': 0.8,
        'market_quarter_end': 0.7
    })

@dataclass
class PositionSizingMethods:
    """Métodos de dimensionamento de posição"""

    FIXED_LOT: str = "fixed_lot"
    FIXED_RISK: str = "fixed_risk"
    KELLY_CRITERION: str = "kelly"
    VOLATILITY_BASED: str = "volatility"
    EQUITY_CURVE: str = "equity_curve"
    # MARTINGALE: str = "martingale"        # Não recomendado!
    # ANTI_MARTINGALE: str = "anti_martingale"

    METHOD_CONFIG: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "fixed_lot": { # Renomeado para corresponder
            'lot_size': 0.01,
            'max_lots': 1.0 # Limite superior para este método
        },
        "fixed_risk": { # Renomeado
            'risk_percent': 0.01, # 1% do saldo da conta
            'round_to': 0.01 # Arredondar para 0.01 lotes
        },
        "kelly": { # Renomeado
            'kelly_fraction': 0.25,  # Fração conservadora do Kelly (ex: half-Kelly, quarter-Kelly)
            'max_kelly_risk_percent': 0.02, # Risco máximo em % do saldo, mesmo que Kelly sugira mais
            'lookback_trades': 100, # Trades para calcular win rate e payoff
            'min_trades_for_kelly': 20 # Mínimo de trades para usar Kelly
        },
        "volatility": { # Renomeado
            'base_risk_percent': 0.01, # Risco base em % do saldo
            'vol_adjustment_enabled': True, # Se deve ajustar pelo multiplicador de volatilidade
            'vol_lookback_periods': 20 # Períodos para ATR ou std dev de retornos
        },
        "equity_curve": { # Renomeado
            'base_risk_percent': 0.01,
            'increase_factor_on_winning': 1.1, # Multiplicador se acima da MA da equity
            'decrease_factor_on_losing': 0.9,  # Multiplicador se abaixo da MA da equity
            'equity_ma_period': 20 # Período da MA da curva de equity
        }
    })

# Risk matrix para decisões rápidas
# As chaves aqui são tuplas de strings, o que é bom.
# Os valores para 'volatility' (low, normal, high, extreme) e 'drawdown' (low, medium, high)
# precisariam ser definidos quantitativamente em algum lugar para que a matriz seja utilizável.
RISK_MATRIX: Dict[Tuple[str, str, str], float] = field(default_factory=lambda: {
    # (volatility_level, drawdown_level, correlation_level) -> risk_multiplier
    ('low', 'low', 'low'): 1.2,
    ('low', 'low', 'high'): 0.9,
    ('low', 'medium', 'low'): 0.8,
    ('low', 'medium', 'high'): 0.6,
    ('low', 'high', 'any'): 0.3, # 'any' para correlação aqui

    ('normal', 'low', 'low'): 1.0,
    ('normal', 'low', 'high'): 0.8,
    ('normal', 'medium', 'low'): 0.7,
    ('normal', 'medium', 'high'): 0.5,
    ('normal', 'high', 'any'): 0.2,

    ('high', 'low', 'low'): 0.7,
    ('high', 'low', 'high'): 0.5,
    ('high', 'medium', 'any'): 0.3,
    ('high', 'high', 'any'): 0.1, # Reduzido de 0.0 para permitir uma chance mínima se outras condições forem perfeitas

    ('extreme', 'any', 'any'): 0.0  # Não operar
})

# Instâncias globais (mantidas como no original, mas agora são instâncias de dataclasses)
RISK_LIMITS = RiskLimits()
RISK_PARAMS = RiskParameters()
RISK_SCORE_WEIGHTS = RiskScoreWeights()
RISK_ADJUSTMENTS = RiskAdjustments()
POSITION_SIZING = PositionSizingMethods()
# RISK_MATRIX já é uma instância de dicionário, não precisa ser instanciada de uma dataclass.
# Se você quisesse que RISK_MATRIX fosse uma dataclass, a estrutura seria diferente.
# Para o uso atual como um dicionário de consulta, está correto.