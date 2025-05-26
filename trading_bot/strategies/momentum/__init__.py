# strategies/momentum/__init__.py
"""Estratégias de momentum e trend-following"""

from .ema_stack import EMAStackStrategy
from .cci_adx import CCIADXStrategy
from .donchian_breakout import DonchianBreakoutStrategy # Assumindo que esta é a versão de momentum
from .ichimoku_kumo import IchimokuKumoStrategy
# Adicione outras estratégias de momentum aqui
# from .outra_estrategia_momentum import OutraEstrategiaMomentum

__all__ = [
    'EMAStackStrategy',
    'CCIADXStrategy',
    'DonchianBreakoutStrategy',
    'IchimokuKumoStrategy',
    # 'OutraEstrategiaMomentum',
    ]

# ===================================