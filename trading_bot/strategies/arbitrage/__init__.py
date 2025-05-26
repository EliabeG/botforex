# strategies/arbitrage/__init__.py
"""Estratégias de arbitragem estatística"""

# Se você tiver classes de estratégia definidas em outros arquivos dentro desta pasta
# (ex: statistical_arbitrage.py), você deve importá-las e adicioná-las a __all__.
from .statistical_arbitrage import StatisticalArbitrageStrategy

__all__ = [
    'StatisticalArbitrageStrategy'
    ]

# ===================================