# strategies/ml_predictive/__init__.py
"""Estratégias preditivas baseadas em Machine Learning"""

from .gradient_boost_strategy import GradientBoostStrategy
# Adicione outras estratégias de ML aqui, se houver.
# Exemplo: from .lstm_predictive_strategy import LSTMPredictiveStrategy

__all__ = [
    'GradientBoostStrategy',
    # 'LSTMPredictiveStrategy',
]

# ===================================