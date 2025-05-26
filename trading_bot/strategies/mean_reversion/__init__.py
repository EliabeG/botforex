# strategies/mean_reversion/__init__.py
"""Estratégias de mean reversion"""
from .zscore_vwap import ZScoreVWAPStrategy
from .bollinger_fade import BollingerFadeStrategy # Adicionada BollingerFadeStrategy

__all__ = [
    'ZScoreVWAPStrategy',
    'BollingerFadeStrategy', # Adicionada
    ]

# ===================================