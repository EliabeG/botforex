# strategies/momentum/__init__.py
"""Estratégias de momentum e trend-following"""
from .ema_stack import EMAStackStrategy

__all__ = ['EMAStackStrategy']

# ===================================

# strategies/mean_reversion/__init__.py
"""Estratégias de mean reversion"""
from .zscore_vwap import ZScoreVWAPStrategy

__all__ = ['ZScoreVWAPStrategy']

# ===================================