# strategies/news_aware/__init__.py
"""Estratégias sensíveis a notícias e eventos econômicos."""

from .news_fade_strategy import NewsFadeStrategy
# Adicione outras estratégias sensíveis a notícias aqui
# from .news_breakout_strategy import NewsBreakoutStrategy

__all__ = [
    'NewsFadeStrategy',
    # 'NewsBreakoutStrategy',
    ]

# ===================================