# data/__init__.py
"""Módulo de gestão de dados"""
from .tick_storage import TickStorage
from .redis_cache import RedisCache
from .market_data import MarketDataProcessor
# from .data_provider import DataProvider # Exemplo se você tivesse um DataProvider unificado
# from .models import Tick, OHLCBar # Exemplo se você tivesse modelos de dados aqui

__all__ = [
    'TickStorage',
    'RedisCache',
    'MarketDataProcessor',
    # 'DataProvider',
    # 'Tick',
    # 'OHLCBar'
    ]

# ===================================