# core/__init__.py
"""Módulos principais do Trading Bot"""
from .orchestrator import TradingOrchestrator
from .market_regime import MarketRegimeDetector, MarketRegime
from .data_manager import DataManager
from .execution_engine import ExecutionEngine, Order, OrderStatus

__all__ = [
    'TradingOrchestrator',
    'MarketRegimeDetector',
    'MarketRegime',
    'DataManager',
    'ExecutionEngine',
    'Order',
    'OrderStatus'
]

# ===================================