# backtest/__init__.py
"""Módulo de backtesting"""
from .engine import BacktestEngine, BacktestResults, BacktestTrade # Adicionado BacktestResults, BacktestTrade
from .analyzer import BacktestAnalyzer

__all__ = [
    'BacktestEngine',
    'BacktestResults', # Adicionado
    'BacktestTrade',   # Adicionado
    'BacktestAnalyzer'
    ]