# backtest/__init__.py
"""Módulo de backtesting"""
from .engine import BacktestEngine
from .analyzer import BacktestAnalyzer

__all__ = ['BacktestEngine', 'BacktestAnalyzer']