# optimization/__init__.py
"""Módulo de otimização automática"""
from .auto_optimizer import StrategyOptimizer, WalkForwardOptimizer, scheduled_optimization
from .scoring import StrategyScorer, PerformanceMetrics # Adicionado PerformanceMetrics
from .walk_forward import WalkForwardAnalysis, WalkForwardWindow # Adicionado WalkForwardWindow

__all__ = [
    'StrategyOptimizer',
    'WalkForwardOptimizer',
    'scheduled_optimization',
    'StrategyScorer',
    'PerformanceMetrics', # Adicionado
    'WalkForwardAnalysis',
    'WalkForwardWindow'   # Adicionado
]

# ===================================