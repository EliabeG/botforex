# config/__init__.py
"""Módulo de configuração do Trading Bot"""
from .settings import CONFIG, REGIME_CONFIG  # STRATEGY_CONFIG removido - nao definido
from .risk_config import RISK_LIMITS, RISK_PARAMS, RISK_SCORE_WEIGHTS, RISK_ADJUSTMENTS, POSITION_SIZING, RISK_MATRIX # Adicionadas todas as configs de risco

__all__ = [
    'CONFIG',
    'REGIME_CONFIG',
    'RISK_LIMITS',
    'RISK_PARAMS',
    'RISK_SCORE_WEIGHTS',
    'RISK_ADJUSTMENTS',
    'POSITION_SIZING',
    'RISK_MATRIX'
    ]

# ===================================