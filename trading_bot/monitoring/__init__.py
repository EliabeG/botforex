# monitoring/__init__.py
"""Módulo de monitoramento e alertas"""
from .metrics_collector import MetricsCollector
from .alerts import AlertManager, AlertSeverity, AlertChannel, Alert # Exportar Enums e Alert class também
from .dashboard import DashboardConfig

__all__ = [
    'MetricsCollector',
    'AlertManager',
    'AlertSeverity', # Adicionado
    'AlertChannel',  # Adicionado
    'Alert',         # Adicionado
    'DashboardConfig'
    ]

# ===================================