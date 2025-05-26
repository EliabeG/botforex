# strategies/liquidity_hunt/__init__.py
"""Estratégias de liquidity hunt"""

from .stop_hunt_strategy import StopHuntStrategy
# Adicione outras estratégias de caça à liquidez aqui se existirem
# from .outra_estrategia_liquidez import OutraEstrategiaLiquidez

__all__ = [
    'StopHuntStrategy',
    # 'OutraEstrategiaLiquidez',
    ]

# ===================================