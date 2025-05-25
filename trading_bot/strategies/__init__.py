# strategies/__init__.py
"""Módulo de estratégias de trading"""
import os
import importlib
import inspect
from typing import List, Type
from .base_strategy import BaseStrategy

def load_all_strategies() -> List[Type[BaseStrategy]]:
    """Carrega todas as estratégias disponíveis dinamicamente"""
    strategies = []
    
    # Diretórios de estratégias
    strategy_dirs = [
        'momentum', 'mean_reversion', 'breakout', 'orderflow',
        'ml_predictive', 'arbitrage', 'news_aware', 'liquidity_hunt',
        'overnight_carry'
    ]
    
    for dir_name in strategy_dirs:
        dir_path = os.path.join(os.path.dirname(__file__), dir_name)
        
        if os.path.exists(dir_path):
            # Listar arquivos Python no diretório
            for filename in os.listdir(dir_path):
                if filename.endswith('.py') and not filename.startswith('__'):
                    module_name = f"strategies.{dir_name}.{filename[:-3]}"
                    
                    try:
                        # Importar módulo
                        module = importlib.import_module(module_name)
                        
                        # Encontrar classes que herdam de BaseStrategy
                        for name, obj in inspect.getmembers(module):
                            if (inspect.isclass(obj) and 
                                issubclass(obj, BaseStrategy) and 
                                obj != BaseStrategy):
                                strategies.append(obj)
                                
                    except Exception as e:
                        print(f"Erro ao carregar {module_name}: {e}")
    
    return strategies

__all__ = ['BaseStrategy', 'load_all_strategies']

# ===================================