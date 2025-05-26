# strategies/__init__.py
"""Módulo de estratégias de trading"""
import os
import importlib
import inspect
from typing import List, Type, Any # Adicionado Any
from pathlib import Path # Adicionado Path

from .base_strategy import BaseStrategy # Importação de BaseStrategy
from utils.logger import setup_logger # Importar logger

logger = setup_logger("strategy_loader") # Usar logger aqui

def load_all_strategies() -> List[Type[BaseStrategy]]:
    """
    Carrega todas as classes de estratégia que herdam de BaseStrategy
    dos subdiretórios especificados.
    """
    strategies_found: List[Type[BaseStrategy]] = [] # Renomeado e tipado
    
    # Diretório base das estratégias
    base_strategies_dir = Path(__file__).parent # Diretório atual (onde __init__.py está)

    # Subdiretórios onde as estratégias estão localizadas
    # Estes devem corresponder aos nomes das pastas em /strategies/
    strategy_module_dirs = [
        'momentum', 'mean_reversion', 'breakout', 'orderflow',
        'ml_predictive', 'arbitrage', 'news_aware', 'liquidity_hunt',
        'overnight_carry'
    ]

    for dir_name_str in strategy_module_dirs: # Renomeado dir_name
        module_dir_path = base_strategies_dir / dir_name_str # Renomeado dir_path

        if module_dir_path.is_dir(): # Checar se é um diretório
            for filepath_obj in module_dir_path.iterdir(): # Renomeado filename para filepath_obj
                # Verificar se é um arquivo Python e não um __init__.py ou outro arquivo especial
                if filepath_obj.is_file() and filepath_obj.name.endswith('.py') and \
                   not filepath_obj.name.startswith('__') and \
                   not filepath_obj.name.startswith('.'): # Ignorar arquivos ocultos

                    # Construir o nome completo do módulo para importação
                    # Ex: strategies.momentum.ema_stack
                    module_import_name = f"strategies.{dir_name_str}.{filepath_obj.stem}" # Renomeado

                    try:
                        # Importar o módulo dinamicamente
                        strategy_module = importlib.import_module(module_import_name) # Renomeado

                        # Inspecionar o módulo para encontrar classes que herdam de BaseStrategy
                        for member_name, member_obj in inspect.getmembers(strategy_module): # Renomeado name, obj
                            if inspect.isclass(member_obj) and \
                               issubclass(member_obj, BaseStrategy) and \
                               member_obj is not BaseStrategy: # Não adicionar a própria BaseStrategy
                                
                                strategies_found.append(member_obj)
                                logger.debug(f"Estratégia '{member_obj.__name__}' carregada de {module_import_name}")
                                
                    except ImportError as e_import: # Renomeado
                        logger.error(f"Erro de importação ao carregar {module_import_name}: {e_import}")
                    except Exception as e_load_mod: # Renomeado
                        logger.exception(f"Erro geral ao carregar ou inspecionar módulo {module_import_name}:") # Usar exception

    if not strategies_found:
        logger.warning("Nenhuma estratégia foi carregada. Verifique a estrutura de diretórios e as classes.")
    else:
        logger.info(f"Total de {len(strategies_found)} classes de estratégia carregadas: {[s.__name__ for s in strategies_found]}")
        
    return strategies_found

# __all__ define a interface pública do pacote 'strategies'
# Se load_all_strategies for a principal forma de obter estratégias, ela deve estar aqui.
# BaseStrategy também é fundamental.
__all__ = ['BaseStrategy', 'load_all_strategies']

# ===================================