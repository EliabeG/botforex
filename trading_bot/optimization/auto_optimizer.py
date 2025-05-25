# optimization/auto_optimizer.py
import optuna
import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Callable
from datetime import datetime, timedelta
import joblib
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

from config.settings import CONFIG
from strategies.base_strategy import BaseStrategy
from backtest.engine import BacktestEngine
from utils.logger import setup_logger

logger = setup_logger("auto_optimizer")

class StrategyOptimizer:
    """Otimizador automático de estratégias usando Optuna (Bayesian Optimization)"""
    
    def __init__(self, strategy_class: type, historical_data: pd.DataFrame):
        self.strategy_class = strategy_class
        self.historical_data = historical_data
        self.study = None
        self.best_params = None
        self.optimization_history = []
        
        # Configurações de otimização
        self.n_trials = 100
        self.n_jobs = mp.cpu_count() - 1
        self.timeout = 3600  # 1 hora máximo
        
        # Métricas alvo
        self.target_metric = 'expectancy_adjusted_sharpe'
        
    def create_objective(self, train_data: pd.DataFrame, 
                        validation_data: pd.DataFrame) -> Callable:
        """Cria função objetivo para otimização"""
        
        def objective(trial: optuna.Trial) -> float:
            try:
                # Sugerir parâmetros
                params = self._suggest_parameters(trial)
                
                # Criar instância da estratégia
                strategy = self.strategy_class()
                strategy.update_parameters(params)
                
                # Executar backtest no período de treino
                backtest = BacktestEngine()
                train_results = backtest.run(
                    strategy=strategy,
                    data=train_data,
                    initial_balance=10000,
                    commission=0.00002  # 0.2 pips
                )
                
                # Verificar se teve trades suficientes
                if train_results['total_trades'] < 10:
                    return -1000  # Penalizar estratégias sem trades
                
                # Validar no período de teste
                validation_results = backtest.run(
                    strategy=strategy,
                    data=validation_data,
                    initial_balance=10000,
                    commission=0.00002
                )
                
                # Calcular métrica objetivo
                score = self._calculate_objective_score(
                    train_results,
                    validation_results
                )
                
                # Registrar trial
                trial.set_user_attr('train_trades', train_results['total_trades'])
                trial.set_user_attr('validation_trades', validation_results['total_trades'])
                trial.set_user_attr('train_sharpe', train_results.get('sharpe_ratio', 0))
                trial.set_user_attr('validation_sharpe', validation_results.get('sharpe_ratio', 0))
                
                return score
                
            except Exception as e:
                logger.error(f"Erro no trial {trial.number}: {e}")
                return -1000
        
        return objective
    
    def _suggest_parameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Sugere parâmetros baseado na estratégia"""
        strategy_instance = self.strategy_class()
        default_params = strategy_instance.get_default_parameters()
    def _suggest_parameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Sugere parâmetros baseado na estratégia"""
        strategy_instance = self.strategy_class()
        default_params = strategy_instance.get_default_parameters()
        suggested_params = {}
        
        # Mapear tipos de parâmetros e sugerir ranges
        for param_name, default_value in default_params.items():
            if isinstance(default_value, bool):
                suggested_params[param_name] = trial.suggest_categorical(
                    param_name, [True, False]
                )
            
            elif isinstance(default_value, int):
                # Definir ranges baseado no nome do parâmetro
                if 'period' in param_name or 'window' in param_name:
                    suggested_params[param_name] = trial.suggest_int(
                        param_name, 
                        max(5, int(default_value * 0.5)), 
                        min(200, int(default_value * 2))
                    )
                elif 'minutes' in param_name:
                    suggested_params[param_name] = trial.suggest_int(
                        param_name, 5, 120, step=5
                    )
                else:
                    suggested_params[param_name] = trial.suggest_int(
                        param_name,
                        max(1, int(default_value * 0.5)),
                        int(default_value * 2)
                    )
            
            elif isinstance(default_value, float):
                # Ranges para diferentes tipos de parâmetros float
                if 'multiplier' in param_name:
                    suggested_params[param_name] = trial.suggest_float(
                        param_name, 0.5, 5.0, step=0.1
                    )
                elif 'threshold' in param_name:
                    suggested_params[param_name] = trial.suggest_float(
                        param_name,
                        default_value * 0.5,
                        default_value * 1.5,
                        step=0.1
                    )
                elif 'ratio' in param_name:
                    suggested_params[param_name] = trial.suggest_float(
                        param_name, 0.1, 0.9, step=0.1
                    )
                elif param_name in ['min_confidence', 'min_volume_ratio']:
                    suggested_params[param_name] = trial.suggest_float(
                        param_name, 0.3, 0.9, step=0.1
                    )
                else:
                    suggested_params[param_name] = trial.suggest_float(
                        param_name,
                        default_value * 0.5,
                        default_value * 2.0
                    )
            
            else:
                # Manter valor padrão para tipos não suportados
                suggested_params[param_name] = default_value
        
        return suggested_params
    
    def _calculate_objective_score(self, train_results: Dict, 
                                 validation_results: Dict) -> float:
        """Calcula score objetivo combinando múltiplas métricas"""
        # Penalizar se não teve trades na validação
        if validation_results['total_trades'] < 5:
            return -1000
        
        # Métricas principais
        val_sharpe = validation_results.get('sharpe_ratio', 0)
        val_expectancy = validation_results.get('expectancy', 0)
        val_win_rate = validation_results.get('win_rate', 0)
        val_max_dd = validation_results.get('max_drawdown', 1)
        
        # Verificar overfitting
        train_sharpe = train_results.get('sharpe_ratio', 0)
        overfit_penalty = 0
        
        if train_sharpe > 0 and val_sharpe > 0:
            # Penalizar se validação muito pior que treino
            sharpe_ratio_decay = val_sharpe / train_sharpe
            if sharpe_ratio_decay < 0.5:
                overfit_penalty = -0.5
        
        # Score composto
        if val_expectancy > 0 and val_sharpe > 0:
            # Expectancy-adjusted Sharpe
            score = val_sharpe * np.sqrt(abs(val_expectancy)) * (1 - val_max_dd)
            
            # Bônus por consistência
            if val_win_rate > 0.5:
                score *= 1.1
            
            # Aplicar penalidade de overfitting
            score += overfit_penalty
        else:
            score = -1000
        
        return score
    
    async def optimize(self, n_trials: int = None, timeout: int = None):
        """Executa otimização Bayesiana"""
        n_trials = n_trials or self.n_trials
        timeout = timeout or self.timeout
        
        logger.info(f"Iniciando otimização de {self.strategy_class.__name__}")
        logger.info(f"Trials: {n_trials} | Timeout: {timeout}s | Jobs: {self.n_jobs}")
        
        # Preparar dados para walk-forward
        train_data, validation_data = self._prepare_walk_forward_data()
        
        # Criar estudo Optuna
        self.study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=10)
        )
        
        # Função objetivo
        objective = self.create_objective(train_data, validation_data)
        
        # Executar otimização
        try:
            self.study.optimize(
                objective,
                n_trials=n_trials,
                timeout=timeout,
                n_jobs=self.n_jobs,
                show_progress_bar=True
            )
            
            # Salvar melhores parâmetros
            self.best_params = self.study.best_params
            best_value = self.study.best_value
            
            logger.info(f"Otimização concluída!")
            logger.info(f"Melhor score: {best_value:.4f}")
            logger.info(f"Melhores parâmetros: {self.best_params}")
            
            # Analisar resultados
            self._analyze_optimization_results()
            
        except Exception as e:
            logger.error(f"Erro durante otimização: {e}")
            raise
    
    def _prepare_walk_forward_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Prepara dados para walk-forward optimization"""
        total_days = len(self.historical_data.index.unique())
        
        # Configurações walk-forward
        train_days = int(CONFIG.WALK_FORWARD_TRAIN_MONTHS * 30)
        test_days = int(CONFIG.WALK_FORWARD_TEST_MONTHS * 30)
        
        # Usar dados mais recentes
        cutoff_date = self.historical_data.index.max() - timedelta(days=test_days)
        
        train_data = self.historical_data[
            self.historical_data.index < cutoff_date
        ].tail(train_days * 24 * 60 * 60)  # Assumindo dados de tick
        
        validation_data = self.historical_data[
            self.historical_data.index >= cutoff_date
        ]
        
        logger.info(f"Train: {len(train_data)} ticks | "
                   f"Validation: {len(validation_data)} ticks")
        
        return train_data, validation_data
    
    def _analyze_optimization_results(self):
        """Analisa resultados da otimização"""
        # Importância dos parâmetros
        importance = optuna.importance.get_param_importances(self.study)
        
        logger.info("Importância dos parâmetros:")
        for param, imp in importance.items():
            logger.info(f"  {param}: {imp:.3f}")
        
        # Estatísticas dos trials
        df_trials = self.study.trials_dataframe()
        
        successful_trials = df_trials[df_trials['value'] > -1000]
        if len(successful_trials) > 0:
            logger.info(f"\nEstatísticas dos trials bem-sucedidos:")
            logger.info(f"  Total: {len(successful_trials)}")
            logger.info(f"  Score médio: {successful_trials['value'].mean():.4f}")
            logger.info(f"  Score máximo: {successful_trials['value'].max():.4f}")
            logger.info(f"  Desvio padrão: {successful_trials['value'].std():.4f}")
        
        # Visualizações (salvar em arquivo)
        self._save_optimization_plots()
    
    def _save_optimization_plots(self):
        """Salva visualizações da otimização"""
        try:
            import matplotlib.pyplot as plt
            
            # História da otimização
            fig = optuna.visualization.plot_optimization_history(self.study)
            fig.write_html(f"optimization_{self.strategy_class.__name__}_history.html")
            
            # Importância dos parâmetros
            fig = optuna.visualization.plot_param_importances(self.study)
            fig.write_html(f"optimization_{self.strategy_class.__name__}_importance.html")
            
            # Coordenadas paralelas
            fig = optuna.visualization.plot_parallel_coordinate(self.study)
            fig.write_html(f"optimization_{self.strategy_class.__name__}_parallel.html")
            
            logger.info("Gráficos de otimização salvos")
            
        except Exception as e:
            logger.warning(f"Não foi possível salvar gráficos: {e}")
    
    async def validate_parameters(self, params: Dict, test_data: pd.DataFrame) -> Dict:
        """Valida parâmetros em dados de teste"""
        strategy = self.strategy_class()
        strategy.update_parameters(params)
        
        backtest = BacktestEngine()
        results = backtest.run(
            strategy=strategy,
            data=test_data,
            initial_balance=10000,
            commission=0.00002
        )
        
        return results
    
    def save_results(self, filepath: str):
        """Salva resultados da otimização"""
        results = {
            'strategy': self.strategy_class.__name__,
            'best_params': self.best_params,
            'best_score': self.study.best_value if self.study else None,
            'n_trials': len(self.study.trials) if self.study else 0,
            'optimization_date': datetime.now().isoformat(),
            'study': self.study
        }
        
        joblib.dump(results, filepath)
        logger.info(f"Resultados salvos em {filepath}")
    
    def load_results(self, filepath: str):
        """Carrega resultados anteriores"""
        results = joblib.load(filepath)
        self.best_params = results['best_params']
        self.study = results['study']
        logger.info(f"Resultados carregados de {filepath}")


class WalkForwardOptimizer:
    """Otimizador com walk-forward analysis completo"""
    
    def __init__(self, strategy_class: type):
        self.strategy_class = strategy_class
        self.optimization_windows = []
        self.out_of_sample_results = []
        
    async def run_walk_forward(self, historical_data: pd.DataFrame,
                              train_months: int = 6,
                              test_months: int = 1,
                              step_months: int = 1):
        """Executa walk-forward optimization completo"""
        logger.info(f"Iniciando walk-forward para {self.strategy_class.__name__}")
        
        start_date = historical_data.index.min()
        end_date = historical_data.index.max()
        
        current_date = start_date + timedelta(days=train_months * 30)
        
        window_count = 0
        
        while current_date + timedelta(days=test_months * 30) <= end_date:
            window_count += 1
            logger.info(f"\nJanela {window_count}:")
            
            # Definir períodos
            train_start = current_date - timedelta(days=train_months * 30)
            train_end = current_date
            test_start = current_date
            test_end = current_date + timedelta(days=test_months * 30)
            
            # Separar dados
            train_data = historical_data[
                (historical_data.index >= train_start) & 
                (historical_data.index < train_end)
            ]
            
            test_data = historical_data[
                (historical_data.index >= test_start) & 
                (historical_data.index < test_end)
            ]
            
            logger.info(f"  Treino: {train_start.date()} a {train_end.date()}")
            logger.info(f"  Teste: {test_start.date()} a {test_end.date()}")
            
            # Otimizar
            optimizer = StrategyOptimizer(self.strategy_class, train_data)
            await optimizer.optimize(n_trials=50, timeout=1800)  # 30 min por janela
            
            # Validar out-of-sample
            if optimizer.best_params:
                oos_results = await optimizer.validate_parameters(
                    optimizer.best_params,
                    test_data
                )
                
                # Armazenar resultados
                self.optimization_windows.append({
                    'window': window_count,
                    'train_period': (train_start, train_end),
                    'test_period': (test_start, test_end),
                    'best_params': optimizer.best_params,
                    'in_sample_score': optimizer.study.best_value,
                    'out_of_sample_results': oos_results
                })
                
                self.out_of_sample_results.append(oos_results)
                
                logger.info(f"  OOS Sharpe: {oos_results.get('sharpe_ratio', 0):.3f}")
                logger.info(f"  OOS Trades: {oos_results.get('total_trades', 0)}")
            
            # Avançar janela
            current_date += timedelta(days=step_months * 30)
        
        # Analisar resultados agregados
        self._analyze_walk_forward_results()
    
    def _analyze_walk_forward_results(self):
        """Analisa resultados do walk-forward"""
        if not self.out_of_sample_results:
            logger.warning("Sem resultados para analisar")
            return
        
        # Agregar métricas OOS
        total_trades = sum(r.get('total_trades', 0) for r in self.out_of_sample_results)
        avg_sharpe = np.mean([r.get('sharpe_ratio', 0) for r in self.out_of_sample_results])
        avg_win_rate = np.mean([r.get('win_rate', 0) for r in self.out_of_sample_results])
        
        logger.info("\n=== Resultados Walk-Forward ===")
        logger.info(f"Janelas analisadas: {len(self.optimization_windows)}")
        logger.info(f"Total de trades OOS: {total_trades}")
        logger.info(f"Sharpe médio OOS: {avg_sharpe:.3f}")
        logger.info(f"Win rate médio OOS: {avg_win_rate:.3%}")
        
        # Verificar consistência dos parâmetros
        self._check_parameter_stability()
    
    def _check_parameter_stability(self):
        """Verifica estabilidade dos parâmetros otimizados"""
        if len(self.optimization_windows) < 2:
            return
        
        # Coletar todos os parâmetros
        all_params = defaultdict(list)
        
        for window in self.optimization_windows:
            for param, value in window['best_params'].items():
                all_params[param].append(value)
        
        logger.info("\n=== Estabilidade dos Parâmetros ===")
        
        for param, values in all_params.items():
            if isinstance(values[0], (int, float)):
                mean_val = np.mean(values)
                std_val = np.std(values)
                cv = std_val / mean_val if mean_val != 0 else 0
                
                logger.info(f"{param}:")
                logger.info(f"  Média: {mean_val:.3f}")
                logger.info(f"  Desvio: {std_val:.3f}")
                logger.info(f"  CV: {cv:.3%}")
                
                if cv > 0.3:
                    logger.warning(f"  ⚠️ Alta variabilidade!")


# Função auxiliar para executar otimização scheduled
async def scheduled_optimization(strategy_classes: List[type], data_manager):
    """Executa otimização agendada de todas as estratégias"""
    logger.info("Iniciando otimização scheduled")
    
    for strategy_class in strategy_classes:
        try:
            # Obter dados históricos
            historical_data = await data_manager.get_historical_ticks(
                CONFIG.SYMBOL,
                days=CONFIG.WALK_FORWARD_TRAIN_MONTHS * 30 + CONFIG.WALK_FORWARD_TEST_MONTHS * 30
            )
            
            if len(historical_data) < 1000:
                logger.warning(f"Dados insuficientes para otimizar {strategy_class.__name__}")
                continue
            
            # Executar otimização
            optimizer = StrategyOptimizer(strategy_class, historical_data)
            await optimizer.optimize()
            
            # Salvar parâmetros otimizados
            if optimizer.best_params:
                await data_manager.save_strategy_params(
                    strategy_class.__name__,
                    optimizer.best_params
                )
                
                # Salvar estudo completo
                optimizer.save_results(
                    f"models/optimization_{strategy_class.__name__}_{datetime.now().strftime('%Y%m%d')}.pkl"
                )
            
        except Exception as e:
            logger.error(f"Erro ao otimizar {strategy_class.__name__}: {e}")
            continue
    
    logger.info("Otimização scheduled concluída")