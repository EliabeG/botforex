# optimization/auto_optimizer.py
import optuna # Optuna já está nos requirements
import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Callable, Type # Adicionado Type
from datetime import datetime, timedelta, timezone # Adicionado timezone
import joblib
from concurrent.futures import ProcessPoolExecutor # Mantido para referência, mas Optuna lida com n_jobs
import multiprocessing as mp
from collections import defaultdict # Adicionado
from pathlib import Path # Adicionado Path

from config.settings import CONFIG
from strategies.base_strategy import BaseStrategy
from backtest.engine import BacktestEngine, BacktestResults # Adicionado BacktestResults
from utils.logger import setup_logger

logger = setup_logger("auto_optimizer")

class StrategyOptimizer:
    """Otimizador automático de estratégias usando Optuna (Bayesian Optimization)"""

    def __init__(self, strategy_class: Type[BaseStrategy], historical_data: pd.DataFrame): # Usar Type[BaseStrategy]
        self.strategy_class = strategy_class
        self.historical_data = historical_data.copy() # Trabalhar com uma cópia
        self.study: Optional[optuna.Study] = None # Adicionada tipagem
        self.best_params: Optional[Dict[str, Any]] = None # Adicionada tipagem
        self.optimization_history: List[Dict[str, Any]] = [] # Adicionada tipagem

        # Configurações de otimização
        self.n_trials: int = getattr(CONFIG, 'OPTIMIZATION_N_TRIALS', 100)
        # n_jobs=-1 em Optuna usa todos os cores. Se mp.cpu_count() for 1, -1 pode dar erro.
        # Usar max(1, mp.cpu_count() -1) se quiser deixar um core livre.
        # Ou deixar Optuna gerenciar com n_jobs=-1.
        self.n_jobs: int = max(1, (mp.cpu_count() or 1) - 1) if (mp.cpu_count() or 1) > 1 else 1
        self.timeout_seconds: int = getattr(CONFIG, 'OPTIMIZATION_TIMEOUT_SECONDS', 3600) # Renomeado

        # Métricas alvo (pode ser configurável)
        self.target_metric: str = 'expectancy_adjusted_sharpe' # Mantido, mas a função de score deve retornar isso
        self.optimization_results_path = Path(CONFIG.MODELS_PATH) / "optimizations" # Novo caminho
        self.optimization_results_path.mkdir(parents=True, exist_ok=True)


    def create_objective_function(self, train_data: pd.DataFrame, # Renomeado de create_objective
                        validation_data: pd.DataFrame) -> Callable[[optuna.Trial], float]: # Adicionada tipagem
        """Cria função objetivo para otimização com Optuna."""

        def objective(trial: optuna.Trial) -> float:
            try:
                params = self._suggest_parameters_for_trial(trial) # Renomeado
                strategy = self.strategy_class()
                strategy.update_parameters(params)

                backtest_engine = BacktestEngine() # Criar nova instância para cada trial

                # Backtest no período de treino
                train_results_obj: BacktestResults = backtest_engine.run( # Tipagem
                    strategy=strategy,
                    data=train_data,
                    initial_balance=getattr(CONFIG, 'OPTIMIZATION_INITIAL_BALANCE', 10000),
                    commission=getattr(CONFIG, 'OPTIMIZATION_COMMISSION', 0.00002)
                )
                # Limpar estado do backtest_engine para o próximo run
                backtest_engine._reset()


                if train_results_obj.total_trades < getattr(CONFIG, 'OPTIMIZATION_MIN_TRAIN_TRADES', 10):
                    logger.debug(f"Trial {trial.number}: Poucos trades no treino ({train_results_obj.total_trades}). Penalizando.")
                    return -1000.0 # Penalizar se não houver trades suficientes

                # Backtest no período de validação
                validation_results_obj: BacktestResults = backtest_engine.run( # Tipagem
                    strategy=strategy,
                    data=validation_data,
                    initial_balance=getattr(CONFIG, 'OPTIMIZATION_INITIAL_BALANCE', 10000),
                    commission=getattr(CONFIG, 'OPTIMIZATION_COMMISSION', 0.00002)
                )
                backtest_engine._reset()

                if validation_results_obj.total_trades < getattr(CONFIG, 'OPTIMIZATION_MIN_VALIDATION_TRADES', 5):
                    logger.debug(f"Trial {trial.number}: Poucos trades na validação ({validation_results_obj.total_trades}). Penalizando.")
                    return -1000.0


                score = self._calculate_objective_score(
                    train_results_obj.to_dict(), # Passar como dict
                    validation_results_obj.to_dict() # Passar como dict
                )

                # Armazenar métricas chave no trial para análise posterior
                trial.set_user_attr('train_total_trades', train_results_obj.total_trades)
                trial.set_user_attr('train_sharpe_ratio', train_results_obj.sharpe_ratio)
                trial.set_user_attr('train_net_pnl', train_results_obj.net_pnl)
                trial.set_user_attr('validation_total_trades', validation_results_obj.total_trades)
                trial.set_user_attr('validation_sharpe_ratio', validation_results_obj.sharpe_ratio)
                trial.set_user_attr('validation_net_pnl', validation_results_obj.net_pnl)
                trial.set_user_attr('validation_max_drawdown', validation_results_obj.max_drawdown)

                # Guardar parâmetros testados neste trial para histórico
                self.optimization_history.append({
                    'trial_number': trial.number,
                    'params': params,
                    'score': score,
                    'train_trades': train_results_obj.total_trades,
                    'validation_trades': validation_results_obj.total_trades
                })


                return score

            except optuna.exceptions.TrialPruned: # Capturar se o pruner cortar o trial
                raise # Relançar para Optuna lidar
            except Exception as e_obj: # Renomeado
                logger.error(f"Erro no trial de otimização {trial.number} para {self.strategy_class.__name__}: {e_obj}", exc_info=False) # exc_info=False para não poluir com tracebacks de trials ruins
                logger.debug(f"Detalhes do erro no trial {trial.number}:", exc_info=True) # Logar traceback no debug
                return -1000.0 # Retornar score baixo em caso de erro

        return objective

    def _suggest_parameters_for_trial(self, trial: optuna.Trial) -> Dict[str, Any]: # Renomeado
        """Sugere parâmetros para um trial, baseado nos defaults da estratégia."""
        strategy_instance = self.strategy_class() # Obter instância para defaults
        default_params = strategy_instance.get_default_parameters()
        suggested_params: Dict[str, Any] = {} # Adicionada tipagem

        for param_name, default_value in default_params.items():
            # Lógica de sugestão de ranges (pode ser mais granular e específica por estratégia)
            param_config = getattr(CONFIG, 'OPTIMIZATION_PARAM_RANGES', {}).get(self.strategy_class.__name__, {}).get(param_name)

            if param_config: # Se houver configuração específica de range
                param_type = param_config.get('type', 'float' if isinstance(default_value, float) else 'int')
                if param_type == 'categorical':
                    suggested_params[param_name] = trial.suggest_categorical(param_name, param_config['choices'])
                elif param_type == 'int':
                    suggested_params[param_name] = trial.suggest_int(param_name, param_config['low'], param_config['high'], step=param_config.get('step', 1))
                elif param_type == 'float':
                    suggested_params[param_name] = trial.suggest_float(param_name, param_config['low'], param_config['high'], step=param_config.get('step'), log=param_config.get('log', False))
            else: # Lógica de fallback baseada no tipo e nome (como no original)
                if isinstance(default_value, bool):
                    suggested_params[param_name] = trial.suggest_categorical(param_name, [True, False])
                elif isinstance(default_value, int):
                    low = max(1, int(default_value * 0.5))
                    high = int(default_value * 2.0) + (5 if default_value < 5 else 0) # Garantir um range
                    if low >= high: high = low + 1 # Garantir high > low
                    step = 1
                    if 'period' in param_name.lower() or 'window' in param_name.lower():
                        low = max(5, int(default_value * 0.3))
                        high = min(250, int(default_value * 2.5) + 10)
                        if low >= high: high = low + 5
                        step = 1 if high - low < 20 else 5
                    suggested_params[param_name] = trial.suggest_int(param_name, low, high, step=step)
                elif isinstance(default_value, float):
                    # Ajustar ranges para floats para serem mais significativos
                    low_f = default_value * 0.5
                    high_f = default_value * 2.0
                    if abs(default_value) < 1e-3: # Para valores pequenos como percentuais
                        low_f = default_value - abs(default_value * 0.5) if default_value !=0 else -0.001
                        high_f = default_value + abs(default_value * 0.5) if default_value !=0 else 0.001
                    elif abs(default_value) < 1.0: # ex: 0.5
                        low_f = max(0.1, default_value * 0.5)
                        high_f = min(5.0, default_value * 2.0)

                    if low_f >= high_f : high_f = low_f + abs(low_f * 0.1) + 1e-3 # Garantir range

                    suggested_params[param_name] = trial.suggest_float(param_name, low_f, high_f) # Remover step para float por default
                else:
                    suggested_params[param_name] = default_value # Manter valor padrão

        return suggested_params


    def _calculate_objective_score(self, train_results: Dict[str, Any],
                                 validation_results: Dict[str, Any]) -> float:
        """Calcula score objetivo combinando múltiplas métricas de treino e validação."""
        # Penalizar se não teve trades suficientes na validação
        if validation_results.get('total_trades', 0) < getattr(CONFIG, 'OPTIMIZATION_MIN_VALIDATION_TRADES_FOR_SCORE', 3):
            return -1000.0

        val_sharpe = validation_results.get('sharpe_ratio', -5.0) # Default baixo se ausente
        val_expectancy = validation_results.get('expectancy', -1.0) # Pips ou $? Assumindo $ por trade
        val_win_rate = validation_results.get('win_rate', 0.0)
        val_max_dd = validation_results.get('max_drawdown', 1.0) # 0-1 range, 1 é 100% DD
        val_pnl = validation_results.get('net_pnl', 0.0)

        # Se PnL de validação for negativo, penalizar fortemente
        if val_pnl <= 0:
            return val_pnl - 100 # Retornar PnL negativo para Optuna minimizar (ou um valor grande negativo)

        # Verificar overfitting comparando Sharpe de treino e validação
        train_sharpe = train_results.get('sharpe_ratio', -5.0)
        overfit_penalty_factor = 1.0
        if train_sharpe > 0.1 and val_sharpe < (train_sharpe * 0.5): # Se val_sharpe é menos da metade do train_sharpe
            overfit_penalty_factor = 0.5 # Reduzir score pela metade
            logger.debug(f"Penalidade de Overfitting aplicada: Train Sharpe {train_sharpe:.2f}, Val Sharpe {val_sharpe:.2f}")
        elif train_sharpe > 0.1 and val_sharpe < 0: # Sharpe de treino positivo, validação negativo
            overfit_penalty_factor = 0.1 # Penalidade severa
            logger.debug(f"Penalidade Severa de Overfitting: Train Sharpe {train_sharpe:.2f}, Val Sharpe {val_sharpe:.2f}")


        # Score composto: focar em Sharpe de validação, PnL e estabilidade (baixo DD)
        # (PnL / MaxDD) * sqrt(Trades) é uma variação do SQN ou similar
        # Um score mais robusto pode ser:
        # score = (val_sharpe * 0.4) + ( (val_pnl / (val_max_dd * CONFIG.OPTIMIZATION_INITIAL_BALANCE + 1e-6)) * 0.3 ) + (val_win_rate * 0.1) + (min(1, validation_results.get('total_trades',0)/50.0) * 0.2)
        # O score original: score = val_sharpe * np.sqrt(abs(val_expectancy)) * (1 - val_max_dd)

        # Score focado em PnL de validação ajustado pelo drawdown e Sharpe
        # Garantir que val_max_dd não seja zero para evitar divisão por zero
        # Se o PnL for usado, o target_metric deve refletir isso.
        # Se o objetivo é 'expectancy_adjusted_sharpe':
        if val_expectancy > 0: # Se a expectância for positiva
            score = val_sharpe * np.sqrt(val_expectancy) * (1.0 - val_max_dd) * overfit_penalty_factor
        else: # Se a expectância for negativa, o score deve ser ruim
            score = val_sharpe * (1.0 - val_max_dd) * overfit_penalty_factor # Sem o sqrt da expectância negativa

        # Bônus por número de trades (até um certo ponto)
        # score *= min(1.0, np.sqrt(validation_results.get('total_trades', 0) / 20.0)) # Ex: normalizar por 20 trades

        return score if np.isfinite(score) else -1000.0 # Garantir que não é NaN/Inf


    async def optimize(self, n_trials_opt: Optional[int] = None, timeout_opt_seconds: Optional[int] = None): # Renomeado
        """Executa otimização Bayesiana (TPE) com Optuna."""
        n_trials_to_run = n_trials_opt if n_trials_opt is not None else self.n_trials # Renomeado
        timeout_to_use = timeout_opt_seconds if timeout_opt_seconds is not None else self.timeout_seconds # Renomeado

        logger.info(f"Iniciando otimização de {self.strategy_class.__name__}")
        logger.info(f"Trials: {n_trials_to_run} | Timeout: {timeout_to_use}s | Jobs: {self.n_jobs}")

        # Preparar dados de treino e validação (ex: 70% treino, 30% validação mais recente)
        # Esta é uma forma simples de split. Walk-forward é mais robusto.
        if self.historical_data.empty:
            logger.error("Dados históricos vazios. Otimização não pode prosseguir.")
            return

        split_ratio = 0.7
        split_index = int(len(self.historical_data) * split_ratio)
        train_df = self.historical_data.iloc[:split_index]
        validation_df = self.historical_data.iloc[split_index:]

        if train_df.empty or validation_df.empty:
            logger.error("Dados de treino ou validação vazios após split. Verifique o tamanho dos dados históricos.")
            return

        logger.info(f"Dados de Treino: {len(train_df)} ticks | Dados de Validação: {len(validation_df)} ticks")

        # Criar estudo Optuna
        # Sampler TPESampler (Tree-structured Parzen Estimator) é bom para otimização Bayesiana.
        # Pruner MedianPruner corta trials não promissores cedo.
        self.study = optuna.create_study(
            study_name=f"{self.strategy_class.__name__}_opt_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M')}",
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=CONFIG.GLOBAL_SEED if hasattr(CONFIG, 'GLOBAL_SEED') else 42, multivariate=True, group=True), # Adicionado multivariate e group
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=max(5, n_trials_to_run // 10), n_min_trials=max(3, n_trials_to_run // 20)) # Ajustado pruner
        )

        objective_func = self.create_objective_function(train_df, validation_df) # Renomeado

        try:
            # Optuna.optimize pode ser bloqueante. Para rodar em asyncio,
            # ou usamos run_in_executor, ou Optuna precisa de um loop de eventos integrado.
            # A documentação do Optuna sugere que para paralelização com `n_jobs`,
            # a função objetivo deve ser picklable, o que geralmente é o caso.
            # Se a função objetivo em si for `async`, a integração é mais complexa.
            # Assumindo que `objective_func` é síncrona e o `BacktestEngine.run` é síncrono.
            loop = asyncio.get_event_loop()
            await loop.run_in_executor( # Executar a otimização bloqueante em um thread pool
                None, # Default executor (ThreadPoolExecutor)
                lambda: self.study.optimize( # type: ignore
                    objective_func,
                    n_trials=n_trials_to_run,
                    timeout=timeout_to_use,
                    n_jobs=self.n_jobs, # n_jobs > 1 usará multiprocessing
                    show_progress_bar=True # Pode não funcionar bem com n_jobs ou em logs não-TTY
                )
            )


            if self.study.best_trial: # Checar se best_trial existe
                self.best_params = self.study.best_trial.params
                best_value_found = self.study.best_trial.value # Renomeado
                logger.info("Otimização concluída!")
                logger.info(f"Melhor score objetivo: {best_value_found:.4f}")
                logger.info(f"Melhores parâmetros encontrados: {self.best_params}")
                self._analyze_optimization_results()
                self.save_study_results() # Renomeado
            else:
                logger.warning("Otimização concluída, mas nenhum trial bem-sucedido para determinar melhores parâmetros.")
                self.best_params = None


        except Exception as e_opt: # Renomeado
            logger.exception("Erro crítico durante o processo de otimização:") # Usar logger.exception
            # Não relançar para permitir que o bot continue com params default se a otimização falhar.

    # _prepare_walk_forward_data foi removido pois a otimização agora usa um split simples.
    # A classe WalkForwardOptimizer lidará com a lógica walk-forward.

    def _analyze_optimization_results(self):
        """Analisa e loga resultados da otimização de forma mais detalhada."""
        if not self.study:
            logger.warning("Estudo de otimização não encontrado para análise.")
            return

        logger.info("\n--- Análise Detalhada da Otimização ---")
        logger.info(f"Número total de trials: {len(self.study.trials)}")

        # Importância dos parâmetros
        try:
            if len(self.study.trials) > 1: # Precisa de mais de um trial para calcular importância
                param_importances = optuna.importance.get_param_importances(self.study) # Renomeado
                logger.info("Importância dos Parâmetros (do mais para o menos importante):")
                for param_name, importance_value in sorted(param_importances.items(), key=lambda item: item[1], reverse=True): # Renomeado
                    logger.info(f"  - {param_name}: {importance_value:.4f}")
            else:
                logger.info("Importância dos parâmetros não pode ser calculada (trials insuficientes).")
        except Exception as e_imp: # Renomeado
            logger.error(f"Erro ao calcular importância dos parâmetros: {e_imp}")


        # Estatísticas dos trials
        df_trials = self.study.trials_dataframe(
            attrs=('number', 'value', 'datetime_start', 'datetime_complete', 'duration', 'state', 'params', 'user_attrs')
        )
        # Filtrar trials que não foram penalizados (score > -1000)
        successful_trials_df = df_trials[df_trials['value'] > -999.0] # Renomeado

        if not successful_trials_df.empty:
            logger.info(f"\nEstatísticas dos Trials Bem-Sucedidos ({len(successful_trials_df)} de {len(df_trials)}):")
            logger.info(f"  Score Médio: {successful_trials_df['value'].mean():.4f}")
            logger.info(f"  Score Mediano: {successful_trials_df['value'].median():.4f}")
            logger.info(f"  Melhor Score: {successful_trials_df['value'].max():.4f}")
            logger.info(f"  Pior Score (não penalizado): {successful_trials_df['value'].min():.4f}")
            logger.info(f"  Desvio Padrão do Score: {successful_trials_df['value'].std():.4f}")
            logger.info(f"  Duração Média do Trial: {successful_trials_df['duration'].mean()}")

            # User attributes (ex: trades, sharpe)
            for attr in ['train_sharpe_ratio', 'validation_sharpe_ratio', 'train_total_trades', 'validation_total_trades', 'validation_net_pnl']:
                if f'user_attrs_{attr}' in successful_trials_df.columns:
                    logger.info(f"  Média de '{attr}': {successful_trials_df[f'user_attrs_{attr}'].astype(float).mean():.2f}")
        else:
            logger.info("Nenhum trial bem-sucedido (não penalizado) para analisar estatísticas.")


        # Salvar visualizações se matplotlib estiver disponível e configurado
        # self._save_optimization_plots() # Chamada movida para save_study_results


    def _save_optimization_plots(self, study_name_for_file: str): # Adicionado arg
        """Salva visualizações da otimização em arquivos HTML."""
        if not self.study:
            logger.warning("Estudo Optuna não disponível, não é possível salvar plots.")
            return

        plots_to_generate = {
            "history": optuna.visualization.plot_optimization_history,
            "param_importances": optuna.visualization.plot_param_importances,
            "parallel_coordinate": optuna.visualization.plot_parallel_coordinate,
            "slice": optuna.visualization.plot_slice,
            "contour": optuna.visualization.plot_contour, # Requer pelo menos 2 params
            "intermediate_values": optuna.visualization.plot_intermediate_values, # Se pruner for usado
        }

        for plot_name, plot_func in plots_to_generate.items():
            try:
                if plot_name == "contour" and len(self.study.best_params) < 2:
                    logger.debug("Plot de contorno requer pelo menos 2 parâmetros otimizados.")
                    continue
                if plot_name == "slice" and not self.study.best_params:
                    logger.debug("Plot slice requer parâmetros definidos.")
                    continue

                fig = plot_func(self.study)
                filepath = self.optimization_results_path / f"opt_{study_name_for_file}_{plot_name}.html"
                fig.write_html(str(filepath))
                logger.info(f"Plot de otimização '{plot_name}' salvo em: {filepath}")
            except (ImportError, ValueError, RuntimeError, AttributeError) as e_plot: # Capturar mais erros de plot
                logger.warning(f"Não foi possível gerar ou salvar plot de otimização '{plot_name}': {e_plot}")
            except Exception as e_plot_generic:
                logger.error(f"Erro inesperado ao gerar plot '{plot_name}': {e_plot_generic}")



    async def validate_parameters_on_test_set(self, params: Dict[str, Any], test_data: pd.DataFrame) -> Dict[str, Any]: # Renomeado
        """Valida um conjunto de parâmetros em dados de teste (out-of-sample)."""
        strategy = self.strategy_class()
        strategy.update_parameters(params)

        backtest_engine = BacktestEngine()
        results_obj: BacktestResults = backtest_engine.run( # Tipagem
            strategy=strategy,
            data=test_data,
            initial_balance=getattr(CONFIG, 'VALIDATION_INITIAL_BALANCE', 10000),
            commission=getattr(CONFIG, 'VALIDATION_COMMISSION', 0.00002)
        )
        return results_obj.to_dict() # Retornar como dict


    def save_study_results(self, filename_suffix: Optional[str] = None): # Renomeado
        """Salva o estudo Optuna completo e os melhores parâmetros."""
        if not self.study:
            logger.error("Nenhum estudo de otimização para salvar.")
            return

        study_name_file = self.study.study_name or f"{self.strategy_class.__name__}_{datetime.now(timezone.utc).strftime('%Y%m%d')}"
        if filename_suffix:
            study_name_file = f"{study_name_file}_{filename_suffix}"

        # Salvar estudo completo (permite reanálise posterior)
        study_filepath = self.optimization_results_path / f"study_{study_name_file}.pkl"
        try:
            joblib.dump(self.study, study_filepath)
            logger.info(f"Estudo Optuna completo salvo em: {study_filepath}")
        except Exception as e_save_study: # Renomeado
            logger.error(f"Erro ao salvar estudo Optuna: {e_save_study}")


        # Salvar melhores parâmetros em JSON para fácil leitura e uso
        if self.best_params:
            params_filepath = self.optimization_results_path / f"best_params_{study_name_file}.json"
            try:
                with open(params_filepath, 'w') as f:
                    json.dump(self.best_params, f, indent=4)
                logger.info(f"Melhores parâmetros salvos em: {params_filepath}")
            except Exception as e_save_params: # Renomeado
                logger.error(f"Erro ao salvar melhores parâmetros: {e_save_params}")


        # Salvar plots
        self._save_optimization_plots(study_name_file)


    def load_best_parameters(self, study_name_prefix: Optional[str] = None) -> Optional[Dict[str, Any]]: # Renomeado
        """Carrega os melhores parâmetros de um estudo salvo anteriormente."""
        # Lógica para encontrar o arquivo de parâmetros mais recente ou específico
        if not study_name_prefix:
            study_name_prefix = self.strategy_class.__name__

        param_files = sorted(
            self.optimization_results_path.glob(f"best_params_{study_name_prefix}*.json"),
            key=os.path.getmtime,
            reverse=True
        )
        if param_files:
            latest_param_file = param_files[0]
            try:
                with open(latest_param_file, 'r') as f:
                    loaded_params = json.load(f)
                self.best_params = loaded_params
                logger.info(f"Melhores parâmetros carregados de: {latest_param_file} -> {self.best_params}")
                return self.best_params
            except Exception as e_load: # Renomeado
                logger.error(f"Erro ao carregar melhores parâmetros de {latest_param_file}: {e_load}")
        else:
            logger.warning(f"Nenhum arquivo de parâmetros encontrado para '{study_name_prefix}' em {self.optimization_results_path}")
        return None



class WalkForwardOptimizer:
    """Otimizador com análise walk-forward completa, usando StrategyOptimizer para cada janela."""

    def __init__(self, strategy_class: Type[BaseStrategy]): # Usar Type
        self.strategy_class = strategy_class
        self.optimization_windows_results: List[Dict[str, Any]] = [] # Renomeado e tipado
        self.aggregated_oos_results: Dict[str, Any] = {} # Para resultados agregados
        self.wf_results_path = Path(CONFIG.MODELS_PATH) / "walk_forward" # Novo caminho
        self.wf_results_path.mkdir(parents=True, exist_ok=True)


    async def run_walk_forward_optimization(self, historical_data_wf: pd.DataFrame, # Renomeado
                              train_months_wf: int = 6, # Renomeado
                              test_months_wf: int = 1, # Renomeado
                              step_months_wf: int = 1, # Renomeado
                              n_trials_per_window: int = 50, # Adicionado
                              timeout_per_window_seconds: int = 1800): # Adicionado
        """Executa otimização walk-forward completa."""
        logger.info(f"Iniciando otimização Walk-Forward para {self.strategy_class.__name__}")
        logger.info(f"Config: Treino={train_months_wf}m, Teste={test_months_wf}m, Passo={step_months_wf}m, Trials/Janela={n_trials_per_window}")

        if historical_data_wf.empty:
            logger.error("Dados históricos para Walk-Forward estão vazios. Encerrando.")
            return

        # Gerar janelas de treino/teste
        windows = self._create_walk_forward_windows(
            historical_data_wf, train_months_wf, test_months_wf, step_months_wf
        )
        if not windows:
            logger.error("Nenhuma janela de walk-forward pôde ser criada. Verifique os dados e parâmetros de período.")
            return

        logger.info(f"Total de {len(windows)} janelas de walk-forward a serem processadas.")
        self.optimization_windows_results = [] # Limpar resultados anteriores

        all_oos_trades_for_aggregation: List[Dict[str, Any]] = [] # Para agregar trades OOS

        for i, window in enumerate(windows):
            logger.info(f"\n--- Processando Janela Walk-Forward {i + 1}/{len(windows)} ---")
            logger.info(f"  Período de Treino: {window['train_start'].strftime('%Y-%m-%d')} a {window['train_end'].strftime('%Y-%m-%d')}")
            logger.info(f"  Período de Teste (Out-of-Sample): {window['test_start'].strftime('%Y-%m-%d')} a {window['test_end'].strftime('%Y-%m-%d')}")

            if window['train_data'].empty or window['test_data'].empty:
                logger.warning(f"Janela {i+1} tem dados de treino ou teste vazios. Pulando.")
                self.optimization_windows_results.append({
                    'window_number': i + 1, 'status': 'skipped_empty_data', **window})
                continue

            # Otimizar parâmetros na janela de treino
            # Aqui, StrategyOptimizer usa um split interno (train/validation) nos dados de treino da janela WF.
            optimizer = StrategyOptimizer(self.strategy_class, window['train_data'])
            await optimizer.optimize(n_trials_opt=n_trials_per_window, timeout_opt_seconds=timeout_per_window_seconds)

            window_result: Dict[str, Any] = { # Tipagem
                'window_number': i + 1,
                'train_period_start': window['train_start'].isoformat(),
                'train_period_end': window['train_end'].isoformat(),
                'test_period_start': window['test_start'].isoformat(),
                'test_period_end': window['test_end'].isoformat(),
                'best_params_in_sample': optimizer.best_params,
                'in_sample_score': optimizer.study.best_value if optimizer.study and optimizer.study.best_trial else None, # type: ignore
                'out_of_sample_performance': None,
                'status': 'pending_validation'
            }

            if optimizer.best_params:
                # Validar parâmetros otimizados no período de teste (out-of-sample) da janela WF
                logger.info(f"  Validando melhores parâmetros OOS para janela {i+1}: {optimizer.best_params}")
                oos_performance_dict = await optimizer.validate_parameters_on_test_set( # Renomeado
                    optimizer.best_params,
                    window['test_data'] # Passar os dados de teste da janela WF
                )
                window_result['out_of_sample_performance'] = oos_performance_dict
                window_result['status'] = 'completed'

                if oos_performance_dict.get('trades'): # Se BacktestResults tiver 'trades' como lista de dicts
                    all_oos_trades_for_aggregation.extend(oos_performance_dict['trades'])


                logger.info(f"  Janela {i+1} OOS: Trades={oos_performance_dict.get('total_trades', 0)}, "
                           f"Sharpe={oos_performance_dict.get('sharpe_ratio', 0):.3f}, PnL=${oos_performance_dict.get('net_pnl',0):.2f}")
            else:
                logger.warning(f"  Nenhum parâmetro ótimo encontrado para a janela de treino {i+1}. Não foi possível validar OOS.")
                window_result['status'] = 'optimization_failed'


            self.optimization_windows_results.append(window_result)
            optimizer.save_study_results(filename_suffix=f"wf_window_{i+1}") # Salvar estudo da janela


        self._analyze_and_aggregate_walk_forward_results(all_oos_trades_for_aggregation)
        self.save_walk_forward_summary()


    def _create_walk_forward_windows(self, historical_data: pd.DataFrame,
                                    train_months: int, test_months: int, step_months: int) -> List[Dict[str, Any]]:
        """Cria as janelas de treino e teste para o walk-forward."""
        windows = []
        if historical_data.empty or not isinstance(historical_data.index, pd.DatetimeIndex):
            logger.error("Dados históricos inválidos ou sem DatetimeIndex para criar janelas WF.")
            return windows

        min_required_data_months = train_months + test_months
        if (historical_data.index.max() - historical_data.index.min()) < pd.DateOffset(months=min_required_data_months):
            logger.error(f"Dados históricos insuficientes para a configuração de walk-forward (mínimo {min_required_data_months} meses).")
            return windows


        # Data inicial para o fim da primeira janela de treino
        # (e início da primeira janela de teste)
        current_test_start_date = historical_data.index.min() + pd.DateOffset(months=train_months)

        while current_test_start_date + pd.DateOffset(months=test_months) <= historical_data.index.max():
            train_start_date = current_test_start_date - pd.DateOffset(months=train_months)
            train_end_date = current_test_start_date - pd.Timedelta(days=1) # Treino até o dia anterior ao teste
            # test_start_date é current_test_start_date
            test_end_date = current_test_start_date + pd.DateOffset(months=test_months) - pd.Timedelta(days=1) # Teste até o fim do período

            # Filtrar dados para a janela atual
            # Usar loc para slicing baseado em data
            train_data_window = historical_data.loc[train_start_date : train_end_date]
            test_data_window = historical_data.loc[current_test_start_date : test_end_date]

            if not train_data_window.empty and not test_data_window.empty:
                windows.append({
                    'train_start': train_start_date,
                    'train_end': train_end_date,
                    'test_start': current_test_start_date,
                    'test_end': test_end_date,
                    'train_data': train_data_window,
                    'test_data': test_data_window
                })
            else:
                logger.warning(f"Janela WF de {current_test_start_date.date()} resultou em dados vazios. Pulando.")


            # Avançar para a próxima janela
            current_test_start_date += pd.DateOffset(months=step_months)

        return windows


    def _analyze_and_aggregate_walk_forward_results(self, all_oos_trades: List[Dict[str,Any]]): # Usar Any
        """Analisa os resultados OOS agregados de todas as janelas walk-forward."""
        if not self.optimization_windows_results:
            logger.warning("Nenhum resultado de janela walk-forward para analisar.")
            return

        logger.info("\n\n=== Análise Agregada dos Resultados Walk-Forward (Out-of-Sample) ===")
        num_windows = len(self.optimization_windows_results)
        successful_optimizations = sum(1 for w in self.optimization_windows_results if w.get('out_of_sample_performance'))
        logger.info(f"Total de Janelas Processadas: {num_windows}")
        logger.info(f"Janelas com Validação OOS Bem-sucedida: {successful_optimizations}")

        if successful_optimizations == 0:
            logger.warning("Nenhuma janela teve validação OOS bem-sucedida. Análise agregada não pode prosseguir.")
            return

        # Coletar métricas OOS de cada janela
        oos_sharpes = [w['out_of_sample_performance'].get('sharpe_ratio', 0) for w in self.optimization_windows_results if w.get('out_of_sample_performance')]
        oos_pnls = [w['out_of_sample_performance'].get('net_pnl', 0) for w in self.optimization_windows_results if w.get('out_of_sample_performance')]
        oos_win_rates = [w['out_of_sample_performance'].get('win_rate', 0) for w in self.optimization_windows_results if w.get('out_of_sample_performance')]
        oos_max_dds = [w['out_of_sample_performance'].get('max_drawdown', 0) for w in self.optimization_windows_results if w.get('out_of_sample_performance')]
        oos_total_trades = [w['out_of_sample_performance'].get('total_trades', 0) for w in self.optimization_windows_results if w.get('out_of_sample_performance')]


        self.aggregated_oos_results = {
            'avg_oos_sharpe_ratio': np.mean(oos_sharpes) if oos_sharpes else 0.0,
            'median_oos_sharpe_ratio': np.median(oos_sharpes) if oos_sharpes else 0.0,
            'std_dev_oos_sharpe_ratio': np.std(oos_sharpes) if oos_sharpes else 0.0,
            'total_oos_pnl': np.sum(oos_pnls),
            'avg_oos_pnl_per_window': np.mean(oos_pnls) if oos_pnls else 0.0,
            'avg_oos_win_rate': np.mean(oos_win_rates) if oos_win_rates else 0.0,
            'avg_oos_max_drawdown': np.mean(oos_max_dds) if oos_max_dds else 0.0,
            'avg_oos_trades_per_window': np.mean(oos_total_trades) if oos_total_trades else 0.0,
            'consistency_profitable_windows_pct': (sum(1 for pnl in oos_pnls if pnl > 0) / len(oos_pnls) * 100) if oos_pnls else 0.0
        }

        logger.info(f"  Sharpe Ratio Médio OOS: {self.aggregated_oos_results['avg_oos_sharpe_ratio']:.3f}")
        logger.info(f"  PnL Total OOS Agregado: ${self.aggregated_oos_results['total_oos_pnl']:.2f}")
        logger.info(f"  Win Rate Médio OOS: {self.aggregated_oos_results['avg_oos_win_rate']:.2%}")
        logger.info(f"  Consistência (Janelas Lucrativas): {self.aggregated_oos_results['consistency_profitable_windows_pct']:.1f}%")

        # Análise de estabilidade dos parâmetros otimizados entre janelas
        self._analyze_walk_forward_parameter_stability()

        # Calcular métricas sobre a curva de equity OOS concatenada (se 'all_oos_trades' foi populado)
        if all_oos_trades:
            # É preciso um BacktestEngine ou Scorer para calcular métricas sobre a lista de trades
            # Supondo que o Scorer possa fazer isso
            scorer = StrategyScorer()
            # O Scorer.calculate_metrics espera uma lista de dicionários de trades
            # Verifique se o formato de 'all_oos_trades' é compatível
            # Ex: [{'pnl': 10, 'duration': 300, 'capital': 10000}, ...]
            # Se BacktestResults.trades é List[BacktestTrade], converter para List[Dict]
            # performance_agregada_oos = scorer.calculate_metrics(all_oos_trades_dict_list)
            # logger.info("\nMétricas da Curva de Equity OOS Concatenada:")
            # logger.info(f"  Sharpe: {performance_agregada_oos.sharpe_ratio:.3f}")
            # logger.info(f"  Max DD: {performance_agregada_oos.max_drawdown:.2%}")
            # self.aggregated_oos_results['concatenated_equity_metrics'] = performance_agregada_oos.to_dict()
            pass # Implementar se necessário e se o formato dos trades for compatível



    def _analyze_walk_forward_parameter_stability(self): # Renomeado
        """Analisa a estabilidade dos parâmetros 'best_params_in_sample' entre as janelas walk-forward."""
        if len(self.optimization_windows_results) < 2:
            logger.info("Estabilidade de parâmetros não pode ser analisada (janelas insuficientes).")
            return

        param_series: Dict[str, List[Any]] = defaultdict(list) # Adicionada tipagem

        for window_res in self.optimization_windows_results:
            if window_res.get('best_params_in_sample'):
                for param_key, param_val in window_res['best_params_in_sample'].items(): # Renomeado param, value
                    param_series[param_key].append(param_val)

        logger.info("\n--- Estabilidade dos Parâmetros Otimizados (Walk-Forward) ---")
        parameter_stability_report: Dict[str, Dict[str, Any]] = {} # Adicionada tipagem

        for param_key, values_list in param_series.items(): # Renomeado param, values
            if not values_list: continue

            # Calcular estatísticas apenas se os valores forem numéricos
            if all(isinstance(v, (int, float)) for v in values_list):
                mean_val = np.mean(values_list)
                std_val = np.std(values_list)
                cv = (std_val / abs(mean_val)) if abs(mean_val) > 1e-9 else 0.0 # Coeficiente de Variação
                min_val, max_val = np.min(values_list), np.max(values_list)

                logger.info(f"Parâmetro '{param_key}':")
                logger.info(f"  Média: {mean_val:.3f}, Desvio Padrão: {std_val:.3f}, CV: {cv:.2%}")
                logger.info(f"  Range: [{min_val:.3f} - {max_val:.3f}]")
                if cv > 0.5: # Limiar arbitrário para alta variabilidade
                    logger.warning(f"  ATENÇÃO: Parâmetro '{param_key}' mostra alta variabilidade (CV > 50%).")
                parameter_stability_report[param_key] = {'mean': mean_val, 'std': std_val, 'cv': cv, 'min': min_val, 'max': max_val, 'values': values_list}
            elif all(isinstance(v, bool) for v in values_list): # Para parâmetros booleanos
                mode = max(set(values_list), key=values_list.count)
                freq = values_list.count(mode) / len(values_list)
                logger.info(f"Parâmetro Booleano '{param_key}': Moda = {mode} (Frequência: {freq:.2%})")
                parameter_stability_report[param_key] = {'mode': mode, 'frequency_of_mode': freq, 'values': values_list}
            # Adicionar tratamento para outros tipos se necessário (ex: strings categóricas)

        self.aggregated_oos_results['parameter_stability'] = parameter_stability_report


    def save_walk_forward_summary(self, filename_suffix: Optional[str] = None):
        """Salva um resumo dos resultados do walk-forward e os parâmetros por janela."""
        summary_data = {
            'strategy_name': self.strategy_class.__name__,
            'run_timestamp_utc': datetime.now(timezone.utc).isoformat(),
            'walk_forward_config': {
                # Adicionar aqui os parâmetros de configuração do WF se forem passados para __init__
                # 'train_months': self.train_months, ...
            },
            'aggregated_out_of_sample_results': self.aggregated_oos_results,
            'window_details': self.optimization_windows_results # Contém params e performance OOS por janela
        }
        filename_base = f"wf_summary_{self.strategy_class.__name__}"
        if filename_suffix:
            filename_base = f"{filename_base}_{filename_suffix}"
        summary_filepath = self.wf_results_path / f"{filename_base}.json"

        try:
            with open(summary_filepath, 'w') as f:
                json.dump(summary_data, f, indent=4, default=str) # default=str para lidar com datetimes, etc.
            logger.info(f"Resumo da análise Walk-Forward salvo em: {summary_filepath}")
        except Exception as e_save_wf: # Renomeado
            logger.error(f"Erro ao salvar resumo Walk-Forward: {e_save_wf}")




async def scheduled_optimization(strategy_classes: List[Type[BaseStrategy]], data_manager_instance: DataManager): # Renomeado
    """Executa otimização agendada para uma lista de estratégias."""
    logger.info(f"Iniciando otimização agendada para {len(strategy_classes)} estratégias.")

    for strategy_cls in strategy_classes: # Renomeado
        logger.info(f"\n--- Otimizando Estratégia: {strategy_cls.__name__} ---")
        try:
            # Obter dados históricos (ex: últimos 12 meses para otimização)
            # A quantidade de dados deve ser suficiente para o split train/validation do StrategyOptimizer
            days_for_opt = (getattr(CONFIG, 'OPTIMIZATION_HISTORICAL_DATA_MONTHS', 12)) * 30 # Ex: 12 meses
            historical_data_opt = await data_manager_instance.get_historical_ticks( # Renomeado
                CONFIG.SYMBOL,
                days=days_for_opt
            )

            if historical_data_opt is None or historical_data_opt.empty or len(historical_data_opt) < getattr(CONFIG, 'OPTIMIZATION_MIN_DATA_POINTS', 1000):
                logger.warning(f"Dados históricos insuficientes ou nulos para otimizar {strategy_cls.__name__}. Pulando.")
                continue

            # Criar e executar otimizador para a estratégia
            optimizer = StrategyOptimizer(strategy_cls, historical_data_opt)
            # Usar n_trials e timeout da configuração global ou específicos da estratégia
            await optimizer.optimize(
                 n_trials_opt=getattr(CONFIG, f'OPTIMIZATION_TRIALS_{strategy_cls.__name__.upper()}', optimizer.n_trials),
                 timeout_opt_seconds=getattr(CONFIG, f'OPTIMIZATION_TIMEOUT_{strategy_cls.__name__.upper()}', optimizer.timeout_seconds)
            )


            if optimizer.best_params:
                # Salvar parâmetros otimizados no DataManager (ex: SQLite)
                await data_manager_instance.save_strategy_params(
                    strategy_cls.__name__,
                    optimizer.best_params
                )
                logger.info(f"Parâmetros otimizados para {strategy_cls.__name__} salvos no DataManager.")

                # Salvar estudo completo da otimização (inclui todos os trials)
                optimizer.save_study_results(filename_suffix=f"scheduled_{datetime.now(timezone.utc).strftime('%Y%m%d')}")
            else:
                logger.warning(f"Otimização para {strategy_cls.__name__} não resultou em parâmetros ótimos.")


        except Exception as e_sched_opt: # Renomeado
            logger.exception(f"Erro ao otimizar estratégia {strategy_cls.__name__} na rotina agendada:")
            continue # Continuar com a próxima estratégia

    logger.info("Otimização agendada de todas as estratégias concluída.")