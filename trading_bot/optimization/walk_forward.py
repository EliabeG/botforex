# optimization/walk_forward.py
"""Walk-forward analysis para otimização robusta de estratégias de trading."""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Type # Adicionado Type
from datetime import datetime, timedelta, timezone # Adicionado timezone
from dataclasses import dataclass, field # Adicionado field
import asyncio
# from concurrent.futures import ProcessPoolExecutor # Removido se Optuna já lida com paralelismo
import multiprocessing as mp
import json # Para salvar o relatório
from pathlib import Path # Para manipulação de caminhos

from optimization.auto_optimizer import StrategyOptimizer # StrategyOptimizer é importado
from optimization.scoring import StrategyScorer, PerformanceMetrics # PerformanceMetrics é importado
from backtest.engine import BacktestEngine, BacktestResults # Adicionado BacktestResults
from strategies.base_strategy import BaseStrategy # Adicionado BaseStrategy
from utils.logger import setup_logger
from config.settings import CONFIG # Para caminhos de salvamento e configurações

logger = setup_logger("walk_forward_analysis") # Renomeado logger

@dataclass
class WalkForwardWindow:
    """Representa uma única janela de treino e teste na análise walk-forward."""
    window_id: int
    train_start_dt: datetime # Renomeado para clareza e tipo
    train_end_dt: datetime   # Renomeado para clareza e tipo
    test_start_dt: datetime  # Renomeado para clareza e tipo
    test_end_dt: datetime    # Renomeado para clareza e tipo
    train_data: pd.DataFrame # Dados de treino para esta janela
    test_data: pd.DataFrame  # Dados de teste (out-of-sample) para esta janela
    optimization_results: Optional[Dict[str, Any]] = field(default_factory=dict) # Resultados da otimização In-Sample
    out_of_sample_performance: Optional[BacktestResults] = None # Resultados do backtest Out-of-Sample (como objeto)

    @property
    def train_duration_days(self) -> int: # Renomeado
        return (self.train_end_dt - self.train_start_dt).days

    @property
    def test_duration_days(self) -> int: # Renomeado
        return (self.test_end_dt - self.test_start_dt).days

class WalkForwardAnalysis:
    """Executa uma análise walk-forward completa para uma dada estratégia."""

    def __init__(self, strategy_class_to_test: Type[BaseStrategy]): # Renomeado e tipado
        self.strategy_class = strategy_class_to_test # Renomeado
        self.windows: List[WalkForwardWindow] = []
        self.aggregated_results: Dict[str, Any] = {} # Para métricas agregadas OOS
        self.scorer = StrategyScorer() # Usado para avaliar a performance OOS agregada
        self.parameter_stability_report: Dict[str, Dict[str, Any]] = {} # Renomeado

        # Configurações (podem vir de CONFIG)
        self.min_train_trades_per_opt: int = getattr(CONFIG, 'WFA_MIN_TRAIN_TRADES_OPT', 50) # Mínimo de trades na otimização IS
        self.min_test_trades_oos: int = getattr(CONFIG, 'WFA_MIN_TEST_TRADES_OOS', 10)    # Mínimo de trades no teste OOS
        self.optimization_trials_per_window: int = getattr(CONFIG, 'WFA_OPT_TRIALS_WINDOW', 50) # Trials do Optuna por janela
        # Paralelização de janelas: Se cada otimização de janela já usa múltiplos cores (n_jobs em Optuna),
        # rodar múltiplas janelas em paralelo pode sobrecarregar.
        # Se a otimização de janela for sequencial, então paralelizar janelas faz sentido.
        # Assumindo que StrategyOptimizer pode ser pesado, limitar paralelismo aqui.
        self.max_parallel_windows: int = max(1, (mp.cpu_count() or 1) // 2) # Ex: Metade dos cores para janelas WF

        self.results_path = Path(CONFIG.MODELS_PATH) / "walk_forward_analysis"
        self.results_path.mkdir(parents=True, exist_ok=True)


    async def run_analysis(self, # Renomeado de run
                  historical_data_full: pd.DataFrame, # Renomeado
                  train_period_months: int = 6, # Renomeado
                  test_period_months: int = 1,  # Renomeado
                  step_period_months: int = 1,  # Renomeado
                  analysis_start_date: Optional[datetime] = None) -> Dict[str, Any]: # Renomeado
        """
        Executa a análise walk-forward completa.
        """
        logger.info(f"Iniciando Análise Walk-Forward para: {self.strategy_class.__name__}")
        logger.info(f"Config: Treino={train_period_months}m, Teste={test_period_months}m, Passo={step_period_months}m")

        if historical_data_full.empty or not isinstance(historical_data_full.index, pd.DatetimeIndex):
            logger.error("Dados históricos inválidos ou sem DatetimeIndex para análise walk-forward.")
            self.aggregated_results = {"error": "Dados históricos inválidos."}
            return self.aggregated_results

        self.windows = self._create_walk_forward_windows( # Renomeado
            historical_data_full,
            train_period_months,
            test_period_months,
            step_period_months,
            analysis_start_date
        )

        if not self.windows:
            logger.error("Nenhuma janela walk-forward pôde ser criada. Verifique os dados e parâmetros.")
            self.aggregated_results = {"error": "Nenhuma janela WF criada."}
            return self.aggregated_results

        logger.info(f"Criadas {len(self.windows)} janelas para a análise walk-forward.")

        # Otimizar e testar cada janela
        # A paralelização aqui deve ser feita com cuidado. Se StrategyOptimizer já usa n_jobs,
        # rodar _process_single_window em paralelo pode criar processos demais.
        # Uma abordagem é usar um ProcessPoolExecutor para as janelas, e dentro de cada janela,
        # o StrategyOptimizer usa n_jobs=1 ou um número limitado.
        # Ou, como no original, usar asyncio.gather com um limite de concorrência.

        semaphore = asyncio.Semaphore(self.max_parallel_windows) # Limitar concorrência
        tasks = []
        for window_obj in self.windows: # Renomeado window para window_obj
            tasks.append(self._process_single_window_concurrently(window_obj, semaphore)) # Renomeado

        window_processing_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Atualizar as janelas com os resultados (ou erros)
        for i, result_or_exc in enumerate(window_processing_results):
            if isinstance(result_or_exc, Exception):
                logger.error(f"Erro ao processar janela {self.windows[i].window_id}: {result_or_exc}")
                self.windows[i].optimization_results = {"error": str(result_or_exc)}
                self.windows[i].out_of_sample_performance = None # Marcar como falha
            # Se _process_single_window_concurrently já atualiza self.windows[i], esta parte pode não ser necessária
            # Mas é bom ter uma forma de consolidar os resultados aqui.
            # A forma como os resultados são retornados de _process_single_window_concurrently dita isso.
            # No original, _optimize_single_window atualizava o objeto window diretamente.


        self._log_window_summary() # Renomeado
        self._calculate_and_log_aggregated_metrics() # Renomeado
        self._analyze_and_log_parameter_stability() # Renomeado

        report_data = self.generate_final_report_data() # Renomeado
        self.save_report_to_json(report_data, f"WFA_Report_{self.strategy_class.__name__}.json") # Renomeado

        return self.aggregated_results


    def _create_walk_forward_windows(self, # Renomeado
                       full_data: pd.DataFrame, # Renomeado
                       train_months: int, test_months: int, step_months: int,
                       start_dt_param: Optional[datetime]) -> List[WalkForwardWindow]: # Renomeado
        """Cria as janelas de treino e teste para a análise walk-forward."""
        processed_windows: List[WalkForwardWindow] = [] # Renomeado
        data_start_dt = full_data.index.min() # Renomeado
        data_end_dt = full_data.index.max() # Renomeado

        # Determinar a data de início para o final da primeira janela de treino
        current_train_end_dt = start_dt_param if start_dt_param else data_start_dt + pd.DateOffset(months=train_months)
        if current_train_end_dt < data_start_dt + pd.DateOffset(months=train_months): # Garantir que há dados de treino suficientes
            current_train_end_dt = data_start_dt + pd.DateOffset(months=train_months)


        window_idx = 0 # Renomeado
        while current_train_end_dt + pd.DateOffset(months=test_months) <= data_end_dt:
            train_start = current_train_end_dt - pd.DateOffset(months=train_months)
            # train_end é current_train_end_dt (exclusive, ou o dia anterior inclusive)
            # Para slicing, é mais fácil usar current_train_end_dt como o ponto de corte.
            test_start = current_train_end_dt
            test_end = current_train_end_dt + pd.DateOffset(months=test_months)

            # Ajustar para garantir que as datas estejam dentro dos limites dos dados
            train_start = max(train_start, data_start_dt)
            test_end = min(test_end, data_end_dt)
            
            # Slice dos dados
            # Usar .loc para slicing por data. Ser inclusivo no final do treino e teste.
            train_data_slice = full_data.loc[train_start : test_start - pd.Timedelta(microseconds=1)] # Treino até um microssegundo antes do teste
            test_data_slice = full_data.loc[test_start : test_end - pd.Timedelta(microseconds=1)] # Teste até um microssegundo antes do fim


            # Verificar se há dados suficientes em ambas as janelas
            # Um critério de número mínimo de pontos de dados pode ser mais robusto que trades
            min_data_points_train = getattr(CONFIG, 'WFA_MIN_DATAPOINTS_TRAIN', 2000) # Ex: 2000 ticks/barras
            min_data_points_test = getattr(CONFIG, 'WFA_MIN_DATAPOINTS_TEST', 500)

            if len(train_data_slice) >= min_data_points_train and len(test_data_slice) >= min_data_points_test:
                window = WalkForwardWindow(
                    window_id=window_idx,
                    train_start_dt=train_data_slice.index.min(), # Usar o início real dos dados
                    train_end_dt=train_data_slice.index.max(),   # Usar o fim real dos dados
                    test_start_dt=test_data_slice.index.min(),
                    test_end_dt=test_data_slice.index.max(),
                    train_data=train_data_slice,
                    test_data=test_data_slice
                )
                processed_windows.append(window)
                window_idx += 1
            else:
                logger.warning(f"Janela de {test_start.date()} pulada: dados insuficientes "
                               f"(Treino: {len(train_data_slice)}/{min_data_points_train}, "
                               f"Teste: {len(test_data_slice)}/{min_data_points_test}).")


            current_train_end_dt += pd.DateOffset(months=step_months)
        return processed_windows


    async def _process_single_window_concurrently(self, window: WalkForwardWindow, semaphore: asyncio.Semaphore) -> None: # Renomeado
        """Processa uma única janela (otimização e teste OOS), gerenciado por semáforo."""
        async with semaphore: # Controlar concorrência
            logger.info(f"Iniciando processamento da Janela Walk-Forward {window.window_id}: "
                       f"Treino de {window.train_start_dt.date()} a {window.train_end_dt.date()}")
            try:
                optimizer = StrategyOptimizer(self.strategy_class, window.train_data)
                await optimizer.optimize(
                    n_trials_opt=self.optimization_trials_per_window,
                    timeout_opt_seconds=getattr(CONFIG, 'WFA_OPT_TIMEOUT_WINDOW_S', 1800) # 30 min por janela
                )

                window.optimization_results = {
                    'best_params': optimizer.best_params,
                    'best_in_sample_score': optimizer.study.best_value if optimizer.study and optimizer.study.best_trial else None, # type: ignore
                    'total_trials': len(optimizer.study.trials) if optimizer.study else 0 # type: ignore
                }

                if optimizer.best_params:
                    logger.info(f"  Janela {window.window_id} - Melhores Parâmetros IS: {optimizer.best_params}")
                    logger.info(f"  Janela {window.window_id} - Validando OOS de {window.test_start_dt.date()} a {window.test_end_dt.date()}...")

                    # Testar out-of-sample com os parâmetros otimizados
                    # StrategyOptimizer.validate_parameters_on_test_set retorna um dict de performance
                    # Precisamos converter para BacktestResults se necessário, ou ajustar
                    backtest_engine_oos = BacktestEngine() # Nova instância para OOS
                    oos_results_obj: BacktestResults = backtest_engine_oos.run( # Tipagem
                        strategy=self.strategy_class().update_parameters(optimizer.best_params), # Criar nova instância com params
                        data=window.test_data,
                        initial_balance=getattr(CONFIG, 'WFA_OOS_INITIAL_BALANCE', 10000),
                        commission=getattr(CONFIG, 'WFA_OOS_COMMISSION', 0.00002)
                    )
                    window.out_of_sample_performance = oos_results_obj # Armazenar objeto BacktestResults

                    logger.info(f"  Janela {window.window_id} - Performance OOS: "
                               f"Trades={oos_results_obj.total_trades}, "
                               f"Sharpe={oos_results_obj.sharpe_ratio:.2f}, "
                               f"PnL=${oos_results_obj.net_pnl:.2f}")
                else:
                    logger.warning(f"  Janela {window.window_id} - Otimização não encontrou melhores parâmetros. Teste OOS não realizado.")
                    window.out_of_sample_performance = BacktestResults() # Resultado vazio

            except Exception as e_proc_win: # Renomeado
                logger.exception(f"Erro crítico ao processar janela walk-forward {window.window_id}:")
                # Marcar a janela com erro
                window.optimization_results = {"error": str(e_proc_win)}
                window.out_of_sample_performance = BacktestResults(total_pnl = -float('inf')) # Sinalizar erro
            # Não precisa retornar nada, pois o objeto 'window' é modificado diretamente.


    def _log_window_summary(self): # Renomeado
        """Loga um resumo dos resultados de todas as janelas."""
        logger.info("\n\n" + "="*30 + " RESUMO DAS JANELAS WALK-FORWARD " + "="*30)
        successful_windows_count = 0 # Renomeado

        for window_obj in self.windows: # Renomeado
            logger.info(f"\n--- Janela ID: {window_obj.window_id} ---")
            logger.info(f"  Período de Treino: {window_obj.train_start_dt.date()} a {window_obj.train_end_dt.date()} ({window_obj.train_duration_days} dias)")
            logger.info(f"  Período de Teste (OOS): {window_obj.test_start_dt.date()} a {window_obj.test_end_dt.date()} ({window_obj.test_duration_days} dias)")

            if window_obj.optimization_results and not window_obj.optimization_results.get('error'):
                logger.info(f"  Parâmetros Otimizados (IS): {window_obj.optimization_results.get('best_params')}")
                logger.info(f"  Score IS: {window_obj.optimization_results.get('best_in_sample_score', 'N/A'):.4f}")
            else:
                logger.warning(f"  Otimização falhou ou não produziu parâmetros para a janela {window_obj.window_id}.")


            if window_obj.out_of_sample_performance and window_obj.out_of_sample_performance.total_trades > 0 : # Checar se existe e tem trades
                oos_perf = window_obj.out_of_sample_performance
                logger.info(f"  Performance OOS: Trades={oos_perf.total_trades}, PnL=${oos_perf.net_pnl:.2f}, "
                           f"Sharpe={oos_perf.sharpe_ratio:.2f}, WinRate={oos_perf.win_rate:.2%}, MaxDD={oos_perf.max_drawdown_pct:.2%}")
                if oos_perf.net_pnl > 0 : successful_windows_count +=1
            elif window_obj.out_of_sample_performance: # Existe mas sem trades
                 logger.info(f"  Performance OOS: Sem trades executados.")
            else:
                logger.warning(f"  Performance OOS não disponível ou inválida para a janela {window_obj.window_id}.")

        logger.info("="*80)
        logger.info(f"Total de Janelas Processadas: {len(self.windows)}")
        logger.info(f"Total de Janelas com PnL OOS Positivo: {successful_windows_count}")


    def _calculate_and_log_aggregated_metrics(self): # Renomeado
        """Calcula e loga métricas agregadas de todas as performances OOS."""
        all_oos_performances = [w.out_of_sample_performance for w in self.windows if w.out_of_sample_performance and w.out_of_sample_performance.total_trades > 0]

        if not all_oos_performances:
            logger.warning("Nenhuma performance OOS válida encontrada para calcular métricas agregadas.")
            self.aggregated_results = {'status': 'sem_resultados_oos_validos'}
            return

        # Concatenar todos os trades OOS para uma análise de curva de equity global
        all_oos_trades_list: List[BacktestTrade] = [] # Tipagem explícita
        for perf_obj in all_oos_performances: # Renomeado perf para perf_obj
            all_oos_trades_list.extend(perf_obj.trades) # perf_obj.trades é List[BacktestTrade]

        # Calcular métricas sobre a curva de equity OOS concatenada
        if all_oos_trades_list:
            # O Scorer espera List[Dict], então converter BacktestTrade para dict
            all_oos_trades_dict_list = [trade.to_dict() if hasattr(trade, 'to_dict') else vars(trade) for trade in all_oos_trades_list]
            
            # Determinar a duração total em dias para anualização correta
            first_trade_oos = min(t['entry_time'] for t in all_oos_trades_dict_list if t.get('entry_time')) if all_oos_trades_dict_list else None
            last_trade_oos = max(t['exit_time'] for t in all_oos_trades_dict_list if t.get('exit_time')) if all_oos_trades_dict_list else None
            total_duration_days_oos = (last_trade_oos - first_trade_oos).days if first_trade_oos and last_trade_oos else len(self.windows) * 30 # Estimativa

            aggregated_oos_metrics_obj: PerformanceMetrics = self.scorer.calculate_all_performance_metrics(
                all_oos_trades_dict_list,
                initial_balance=getattr(CONFIG, 'WFA_OOS_INITIAL_BALANCE', 10000), # Usar o mesmo balanço inicial
                total_duration_days=total_duration_days_oos
            )
            self.aggregated_results['concatenated_oos_metrics'] = aggregated_oos_metrics_obj.to_dict()
            logger.info("\n--- Métricas da Curva de Equity OOS Concatenada ---")
            for key, val in aggregated_oos_metrics_obj.to_dict().items():
                if isinstance(val, float): logger.info(f"  {key.replace('_', ' ').capitalize()}: {val:.3f}")
                else: logger.info(f"  {key.replace('_', ' ').capitalize()}: {val}")


        # Estatísticas das métricas OOS por janela
        avg_sharpe = np.mean([p.sharpe_ratio for p in all_oos_performances])
        avg_win_rate = np.mean([p.win_rate for p in all_oos_performances])
        total_pnl_sum = np.sum([p.net_pnl for p in all_oos_performances])
        profitable_windows_count = sum(1 for p in all_oos_performances if p.net_pnl > 0) # Renomeado

        self.aggregated_results['per_window_oos_stats'] = {
            'num_valid_oos_windows': len(all_oos_performances),
            'avg_trades_per_window': np.mean([p.total_trades for p in all_oos_performances]),
            'avg_sharpe_ratio_per_window': avg_sharpe,
            'median_sharpe_ratio_per_window': np.median([p.sharpe_ratio for p in all_oos_performances]),
            'avg_win_rate_per_window': avg_win_rate,
            'total_pnl_across_all_windows': total_pnl_sum,
            'consistency_profitable_windows_pct': (profitable_windows_count / len(all_oos_performances) * 100) if all_oos_performances else 0.0
        }
        logger.info("\n--- Estatísticas Médias por Janela OOS ---")
        for key, val in self.aggregated_results['per_window_oos_stats'].items():
            if isinstance(val, float): logger.info(f"  {key.replace('_', ' ').capitalize()}: {val:.3f}")
            else: logger.info(f"  {key.replace('_', ' ').capitalize()}: {val}")


    def _analyze_and_log_parameter_stability(self): # Renomeado
        """Analisa e loga a estabilidade dos parâmetros otimizados entre as janelas."""
        all_best_params_per_window: Dict[str, List[Any]] = defaultdict(list) # Renomeado

        for window_obj in self.windows: # Renomeado
            if window_obj.optimization_results and window_obj.optimization_results.get('best_params'):
                params_dict = window_obj.optimization_results['best_params'] # Renomeado
                for param_name_stab, param_value_stab in params_dict.items(): # Renomeado param_name, value
                    all_best_params_per_window[param_name_stab].append(param_value_stab)

        if not all_best_params_per_window:
            logger.warning("Nenhum parâmetro otimizado encontrado para análise de estabilidade.")
            return

        logger.info("\n--- Análise de Estabilidade dos Parâmetros Otimizados (In-Sample) ---")
        self.parameter_stability_report.clear() # Limpar relatório anterior

        for param_name_stab, values_list_stab in all_best_params_per_window.items(): # Renomeado
            if not values_list_stab: continue

            if all(isinstance(v_stab, (int, float)) for v_stab in values_list_stab): # Renomeado v para v_stab
                mean_val = np.mean(values_list_stab)
                std_val = np.std(values_list_stab)
                cv = (std_val / abs(mean_val)) if abs(mean_val) > 1e-9 else np.nan # Coeficiente de Variação
                min_val, max_val = np.min(values_list_stab), np.max(values_list_stab)
                stability_info = {
                    'mean': mean_val, 'std': std_val, 'cv': cv,
                    'min': min_val, 'max': max_val, 'range': max_val - min_val,
                    'values_over_windows': values_list_stab # Guardar os valores
                }
                self.parameter_stability_report[param_name_stab] = stability_info
                logger.info(f"Parâmetro '{param_name_stab}': Média={mean_val:.3f}, StdDev={std_val:.3f}, CV={cv:.2%}, Range=[{min_val:.3f}-{max_val:.3f}]")
                if cv > getattr(CONFIG, 'WFA_PARAM_CV_THRESHOLD_WARN', 0.5): # Limiar de CV para aviso
                    logger.warning(f"  -> ATENÇÃO: Parâmetro '{param_name_stab}' demonstra alta variabilidade (CV > 50%) entre janelas.")
            elif all(isinstance(v_stab, bool) for v_stab in values_list_stab):
                 mode_val = max(set(values_list_stab), key=values_list_stab.count)
                 freq_val = values_list_stab.count(mode_val) / len(values_list_stab)
                 self.parameter_stability_report[param_name_stab] = {'mode': mode_val, 'frequency_of_mode': freq_val, 'values_over_windows': values_list_stab}
                 logger.info(f"Parâmetro Booleano '{param_name_stab}': Moda={mode_val} (Freq: {freq_val:.2%})")
            # Adicionar lógica para outros tipos de parâmetros (ex: strings categóricas)


    def get_robust_parameters_from_wfa(self) -> Dict[str, Any]: # Renomeado
        """
        Determina um conjunto de parâmetros robustos com base na análise walk-forward.
        Exemplo: mediana ou média dos melhores parâmetros de cada janela.
        """
        if not self.parameter_stability_report: # Se o relatório de estabilidade não foi gerado
            self._analyze_and_log_parameter_stability() # Tentar gerar

        final_robust_params: Dict[str, Any] = {} # Renomeado
        if not self.parameter_stability_report:
            logger.warning("Não foi possível determinar parâmetros robustos: análise de estabilidade vazia.")
            # Fallback: tentar usar os parâmetros do último StrategyOptimizer bem-sucedido
            for window in reversed(self.windows):
                if window.optimization_results and window.optimization_results.get('best_params'):
                    logger.info("Usando best_params da última janela WF como fallback para parâmetros robustos.")
                    return window.optimization_results['best_params']
            return self.strategy_class().get_default_parameters() # Último fallback


        for param_name_robust, stability_data in self.parameter_stability_report.items(): # Renomeado
            if 'values_over_windows' not in stability_data: continue

            values = stability_data['values_over_windows']
            if not values: continue

            if isinstance(values[0], (int, float)):
                # Para numéricos, usar mediana é geralmente mais robusto a outliers que a média
                median_val = np.median(values)
                final_robust_params[param_name_robust] = int(round(median_val)) if isinstance(values[0], int) else round(median_val, 8) # Arredondar para precisão
            elif isinstance(values[0], bool):
                final_robust_params[param_name_robust] = stability_data.get('mode', values[0]) # Usar a moda
            else: # Para outros tipos (ex: strings), usar o mais frequente (moda) ou o último
                try:
                    final_robust_params[param_name_robust] = max(set(values), key=values.count)
                except TypeError: # Se os valores não forem hasheáveis (ex: dicts)
                     final_robust_params[param_name_robust] = values[-1] # Usar o último


        logger.info(f"Parâmetros Robustos determinados pela Análise Walk-Forward: {final_robust_params}")
        if not final_robust_params: # Se ainda vazio, usar defaults da estratégia
            logger.warning("Nenhum parâmetro robusto pôde ser determinado. Usando defaults da estratégia.")
            return self.strategy_class().get_default_parameters()

        return final_robust_params


    def generate_final_report_data(self) -> Dict[str, Any]: # Renomeado
        """Compila todos os dados relevantes em um dicionário para o relatório JSON."""
        report_dict = { # Renomeado
            'strategy_name': self.strategy_class.__name__, # Renomeado
            'analysis_timestamp_utc': datetime.now(timezone.utc).isoformat(), # Renomeado e UTC
            'total_walk_forward_windows': len(self.windows),
            'configuration': {
                # Adicionar aqui os parâmetros de configuração do WFA se passados para __init__
                # 'train_months': self.train_period_months, ...
            },
            'aggregated_out_of_sample_results': self.aggregated_results.get('concatenated_oos_metrics', {}),
            'per_window_out_of_sample_stats': self.aggregated_results.get('per_window_oos_stats', {}),
            'parameter_stability_analysis': self.parameter_stability_report,
            'derived_robust_parameters': self.get_robust_parameters_from_wfa(), # Renomeado
            'individual_window_details': []
        }

        for window_obj in self.windows: # Renomeado
            oos_perf_dict = window_obj.out_of_sample_performance.to_dict() if window_obj.out_of_sample_performance else {}
            report_dict['individual_window_details'].append({
                'window_id': window_obj.window_id,
                'train_period': f"{window_obj.train_start_dt.date()} a {window_obj.train_end_dt.date()}",
                'test_period_oos': f"{window_obj.test_start_dt.date()} a {window_obj.test_end_dt.date()}",
                'in_sample_best_params': window_obj.optimization_results.get('best_params', {}),
                'in_sample_score': window_obj.optimization_results.get('best_in_sample_score'),
                'out_of_sample_performance_summary': {
                    'total_trades': oos_perf_dict.get('total_trades', 0),
                    'net_pnl': oos_perf_dict.get('net_pnl', 0.0),
                    'sharpe_ratio': oos_perf_dict.get('sharpe_ratio', 0.0),
                    'win_rate': oos_perf_dict.get('win_rate', 0.0),
                    'max_drawdown_pct': oos_perf_dict.get('max_drawdown_pct', 0.0)
                }
                # Poderia adicionar o objeto oos_perf_dict completo se desejado
            })
        return report_dict


    def save_report_to_json(self, report_data: Dict[str, Any], filename: str): # Renomeado
        """Salva o dicionário do relatório em um arquivo JSON."""
        filepath_report = self.results_path / filename # Renomeado
        try:
            with open(filepath_report, 'w') as f:
                json.dump(report_data, f, indent=2, default=str) # default=str para lidar com tipos não serializáveis
            logger.info(f"Relatório Walk-Forward completo salvo em: {filepath_report}")
        except Exception as e_save_json: # Renomeado
            logger.exception(f"Erro ao salvar relatório Walk-Forward JSON em {filepath_report}:")