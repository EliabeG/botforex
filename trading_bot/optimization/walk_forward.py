# optimization/walk_forward.py
"""Walk-forward analysis para otimização robusta"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import asyncio
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

from optimization.auto_optimizer import StrategyOptimizer
from optimization.scoring import StrategyScorer, PerformanceMetrics
from backtest.engine import BacktestEngine
from utils.logger import setup_logger

logger = setup_logger("walk_forward")

@dataclass
class WalkForwardWindow:
    """Janela de walk-forward"""
    window_id: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    train_data: pd.DataFrame
    test_data: pd.DataFrame
    optimization_results: Optional[Dict] = None
    out_of_sample_results: Optional[Dict] = None
    
    @property
    def train_days(self) -> int:
        return (self.train_end - self.train_start).days
    
    @property
    def test_days(self) -> int:
        return (self.test_end - self.test_start).days

class WalkForwardAnalysis:
    """Análise walk-forward completa para estratégias"""
    
    def __init__(self, strategy_class: type):
        self.strategy_class = strategy_class
        self.windows: List[WalkForwardWindow] = []
        self.aggregated_results = {}
        self.scorer = StrategyScorer()
        self.parameter_stability = {}
        
        # Configurações
        self.min_train_trades = 100
        self.min_test_trades = 20
        self.optimization_trials = 50
        self.parallel_windows = mp.cpu_count() - 1
        
    async def run(self, 
                  historical_data: pd.DataFrame,
                  train_months: int = 6,
                  test_months: int = 1,
                  step_months: int = 1,
                  start_date: Optional[datetime] = None) -> Dict:
        """
        Executa análise walk-forward completa
        
        Args:
            historical_data: Dados históricos completos
            train_months: Meses para treino
            test_months: Meses para teste
            step_months: Meses para avançar janela
            start_date: Data inicial (opcional)
        
        Returns:
            Resultados agregados da análise
        """
        logger.info(f"Iniciando walk-forward analysis para {self.strategy_class.__name__}")
        logger.info(f"Config: Train={train_months}m, Test={test_months}m, Step={step_months}m")
        
        # Preparar janelas
        self.windows = self._create_windows(
            historical_data,
            train_months,
            test_months,
            step_months,
            start_date
        )
        
        logger.info(f"Criadas {len(self.windows)} janelas para análise")
        
        # Otimizar cada janela
        results = await self._optimize_windows()
        
        # Analisar resultados
        self._analyze_results()
        
        # Calcular métricas agregadas
        self._calculate_aggregated_metrics()
        
        # Analisar estabilidade dos parâmetros
        self._analyze_parameter_stability()
        
        return self.aggregated_results
    
    def _create_windows(self,
                       data: pd.DataFrame,
                       train_months: int,
                       test_months: int,
                       step_months: int,
                       start_date: Optional[datetime]) -> List[WalkForwardWindow]:
        """Cria janelas de walk-forward"""
        windows = []
        
        # Determinar datas
        data_start = data.index.min()
        data_end = data.index.max()
        
        if start_date is None:
            start_date = data_start + timedelta(days=train_months * 30)
        
        current_date = start_date
        window_id = 0
        
        while current_date + timedelta(days=test_months * 30) <= data_end:
            # Definir períodos
            train_start = current_date - timedelta(days=train_months * 30)
            train_end = current_date
            test_start = current_date
            test_end = current_date + timedelta(days=test_months * 30)
            
            # Filtrar dados
            train_data = data[(data.index >= train_start) & (data.index < train_end)]
            test_data = data[(data.index >= test_start) & (data.index < test_end)]
            
            # Verificar se há dados suficientes
            if len(train_data) >= self.min_train_trades * 10:  # Assumindo ~10 ticks por trade
                window = WalkForwardWindow(
                    window_id=window_id,
                    train_start=train_start,
                    train_end=train_end,
                    test_start=test_start,
                    test_end=test_end,
                    train_data=train_data,
                    test_data=test_data
                )
                windows.append(window)
                window_id += 1
            
            # Avançar janela
            current_date += timedelta(days=step_months * 30)
        
        return windows
    
    async def _optimize_windows(self) -> List[Dict]:
        """Otimiza todas as janelas"""
        results = []
        
        # Processar janelas em paralelo (com limite)
        for i in range(0, len(self.windows), self.parallel_windows):
            batch = self.windows[i:i + self.parallel_windows]
            
            # Otimizar batch em paralelo
            tasks = [self._optimize_single_window(window) for window in batch]
            batch_results = await asyncio.gather(*tasks)
            
            results.extend(batch_results)
            
            logger.info(f"Processadas {min(i + self.parallel_windows, len(self.windows))}/{len(self.windows)} janelas")
        
        return results
    
    async def _optimize_single_window(self, window: WalkForwardWindow) -> Dict:
        """Otimiza uma única janela"""
        logger.info(f"Otimizando janela {window.window_id}: "
                   f"{window.train_start.date()} a {window.train_end.date()}")
        
        try:
            # Criar otimizador
            optimizer = StrategyOptimizer(self.strategy_class, window.train_data)
            
            # Otimizar
            await optimizer.optimize(
                n_trials=self.optimization_trials,
                timeout=1800  # 30 min por janela
            )
            
            # Salvar resultados de otimização
            window.optimization_results = {
                'best_params': optimizer.best_params,
                'best_score': optimizer.study.best_value if optimizer.study else 0,
                'n_trials': len(optimizer.study.trials) if optimizer.study else 0
            }
            
            # Testar out-of-sample
            if optimizer.best_params:
                oos_results = await self._test_out_of_sample(
                    window.test_data,
                    optimizer.best_params
                )
                window.out_of_sample_results = oos_results
                
                logger.info(f"Janela {window.window_id} - OOS: "
                           f"Trades={oos_results.get('total_trades', 0)}, "
                           f"Sharpe={oos_results.get('sharpe_ratio', 0):.2f}")
            
            return {
                'window_id': window.window_id,
                'optimization': window.optimization_results,
                'out_of_sample': window.out_of_sample_results
            }
            
        except Exception as e:
            logger.error(f"Erro ao otimizar janela {window.window_id}: {e}")
            return {
                'window_id': window.window_id,
                'error': str(e)
            }
    
    async def _test_out_of_sample(self, test_data: pd.DataFrame, 
                                  params: Dict) -> Dict:
        """Testa parâmetros out-of-sample"""
        # Criar estratégia com parâmetros otimizados
        strategy = self.strategy_class()
        strategy.update_parameters(params)
        
        # Executar backtest
        backtest = BacktestEngine()
        results = await backtest.run_async(
            strategy=strategy,
            data=test_data,
            initial_balance=10000,
            commission=0.00002
        )
        
        return results
    
    def _analyze_results(self):
        """Analisa resultados de todas as janelas"""
        successful_windows = [w for w in self.windows if w.out_of_sample_results]
        
        logger.info(f"\n=== Análise Walk-Forward ===")
        logger.info(f"Janelas totais: {len(self.windows)}")
        logger.info(f"Janelas bem-sucedidas: {len(successful_windows)}")
        
        if not successful_windows:
            logger.warning("Nenhuma janela com resultados válidos")
            return
        
        # Estatísticas por janela
        for window in successful_windows:
            oos = window.out_of_sample_results
            logger.info(f"\nJanela {window.window_id}:")
            logger.info(f"  Período: {window.test_start.date()} a {window.test_end.date()}")
            logger.info(f"  Trades: {oos.get('total_trades', 0)}")
            logger.info(f"  PnL: ${oos.get('total_pnl', 0):.2f}")
            logger.info(f"  Sharpe: {oos.get('sharpe_ratio', 0):.2f}")
            logger.info(f"  Win Rate: {oos.get('win_rate', 0):.1%}")
    
    def _calculate_aggregated_metrics(self):
        """Calcula métricas agregadas de todas as janelas"""
        oos_results = [w.out_of_sample_results for w in self.windows 
                      if w.out_of_sample_results]
        
        if not oos_results:
            self.aggregated_results = {}
            return
        
        # Agregar trades
        all_trades = []
        for result in oos_results:
            if 'trades' in result:
                all_trades.extend(result['trades'])
        
        # Calcular métricas agregadas
        if all_trades:
            aggregated_metrics = self.scorer.calculate_metrics(all_trades)
            self.aggregated_results['metrics'] = aggregated_metrics.to_dict()
        
        # Estatísticas das janelas
        self.aggregated_results['windows_stats'] = {
            'total_windows': len(self.windows),
            'successful_windows': len(oos_results),
            'avg_trades_per_window': np.mean([r.get('total_trades', 0) for r in oos_results]),
            'avg_sharpe': np.mean([r.get('sharpe_ratio', 0) for r in oos_results]),
            'avg_win_rate': np.mean([r.get('win_rate', 0) for r in oos_results]),
            'total_pnl': sum(r.get('total_pnl', 0) for r in oos_results)
        }
        
        # Consistência
        profitable_windows = sum(1 for r in oos_results if r.get('total_pnl', 0) > 0)
        self.aggregated_results['consistency'] = profitable_windows / len(oos_results)
        
        logger.info(f"\n=== Métricas Agregadas ===")
        logger.info(f"Consistência: {self.aggregated_results['consistency']:.1%}")
        logger.info(f"Sharpe médio: {self.aggregated_results['windows_stats']['avg_sharpe']:.2f}")
        logger.info(f"PnL total: ${self.aggregated_results['windows_stats']['total_pnl']:.2f}")
    
    def _analyze_parameter_stability(self):
        """Analisa estabilidade dos parâmetros otimizados"""
        # Coletar parâmetros de todas as janelas
        all_params = {}
        
        for window in self.windows:
            if window.optimization_results and window.optimization_results.get('best_params'):
                params = window.optimization_results['best_params']
                
                for param_name, value in params.items():
                    if param_name not in all_params:
                        all_params[param_name] = []
                    all_params[param_name].append(value)
        
        # Calcular estatísticas
        self.parameter_stability = {}
        
        logger.info(f"\n=== Estabilidade dos Parâmetros ===")
        
        for param_name, values in all_params.items():
            if values and isinstance(values[0], (int, float)):
                stability = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'cv': np.std(values) / np.mean(values) if np.mean(values) != 0 else 0,
                    'min': min(values),
                    'max': max(values),
                    'range': max(values) - min(values)
                }
                
                self.parameter_stability[param_name] = stability
                
                logger.info(f"\n{param_name}:")
                logger.info(f"  Média: {stability['mean']:.3f}")
                logger.info(f"  Desvio: {stability['std']:.3f}")
                logger.info(f"  CV: {stability['cv']:.1%}")
                logger.info(f"  Range: [{stability['min']:.3f}, {stability['max']:.3f}]")
                
                if stability['cv'] > 0.3:
                    logger.warning(f"  ⚠️ Alta variabilidade detectada!")
    
    def get_robust_parameters(self) -> Dict[str, Any]:
        """Retorna parâmetros robustos baseados na análise"""
        robust_params = {}
        
        # Usar mediana dos parâmetros para robustez
        for window in self.windows:
            if window.optimization_results and window.optimization_results.get('best_params'):
                params = window.optimization_results['best_params']
                
                for param_name, value in params.items():
                    if param_name not in robust_params:
                        robust_params[param_name] = []
                    robust_params[param_name].append(value)
        
        # Calcular mediana
        final_params = {}
        for param_name, values in robust_params.items():
            if values:
                if isinstance(values[0], bool):
                    # Para booleanos, usar moda
                    final_params[param_name] = max(set(values), key=values.count)
                elif isinstance(values[0], (int, float)):
                    # Para numéricos, usar mediana
                    final_params[param_name] = np.median(values)
                    if isinstance(values[0], int):
                        final_params[param_name] = int(final_params[param_name])
                else:
                    # Para outros tipos, usar primeiro valor
                    final_params[param_name] = values[0]
        
        return final_params
    
    def generate_report(self, filepath: str):
        """Gera relatório detalhado da análise"""
        report = {
            'strategy': self.strategy_class.__name__,
            'analysis_date': datetime.now().isoformat(),
            'windows': len(self.windows),
            'aggregated_results': self.aggregated_results,
            'parameter_stability': self.parameter_stability,
            'robust_parameters': self.get_robust_parameters(),
            'window_details': []
        }
        
        # Detalhes de cada janela
        for window in self.windows:
            if window.out_of_sample_results:
                report['window_details'].append({
                    'window_id': window.window_id,
                    'train_period': f"{window.train_start.date()} to {window.train_end.date()}",
                    'test_period': f"{window.test_start.date()} to {window.test_end.date()}",
                    'best_params': window.optimization_results.get('best_params', {}),
                    'in_sample_score': window.optimization_results.get('best_score', 0),
                    'out_of_sample_results': {
                        'total_trades': window.out_of_sample_results.get('total_trades', 0),
                        'total_pnl': window.out_of_sample_results.get('total_pnl', 0),
                        'sharpe_ratio': window.out_of_sample_results.get('sharpe_ratio', 0),
                        'win_rate': window.out_of_sample_results.get('win_rate', 0),
                        'max_drawdown': window.out_of_sample_results.get('max_drawdown', 0)
                    }
                })
        
        # Salvar relatório
        import json
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Relatório salvo em: {filepath}")
        
        return report