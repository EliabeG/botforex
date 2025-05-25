# optimization/scoring.py
"""Sistema de scoring para avaliação de estratégias"""
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import statistics
from datetime import datetime, timedelta

from utils.logger import setup_logger
from utils.helpers import calculate_sharpe_ratio, calculate_max_drawdown

logger = setup_logger("scoring")

@dataclass
class PerformanceMetrics:
    """Métricas de performance de uma estratégia"""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    
    total_pnl: float = 0.0
    average_win: float = 0.0
    average_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    
    expectancy: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0
    average_drawdown: float = 0.0
    recovery_factor: float = 0.0
    
    avg_trade_duration: float = 0.0
    avg_bars_in_trade: int = 0
    
    consecutive_wins: int = 0
    consecutive_losses: int = 0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    
    risk_adjusted_return: float = 0.0
    information_ratio: float = 0.0
    omega_ratio: float = 0.0
    
    def to_dict(self) -> Dict:
        """Converte para dicionário"""
        return {k: v for k, v in self.__dict__.items()}

class StrategyScorer:
    """Calculador de score para estratégias"""
    
    def __init__(self):
        # Pesos para cálculo do score
        self.weights = {
            'win_rate': 0.15,
            'expectancy': 0.20,
            'sharpe_ratio': 0.25,
            'profit_factor': 0.10,
            'max_drawdown': 0.15,
            'consistency': 0.10,
            'risk_adjusted': 0.05
        }
        
        # Thresholds para normalização
        self.thresholds = {
            'win_rate': {'min': 0.3, 'max': 0.7, 'target': 0.55},
            'expectancy': {'min': -0.5, 'max': 2.0, 'target': 0.5},
            'sharpe_ratio': {'min': 0.0, 'max': 3.0, 'target': 1.5},
            'profit_factor': {'min': 0.8, 'max': 2.0, 'target': 1.3},
            'max_drawdown': {'min': 0.05, 'max': 0.30, 'target': 0.10},
            'consistency': {'min': 0.0, 'max': 1.0, 'target': 0.7}
        }
    
    def calculate_score(self, performance: Dict) -> float:
        """
        Calcula score geral da estratégia
        
        Args:
            performance: Dicionário com métricas de performance
        
        Returns:
            Score entre 0 e 1
        """
        try:
            # Verificar se tem trades suficientes
            if performance.get('total_trades', 0) < 10:
                return 0.0
            
            # Calcular componentes do score
            components = {
                'win_rate': self._score_win_rate(performance),
                'expectancy': self._score_expectancy(performance),
                'sharpe_ratio': self._score_sharpe(performance),
                'profit_factor': self._score_profit_factor(performance),
                'max_drawdown': self._score_drawdown(performance),
                'consistency': self._score_consistency(performance),
                'risk_adjusted': self._score_risk_adjusted(performance)
            }
            
            # Calcular score ponderado
            weighted_score = sum(
                components[metric] * weight 
                for metric, weight in self.weights.items()
            )
            
            # Log componentes para debug
            logger.debug(f"Score components: {components}")
            logger.debug(f"Weighted score: {weighted_score:.4f}")
            
            return np.clip(weighted_score, 0, 1)
            
        except Exception as e:
            logger.error(f"Erro ao calcular score: {e}")
            return 0.0
    
    def calculate_metrics(self, trades: List[Dict]) -> PerformanceMetrics:
        """
        Calcula todas as métricas de performance
        
        Args:
            trades: Lista de trades executados
        
        Returns:
            Objeto PerformanceMetrics
        """
        metrics = PerformanceMetrics()
        
        if not trades:
            return metrics
        
        # Separar wins e losses
        wins = [t['pnl'] for t in trades if t['pnl'] > 0]
        losses = [abs(t['pnl']) for t in trades if t['pnl'] < 0]
        
        # Métricas básicas
        metrics.total_trades = len(trades)
        metrics.winning_trades = len(wins)
        metrics.losing_trades = len(losses)
        metrics.win_rate = metrics.winning_trades / metrics.total_trades if metrics.total_trades > 0 else 0
        
        # PnL
        metrics.total_pnl = sum(t['pnl'] for t in trades)
        metrics.average_win = np.mean(wins) if wins else 0
        metrics.average_loss = np.mean(losses) if losses else 0
        metrics.largest_win = max(wins) if wins else 0
        metrics.largest_loss = max(losses) if losses else 0
        
        # Expectancy
        metrics.expectancy = self._calculate_expectancy(wins, losses)
        
        # Profit Factor
        metrics.profit_factor = self._calculate_profit_factor(wins, losses)
        
        # Ratios
        returns = [t['pnl'] / t.get('capital', 10000) for t in trades]
        metrics.sharpe_ratio = calculate_sharpe_ratio(returns)
        metrics.sortino_ratio = self._calculate_sortino_ratio(returns)
        
        # Drawdown
        equity_curve = self._build_equity_curve(trades)
        metrics.max_drawdown, _, _ = calculate_max_drawdown(equity_curve)
        metrics.average_drawdown = self._calculate_average_drawdown(equity_curve)
        
        # Duração média
        durations = [t.get('duration', 0) for t in trades if 'duration' in t]
        metrics.avg_trade_duration = np.mean(durations) if durations else 0
        
        # Consecutivos
        metrics = self._calculate_consecutive_stats(trades, metrics)
        
        # Risk-adjusted
        metrics.risk_adjusted_return = metrics.total_pnl / metrics.max_drawdown if metrics.max_drawdown > 0 else 0
        
        # Recovery Factor
        metrics.recovery_factor = metrics.total_pnl / metrics.max_drawdown if metrics.max_drawdown > 0 else 0
        
        return metrics
    
    def _score_win_rate(self, performance: Dict) -> float:
        """Score baseado em win rate"""
        win_rate = performance.get('win_rate', 0)
        threshold = self.thresholds['win_rate']
        
        # Penalizar win rates extremos (muito alto pode indicar overfitting)
        if win_rate > 0.8:
            return 0.7
        
        return self._normalize_metric(win_rate, threshold)
    
    def _score_expectancy(self, performance: Dict) -> float:
        """Score baseado em expectancy"""
        expectancy = performance.get('expectancy', 0)
        threshold = self.thresholds['expectancy']
        
        return self._normalize_metric(expectancy, threshold)
    
    def _score_sharpe(self, performance: Dict) -> float:
        """Score baseado em Sharpe ratio"""
        sharpe = performance.get('sharpe_ratio', 0)
        threshold = self.thresholds['sharpe_ratio']
        
        return self._normalize_metric(sharpe, threshold)
    
    def _score_profit_factor(self, performance: Dict) -> float:
        """Score baseado em profit factor"""
        pf = performance.get('profit_factor', 0)
        threshold = self.thresholds['profit_factor']
        
        return self._normalize_metric(pf, threshold)
    
    def _score_drawdown(self, performance: Dict) -> float:
        """Score baseado em drawdown (invertido)"""
        dd = performance.get('max_drawdown', 1)
        threshold = self.thresholds['max_drawdown']
        
        # Inverter: menor drawdown = maior score
        if dd <= threshold['target']:
            return 1.0
        elif dd >= threshold['max']:
            return 0.0
        else:
            return 1 - ((dd - threshold['target']) / (threshold['max'] - threshold['target']))
    
    def _score_consistency(self, performance: Dict) -> float:
        """Score baseado em consistência"""
        # Usar desvio padrão dos retornos mensais
        monthly_returns = performance.get('monthly_returns', [])
        
        if len(monthly_returns) < 3:
            return 0.5
        
        # Calcular percentual de meses positivos
        positive_months = sum(1 for r in monthly_returns if r > 0)
        consistency = positive_months / len(monthly_returns)
        
        threshold = self.thresholds['consistency']
        return self._normalize_metric(consistency, threshold)
    
    def _score_risk_adjusted(self, performance: Dict) -> float:
        """Score baseado em retorno ajustado ao risco"""
        risk_adj = performance.get('risk_adjusted_return', 0)
        
        # Normalizar entre 0 e 1
        if risk_adj <= 0:
            return 0.0
        elif risk_adj >= 5:
            return 1.0
        else:
            return risk_adj / 5
    
    def _normalize_metric(self, value: float, threshold: Dict) -> float:
        """Normaliza métrica para score entre 0 e 1"""
        if value <= threshold['min']:
            return 0.0
        elif value >= threshold['max']:
            return 1.0
        elif value <= threshold['target']:
            # Escala de min para target (0 a 0.8)
            return 0.8 * (value - threshold['min']) / (threshold['target'] - threshold['min'])
        else:
            # Escala de target para max (0.8 a 1.0)
            return 0.8 + 0.2 * (value - threshold['target']) / (threshold['max'] - threshold['target'])
    
    def _calculate_expectancy(self, wins: List[float], losses: List[float]) -> float:
        """Calcula expectancy"""
        if not wins and not losses:
            return 0
        
        total_trades = len(wins) + len(losses)
        win_rate = len(wins) / total_trades if total_trades > 0 else 0
        
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        
        return (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
    
    def _calculate_profit_factor(self, wins: List[float], losses: List[float]) -> float:
        """Calcula profit factor"""
        total_wins = sum(wins) if wins else 0
        total_losses = sum(losses) if losses else 0
        
        if total_losses == 0:
            return 3.0 if total_wins > 0 else 0
        
        return total_wins / total_losses
    
    def _calculate_sortino_ratio(self, returns: List[float], 
                                target_return: float = 0) -> float:
        """Calcula Sortino ratio"""
        if not returns:
            return 0
        
        excess_returns = [r - target_return for r in returns]
        downside_returns = [r for r in excess_returns if r < 0]
        
        if not downside_returns:
            return 3.0  # Sem downside
        
        avg_excess = np.mean(excess_returns)
        downside_std = np.std(downside_returns)
        
        if downside_std == 0:
            return 0
        
        return avg_excess / downside_std * np.sqrt(252)
    
    def _build_equity_curve(self, trades: List[Dict]) -> List[float]:
        """Constrói curva de equity"""
        initial_capital = trades[0].get('capital', 10000) if trades else 10000
        equity = [initial_capital]
        
        for trade in trades:
            equity.append(equity[-1] + trade['pnl'])
        
        return equity
    
    def _calculate_average_drawdown(self, equity_curve: List[float]) -> float:
        """Calcula drawdown médio"""
        if len(equity_curve) < 2:
            return 0
        
        drawdowns = []
        peak = equity_curve[0]
        
        for value in equity_curve[1:]:
            if value > peak:
                peak = value
            dd = (peak - value) / peak if peak > 0 else 0
            drawdowns.append(dd)
        
        return np.mean(drawdowns) if drawdowns else 0
    
    def _calculate_consecutive_stats(self, trades: List[Dict], 
                                   metrics: PerformanceMetrics) -> PerformanceMetrics:
        """Calcula estatísticas de trades consecutivos"""
        if not trades:
            return metrics
        
        current_wins = 0
        current_losses = 0
        
        for trade in trades:
            if trade['pnl'] > 0:
                current_wins += 1
                current_losses = 0
                metrics.max_consecutive_wins = max(metrics.max_consecutive_wins, current_wins)
            else:
                current_losses += 1
                current_wins = 0
                metrics.max_consecutive_losses = max(metrics.max_consecutive_losses, current_losses)
        
        metrics.consecutive_wins = current_wins
        metrics.consecutive_losses = current_losses
        
        return metrics
    
    def compare_strategies(self, strategies: Dict[str, Dict]) -> List[Tuple[str, float]]:
        """
        Compara múltiplas estratégias
        
        Args:
            strategies: Dict com nome -> performance
        
        Returns:
            Lista ordenada de (nome, score)
        """
        scores = []
        
        for name, performance in strategies.items():
            score = self.calculate_score(performance)
            scores.append((name, score))
        
        # Ordenar por score decrescente
        scores.sort(key=lambda x: x[1], reverse=True)
        
        return scores
    
    def update_weights(self, new_weights: Dict[str, float]):
        """Atualiza pesos do scoring"""
        # Normalizar pesos para somar 1
        total = sum(new_weights.values())
        
        if total > 0:
            self.weights = {k: v/total for k, v in new_weights.items()}
            logger.info(f"Pesos atualizados: {self.weights}")