# backtest/analyzer.py
"""Analisador avançado de resultados de backtest"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from backtest.engine import BacktestResults, BacktestTrade
from utils.logger import setup_logger

logger = setup_logger("backtest_analyzer")

class BacktestAnalyzer:
    """Analisador detalhado de resultados de backtest"""
    
    def __init__(self, results: BacktestResults):
        self.results = results
        self.trades_df = self._create_trades_dataframe()
        
    def _create_trades_dataframe(self) -> pd.DataFrame:
        """Converte lista de trades em DataFrame"""
        if not self.results.trades:
            return pd.DataFrame()
        
        trades_data = []
        for trade in self.results.trades:
            trades_data.append({
                'id': trade.id,
                'strategy': trade.strategy,
                'side': trade.side,
                'entry_time': trade.entry_time,
                'exit_time': trade.exit_time,
                'duration_hours': trade.duration_seconds / 3600,
                'entry_price': trade.entry_price,
                'exit_price': trade.exit_price,
                'size': trade.size,
                'pnl': trade.pnl,
                'pnl_pips': trade.pnl_pips,
                'commission': trade.commission,
                'exit_reason': trade.exit_reason,
                'max_profit': trade.max_profit,
                'max_loss': trade.max_loss
            })
        
        df = pd.DataFrame(trades_data)
        df['win'] = df['pnl'] > 0
        
        # Adicionar métricas derivadas
        df['return_pct'] = df['pnl'] / self.results.initial_balance * 100
        df['mae'] = df['max_loss'].abs()  # Maximum Adverse Excursion
        df['mfe'] = df['max_profit']      # Maximum Favorable Excursion
        
        return df
    
    def generate_full_report(self) -> Dict:
        """Gera relatório completo de análise"""
        report = {
            'summary': self.get_summary_statistics(),
            'performance_metrics': self.calculate_performance_metrics(),
            'risk_metrics': self.calculate_risk_metrics(),
            'trade_analysis': self.analyze_trades(),
            'time_analysis': self.analyze_time_patterns(),
            'exit_analysis': self.analyze_exits(),
            'monte_carlo': self.monte_carlo_analysis(),
            'recommendations': self.generate_recommendations()
        }
        
        return report
    
    def get_summary_statistics(self) -> Dict:
        """Estatísticas resumidas"""
        return {
            'total_return': (self.results.final_balance - self.results.initial_balance) / self.results.initial_balance,
            'annualized_return': self._calculate_annualized_return(),
            'total_trades': self.results.total_trades,
            'win_rate': self.results.win_rate,
            'avg_trade_return': self.results.expectancy / self.results.initial_balance if self.results.initial_balance > 0 else 0,
            'best_trade': self.results.largest_win,
            'worst_trade': -self.results.largest_loss,
            'avg_win_loss_ratio': self.results.average_win / self.results.average_loss if self.results.average_loss > 0 else 0,
            'total_commission': self.results.total_commission,
            'net_profit': self.results.net_pnl
        }
    
    def calculate_performance_metrics(self) -> Dict:
        """Métricas de performance avançadas"""
        metrics = {}
        
        if len(self.trades_df) > 0:
            # Profit Factor detalhado
            long_trades = self.trades_df[self.trades_df['side'] == 'buy']
            short_trades = self.trades_df[self.trades_df['side'] == 'sell']
            
            metrics['long_profit_factor'] = self._calculate_profit_factor(long_trades)
            metrics['short_profit_factor'] = self._calculate_profit_factor(short_trades)
            
            # Win rates por lado
            metrics['long_win_rate'] = (long_trades['win'].sum() / len(long_trades)) if len(long_trades) > 0 else 0
            metrics['short_win_rate'] = (short_trades['win'].sum() / len(short_trades)) if len(short_trades) > 0 else 0
            
            # Payoff ratio
            avg_win = self.trades_df[self.trades_df['win']]['pnl'].mean()
            avg_loss = abs(self.trades_df[~self.trades_df['win']]['pnl'].mean())
            metrics['payoff_ratio'] = avg_win / avg_loss if avg_loss > 0 else 0
            
            # Expectancy score
            metrics['expectancy_score'] = (metrics['payoff_ratio'] * self.results.win_rate) - (1 - self.results.win_rate)
            
            # Consistency metrics
            monthly_returns = self._calculate_monthly_returns()
            if len(monthly_returns) > 0:
                metrics['monthly_win_rate'] = sum(r > 0 for r in monthly_returns.values()) / len(monthly_returns)
                metrics['best_month'] = max(monthly_returns.values()) if monthly_returns else 0
                metrics['worst_month'] = min(monthly_returns.values()) if monthly_returns else 0
            
            # Recovery factor
            metrics['recovery_factor'] = self.results.net_pnl / (self.results.max_drawdown * self.results.initial_balance) if self.results.max_drawdown > 0 else 0
            
            # System Quality Number (SQN)
            if len(self.trades_df) > 1:
                avg_r = self.trades_df['return_pct'].mean()
                std_r = self.trades_df['return_pct'].std()
                metrics['sqn'] = (avg_r / std_r) * np.sqrt(len(self.trades_df)) if std_r > 0 else 0
        
        return metrics
    
    def calculate_risk_metrics(self) -> Dict:
        """Métricas de risco detalhadas"""
        metrics = {}
        
        # Value at Risk (VaR)
        if len(self.trades_df) > 0:
            returns = self.trades_df['return_pct'].values
            metrics['var_95'] = np.percentile(returns, 5)  # 95% VaR
            metrics['var_99'] = np.percentile(returns, 1)  # 99% VaR
            
            # Conditional VaR (CVaR)
            metrics['cvar_95'] = returns[returns <= metrics['var_95']].mean() if len(returns[returns <= metrics['var_95']]) > 0 else 0
            
            # Ulcer Index
            metrics['ulcer_index'] = self._calculate_ulcer_index()
            
            # Maximum consecutive losses
            metrics['max_consecutive_losses'] = self._calculate_max_consecutive_losses()
            
            # Risk of ruin
            if self.results.win_rate > 0 and self.results.average_win > 0:
                metrics['risk_of_ruin'] = self._calculate_risk_of_ruin()
            
            # Omega ratio
            threshold = 0  # Minimum acceptable return
            above_threshold = returns[returns > threshold]
            below_threshold = returns[returns <= threshold]
            
            if len(below_threshold) > 0:
                metrics['omega_ratio'] = above_threshold.sum() / abs(below_threshold.sum())
            
            # Tail ratio
            if len(returns) > 10:
                right_tail = np.percentile(returns, 95)
                left_tail = abs(np.percentile(returns, 5))
                metrics['tail_ratio'] = right_tail / left_tail if left_tail > 0 else 0
        
        return metrics
    
    def analyze_trades(self) -> Dict:
        """Análise detalhada dos trades"""
        if len(self.trades_df) == 0:
            return {}
        
        analysis = {}
        
        # Distribuição de retornos
        analysis['return_distribution'] = {
            'mean': self.trades_df['return_pct'].mean(),
            'std': self.trades_df['return_pct'].std(),
            'skew': self.trades_df['return_pct'].skew(),
            'kurtosis': self.trades_df['return_pct'].kurtosis()
        }
        
        # Análise de duração
        analysis['duration'] = {
            'avg_hours': self.trades_df['duration_hours'].mean(),
            'median_hours': self.trades_df['duration_hours'].median(),
            'longest_hours': self.trades_df['duration_hours'].max(),
            'shortest_hours': self.trades_df['duration_hours'].min()
        }
        
        # Análise MAE/MFE
        analysis['mae_mfe'] = {
            'avg_mae': self.trades_df['mae'].mean(),
            'avg_mfe': self.trades_df['mfe'].mean(),
            'edge_ratio': self.trades_df['mfe'].mean() / self.trades_df['mae'].mean() if self.trades_df['mae'].mean() > 0 else 0
        }
        
        # Eficiência de saída
        winning_trades = self.trades_df[self.trades_df['win']]
        if len(winning_trades) > 0:
            analysis['exit_efficiency'] = {
                'win_efficiency': (winning_trades['pnl'] / winning_trades['mfe']).mean(),
                'loss_efficiency': 1 - (self.trades_df[~self.trades_df['win']]['pnl'].abs() / 
                                       self.trades_df[~self.trades_df['win']]['mae']).mean()
            }
        
        return analysis
    
    def analyze_time_patterns(self) -> Dict:
        """Analisa padrões temporais"""
        if len(self.trades_df) == 0:
            return {}
        
        analysis = {}
        
        # Performance por hora do dia
        self.trades_df['entry_hour'] = pd.to_datetime(self.trades_df['entry_time']).dt.hour
        hourly_performance = self.trades_df.groupby('entry_hour').agg({
            'pnl': ['sum', 'mean', 'count'],
            'win': 'mean'
        })
        
        analysis['hourly_performance'] = hourly_performance.to_dict()
        
        # Performance por dia da semana
        self.trades_df['entry_weekday'] = pd.to_datetime(self.trades_df['entry_time']).dt.dayofweek
        daily_performance = self.trades_df.groupby('entry_weekday').agg({
            'pnl': ['sum', 'mean', 'count'],
            'win': 'mean'
        })
        
        analysis['daily_performance'] = daily_performance.to_dict()
        
        # Análise de períodos
        analysis['best_period'] = self._find_best_period()
        analysis['worst_period'] = self._find_worst_period()
        
        return analysis
    
    def analyze_exits(self) -> Dict:
        """Analisa razões de saída"""
        if len(self.trades_df) == 0:
            return {}
        
        exit_stats = self.trades_df.groupby('exit_reason').agg({
            'pnl': ['sum', 'mean', 'count'],
            'win': 'mean',
            'duration_hours': 'mean'
        })
        
        return {
            'exit_reasons': exit_stats.to_dict(),
            'most_profitable_exit': exit_stats['pnl']['sum'].idxmax() if len(exit_stats) > 0 else None,
            'most_common_exit': exit_stats['pnl']['count'].idxmax() if len(exit_stats) > 0 else None
        }
    
    def monte_carlo_analysis(self, n_simulations: int = 1000) -> Dict:
        """Análise Monte Carlo dos resultados"""
        if len(self.trades_df) == 0:
            return {}
        
        returns = self.trades_df['pnl'].values
        initial_balance = self.results.initial_balance
        
        final_balances = []
        max_drawdowns = []
        
        for _ in range(n_simulations):
            # Embaralhar retornos
            shuffled_returns = np.random.choice(returns, size=len(returns), replace=True)
            
            # Simular equity curve
            equity = [initial_balance]
            peak = initial_balance
            max_dd = 0
            
            for ret in shuffled_returns:
                new_equity = equity[-1] + ret
                equity.append(new_equity)
                
                if new_equity > peak:
                    peak = new_equity
                
                dd = (peak - new_equity) / peak if peak > 0 else 0
                max_dd = max(max_dd, dd)
            
            final_balances.append(equity[-1])
            max_drawdowns.append(max_dd)
        
        return {
            'median_final_balance': np.median(final_balances),
            'percentile_5': np.percentile(final_balances, 5),
            'percentile_95': np.percentile(final_balances, 95),
            'probability_of_profit': sum(b > initial_balance for b in final_balances) / n_simulations,
            'median_max_drawdown': np.median(max_drawdowns),
            'worst_case_drawdown': np.percentile(max_drawdowns, 95)
        }
    
    def generate_recommendations(self) -> List[str]:
        """Gera recomendações baseadas na análise"""
        recommendations = []
        
        # Win rate
        if self.results.win_rate < 0.4:
            recommendations.append("Win rate baixo (<40%). Considere revisar critérios de entrada.")
        
        # Profit factor
        if self.results.profit_factor < 1.2:
            recommendations.append("Profit factor baixo (<1.2). Sistema precisa de melhorias.")
        
        # Drawdown
        if self.results.max_drawdown > 0.2:
            recommendations.append("Drawdown máximo alto (>20%). Implemente melhor gestão de risco.")
        
        # Duração média
        if len(self.trades_df) > 0:
            avg_duration = self.trades_df['duration_hours'].mean()
            if avg_duration < 1:
                recommendations.append("Trades muito curtos (<1h). Pode estar over-trading.")
            elif avg_duration > 24:
                recommendations.append("Trades muito longos (>24h). Considere stops mais apertados.")
        
        # Eficiência de saída
        if hasattr(self, 'exit_efficiency'):
            if self.exit_efficiency < 0.5:
                recommendations.append("Baixa eficiência de saída. Revise estratégia de saída.")
        
        # Consistência
        if hasattr(self, 'monthly_win_rate'):
            if self.monthly_win_rate < 0.6:
                recommendations.append("Baixa consistência mensal. Sistema pode ser instável.")
        
        return recommendations
    
    def plot_analysis(self):
        """Cria visualizações da análise"""
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        
        # 1. Distribuição de retornos
        ax1 = axes[0, 0]
        if len(self.trades_df) > 0:
            self.trades_df['pnl'].hist(bins=50, ax=ax1, alpha=0.7)
            ax1.axvline(x=0, color='red', linestyle='--')
            ax1.set_title('Distribuição de PnL')
            ax1.set_xlabel('PnL ($)')
        
        # 2. Curva de equity detalhada
        ax2 = axes[0, 1]
        if self.results.equity_curve:
            equity_array = np.array(self.results.equity_curve)
            ax2.plot(equity_array)
            ax2.fill_between(range(len(equity_array)), 
                           self.results.initial_balance, 
                           equity_array,
                           where=(equity_array >= self.results.initial_balance),
                           alpha=0.3, color='green', label='Profit')
            ax2.fill_between(range(len(equity_array)), 
                           self.results.initial_balance, 
                           equity_array,
                           where=(equity_array < self.results.initial_balance),
                           alpha=0.3, color='red', label='Loss')
            ax2.set_title('Curva de Equity Detalhada')
            ax2.legend()
        
        # 3. Performance por hora
        ax3 = axes[0, 2]
        if len(self.trades_df) > 0 and 'entry_hour' in self.trades_df.columns:
            hourly_pnl = self.trades_df.groupby('entry_hour')['pnl'].sum()
            hourly_pnl.plot(kind='bar', ax=ax3)
            ax3.set_title('PnL por Hora do Dia')
            ax3.set_xlabel('Hora')
            ax3.set_ylabel('PnL Total ($)')
        
        # 4. MAE vs MFE
        ax4 = axes[1, 0]
        if len(self.trades_df) > 0 and 'mae' in self.trades_df.columns:
            winning = self.trades_df[self.trades_df['win']]
            losing = self.trades_df[~self.trades_df['win']]
            
            ax4.scatter(winning['mae'], winning['mfe'], alpha=0.5, color='green', label='Winners')
            ax4.scatter(losing['mae'], losing['mfe'], alpha=0.5, color='red', label='Losers')
            ax4.set_xlabel('MAE (Maximum Adverse Excursion)')
            ax4.set_ylabel('MFE (Maximum Favorable Excursion)')
            ax4.set_title('MAE vs MFE Analysis')
            ax4.legend()
        
        # 5. Duração vs PnL
        ax5 = axes[1, 1]
        if len(self.trades_df) > 0:
            ax5.scatter(self.trades_df['duration_hours'], self.trades_df['pnl'], alpha=0.5)
            ax5.set_xlabel('Duração (horas)')
            ax5.set_ylabel('PnL ($)')
            ax5.set_title('Duração vs PnL')
            ax5.axhline(y=0, color='black', linestyle='--')
        
        # 6. Rolling performance
        ax6 = axes[1, 2]
        if len(self.trades_df) > 20:
            rolling_returns = pd.Series(self.trades_df['pnl'].values).rolling(20).mean()
            rolling_returns.plot(ax=ax6)
            ax6.set_title('Média Móvel de 20 Trades')
            ax6.set_xlabel('Trade #')
            ax6.set_ylabel('PnL Médio ($)')
            ax6.axhline(y=0, color='black', linestyle='--')
        
        # 7. Monthly returns heatmap
        ax7 = axes[2, 0]
        if len(self.trades_df) > 0:
            monthly_returns = self._calculate_monthly_returns_matrix()
            if not monthly_returns.empty:
                sns.heatmap(monthly_returns, annot=True, fmt='.1f', cmap='RdYlGn', center=0, ax=ax7)
                ax7.set_title('Retornos Mensais (%)')
        
        # 8. Underwater curve
        ax8 = axes[2, 1]
        if self.results.drawdown_curve:
            dd_array = np.array(self.results.drawdown_curve) * 100
            ax8.fill_between(range(len(dd_array)), 0, -dd_array, color='red', alpha=0.3)
            ax8.set_title('Underwater Curve')
            ax8.set_ylabel('Drawdown (%)')
            ax8.set_xlabel('Tempo')
        
        # 9. Exit reasons
        ax9 = axes[2, 2]
        if len(self.trades_df) > 0:
            exit_counts = self.trades_df['exit_reason'].value_counts()
            exit_counts.plot(kind='pie', ax=ax9, autopct='%1.1f%%')
            ax9.set_title('Razões de Saída')
            ax9.set_ylabel('')
        
        plt.tight_layout()
        plt.show()
    
    # Métodos auxiliares privados
    
    def _calculate_annualized_return(self) -> float:
        """Calcula retorno anualizado"""
        if self.results.start_date and self.results.end_date:
            days = (self.results.end_date - self.results.start_date).days
            if days > 0:
                total_return = (self.results.final_balance - self.results.initial_balance) / self.results.initial_balance
                return (1 + total_return) ** (365 / days) - 1
        return 0
    
    def _calculate_profit_factor(self, trades_df: pd.DataFrame) -> float:
        """Calcula profit factor para subset de trades"""
        if len(trades_df) == 0:
            return 0
        
        gross_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
        gross_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())
        
        return gross_profit / gross_loss if gross_loss > 0 else 0
    
    def _calculate_monthly_returns(self) -> Dict[str, float]:
        """Calcula retornos mensais"""
        if len(self.trades_df) == 0:
            return {}
        
        monthly_returns = {}
        trades_df = self.trades_df.copy()
        trades_df['month'] = pd.to_datetime(trades_df['exit_time']).dt.to_period('M')
        
        for month, group in trades_df.groupby('month'):
            monthly_returns[str(month)] = group['pnl'].sum() / self.results.initial_balance * 100
        
        return monthly_returns
    
    def _calculate_monthly_returns_matrix(self) -> pd.DataFrame:
        """Cria matriz de retornos mensais para heatmap"""
        if len(self.trades_df) == 0:
            return pd.DataFrame()
        
        trades_df = self.trades_df.copy()
        trades_df['year'] = pd.to_datetime(trades_df['exit_time']).dt.year
        trades_df['month'] = pd.to_datetime(trades_df['exit_time']).dt.month
        
        monthly_pnl = trades_df.groupby(['year', 'month'])['pnl'].sum()
        monthly_returns = (monthly_pnl / self.results.initial_balance * 100).unstack(fill_value=0)
        
        # Adicionar nomes dos meses
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        monthly_returns.columns = [month_names[m-1] for m in monthly_returns.columns]
        
        return monthly_returns
    
    def _calculate_ulcer_index(self) -> float:
        """Calcula Ulcer Index"""
        if len(self.results.drawdown_curve) == 0:
            return 0
        
        dd_squared = np.array(self.results.drawdown_curve) ** 2
        return np.sqrt(np.mean(dd_squared)) * 100
    
    def _calculate_max_consecutive_losses(self) -> int:
        """Calcula máximo de perdas consecutivas"""
        if len(self.trades_df) == 0:
            return 0
        
        max_consecutive = 0
        current_consecutive = 0
        
        for win in self.trades_df['win']:
            if not win:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        return max_consecutive
    
    def _calculate_risk_of_ruin(self) -> float:
        """Calcula probabilidade de ruína (simplificado)"""
        # Fórmula simplificada para risk of ruin
        # Assume distribuição normal de retornos
        
        win_rate = self.results.win_rate
        avg_win = self.results.average_win
        avg_loss = self.results.average_loss
        
        if win_rate == 1 or avg_loss == 0:
            return 0
        
        if win_rate == 0:
            return 1
        
        # Kelly criterion
        kelly = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
        
        if kelly <= 0:
            return 1
        
        # Aproximação da probabilidade de ruína
        a = avg_loss / (avg_win + avg_loss)
        p = win_rate
        
        if p > a:
            return ((a/p) ** (self.results.initial_balance / avg_loss))
        else:
            return 1
    
    def _find_best_period(self) -> Dict:
        """Encontra melhor período de performance"""
        if len(self.results.equity_curve) < 2:
            return {}
        
        equity = np.array(self.results.equity_curve)
        returns = np.diff(equity)
        
        # Encontrar melhor sequência de 20 trades
        best_sum = -np.inf
        best_start = 0
        
        for i in range(len(returns) - 20):
            current_sum = returns[i:i+20].sum()
            if current_sum > best_sum:
                best_sum = current_sum
                best_start = i
        
        return {
            'start_index': best_start,
            'end_index': best_start + 20,
            'total_profit': best_sum,
            'avg_profit': best_sum / 20
        }
    
    def _find_worst_period(self) -> Dict:
        """Encontra pior período de performance"""
        if len(self.results.equity_curve) < 2:
            return {}
        
        equity = np.array(self.results.equity_curve)
        returns = np.diff(equity)
        
        # Encontrar pior sequência de 20 trades
        worst_sum = np.inf
        worst_start = 0
        
        for i in range(len(returns) - 20):
            current_sum = returns[i:i+20].sum()
            if current_sum < worst_sum:
                worst_sum = current_sum
                worst_start = i
        
        return {
            'start_index': worst_start,
            'end_index': worst_start + 20,
            'total_loss': worst_sum,
            'avg_loss': worst_sum / 20
        }