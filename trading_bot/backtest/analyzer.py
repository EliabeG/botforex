# backtest/analyzer.py
"""Analisador avançado de resultados de backtest para fornecer insights detalhados."""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any # Adicionado Any
from datetime import datetime, timezone # Adicionado timezone
import matplotlib.pyplot as plt
import seaborn as sns # Seaborn já está nos requirements
from scipy import stats # Scipy já está nos requirements

from backtest.engine import BacktestResults, BacktestTrade # Importar estruturas de dados do engine
# Importar PerformanceMetrics do scoring para consistência na estrutura de métricas
from optimization.scoring import PerformanceMetrics, StrategyScorer # StrategyScorer pode ser usado para consistência
from utils.logger import setup_logger
from utils.helpers import calculate_sharpe_ratio, calculate_max_drawdown # Reutilizar helpers se possível

logger = setup_logger("backtest_results_analyzer") # Nome do logger específico

class BacktestAnalyzer:
    """Analisador detalhado de resultados de backtest, com plots e métricas avançadas."""

    def __init__(self, backtest_results_obj: BacktestResults): # Renomeado e tipado
        if not isinstance(backtest_results_obj, BacktestResults):
            raise TypeError("BacktestAnalyzer espera um objeto BacktestResults.")
        self.results: BacktestResults = backtest_results_obj
        self.trades_as_df: pd.DataFrame = self._convert_trades_to_dataframe() # Renomeado

        # Instanciar um Scorer para usar seus métodos de cálculo de métricas, se aplicável
        # Isso ajuda a manter a lógica de cálculo de métricas centralizada.
        self._internal_scorer = StrategyScorer()


    def _convert_trades_to_dataframe(self) -> pd.DataFrame: # Renomeado
        """Converte a lista de objetos BacktestTrade em um DataFrame do Pandas."""
        if not self.results.trades:
            return pd.DataFrame()

        trades_data_list: List[Dict[str, Any]] = [] # Renomeado
        for trade_obj in self.results.trades: # Renomeado
            # Converter dataclass para dict para o DataFrame
            # Se BacktestTrade tiver um método .to_dict(), usar. Senão, vars() ou manual.
            if hasattr(trade_obj, 'to_dict') and callable(trade_obj.to_dict):
                trade_dict = trade_obj.to_dict()
            else: # Fallback manual (assumindo atributos da dataclass BacktestTrade)
                trade_dict = {
                    'id': trade_obj.id, 'strategy': trade_obj.strategy, 'symbol': trade_obj.symbol,
                    'side': trade_obj.side, 'entry_time': trade_obj.entry_time,
                    'exit_time': trade_obj.exit_time, 'entry_price': trade_obj.entry_price,
                    'exit_price': trade_obj.exit_price, 'size': trade_obj.size,
                    'pnl': trade_obj.pnl, 'pnl_pips': trade_obj.pnl_pips,
                    'commission': trade_obj.commission, 'exit_reason': trade_obj.exit_reason,
                    'duration_seconds': trade_obj.duration_seconds,
                    'max_profit_in_trade': trade_obj.max_profit, # Renomeado para clareza
                    'max_loss_in_trade': trade_obj.max_loss,     # Renomeado para clareza
                    # 'metadata': trade_obj.metadata # Metadata pode ser complexo para DataFrame, considerar serializar
                }
            trades_data_list.append(trade_dict)

        df = pd.DataFrame(trades_data_list)
        if df.empty: return df # Retornar se vazio após conversão

        # Conversões e colunas derivadas
        df['entry_time'] = pd.to_datetime(df['entry_time'], utc=True)
        df['exit_time'] = pd.to_datetime(df['exit_time'], utc=True)
        df['duration_hours'] = pd.to_numeric(df['duration_seconds'], errors='coerce').fillna(0) / 3600.0
        df['is_win_trade'] = df['pnl'] > 0 # Renomeado de 'win'
        df['is_loss_trade'] = df['pnl'] < 0
        df['is_be_trade'] = df['pnl'] == 0


        # Retorno percentual (assumindo que PnL é na moeda da conta)
        # Para retorno percentual por trade, idealmente teríamos o capital alocado ou o balanço no momento do trade.
        # Usar initial_balance como proxy pode não ser ideal para trades tardios.
        # Se self.results.equity_curve estiver disponível e alinhado com os trades, podemos usá-lo.
        # Por simplicidade, usar o initial_balance aqui, mas com ressalvas.
        if self.results.initial_balance > 0:
            df['return_on_initial_balance_pct'] = (df['pnl'] / self.results.initial_balance) * 100 # Renomeado
        else:
            df['return_on_initial_balance_pct'] = 0.0

        # MAE (Maximum Adverse Excursion) e MFE (Maximum Favorable Excursion)
        # O BacktestTrade original tem max_loss e max_profit.
        # MAE é o quão contra a posição o preço foi.
        # MFE é o quão a favor da posição o preço foi.
        # Assumindo que max_loss_in_trade é o PnL mais negativo durante o trade (já é < 0 ou 0)
        # e max_profit_in_trade é o PnL mais positivo durante o trade (já é > 0 ou 0)
        df['mae_value'] = df['max_loss_in_trade'].abs() # MAE como valor absoluto da "pior perda flutuante"
        df['mfe_value'] = df['max_profit_in_trade']     # MFE como "melhor lucro flutuante"

        return df


    def generate_full_analytical_report(self) -> Dict[str, Any]: # Renomeado
        """Gera um relatório analítico completo com múltiplas seções."""
        if self.trades_as_df.empty and self.results.total_trades == 0:
            logger.warning("Nenhum trade para analisar. Relatório estará vazio.")
            return {"error": "Nenhum trade nos resultados do backtest."}

        # Usar o StrategyScorer para calcular métricas de forma consistente
        # calculate_all_performance_metrics espera uma lista de dicts de trades
        trades_for_scorer = []
        if not self.trades_as_df.empty:
            # Converter colunas de timestamp para string ISO para serialização JSON se necessário para o scorer
            # Mas o scorer pode lidar com datetime se for interno.
            # Por enquanto, assumindo que o scorer aceita o formato do DataFrame.
            # O ideal é que BacktestResults já contenha um objeto PerformanceMetrics.
            # Aqui, vamos recalcular usando o Scorer para ter todas as métricas.
            trades_for_scorer = self.trades_as_df.to_dict('records')


        # Determinar duração total para anualização
        total_duration_days_val = None
        if self.results.start_date and self.results.end_date:
            total_duration_days_val = (self.results.end_date - self.results.start_date).days
            if total_duration_days_val <=0: total_duration_days_val = 1 # Mínimo 1 dia para evitar divisão por zero


        # Este objeto PerformanceMetrics será o principal local das métricas
        calculated_metrics: PerformanceMetrics = self._internal_scorer.calculate_all_performance_metrics(
            trades_list=trades_for_scorer,
            initial_balance=self.results.initial_balance,
            total_duration_days=total_duration_days_val,
            risk_free_rate_annual=getattr(CONFIG, 'RISK_FREE_RATE_ANNUAL', 0.02) # Usar de CONFIG
        )
        # Atualizar o objeto results principal com as métricas calculadas se ele não as tiver
        # ou se quisermos sobrescrever com os cálculos do Scorer.
        # Ex: self.results.sharpe_ratio = calculated_metrics.sharpe_ratio (se a estrutura for compatível)


        report_dict: Dict[str, Any] = { # Renomeado
            'overall_summary_stats': self.get_summary_statistics(calculated_metrics), # Passar métricas
            'detailed_performance_metrics': calculated_metrics.to_dict(), # Usar o objeto completo
            'risk_analysis_metrics': self.calculate_risk_analysis_metrics(calculated_metrics), # Passar métricas
            'trade_by_trade_analysis': self.analyze_individual_trades(), # Renomeado
            'temporal_pattern_analysis': self.analyze_temporal_patterns(), # Renomeado
            'exit_reason_analysis': self.analyze_trade_exits(), # Renomeado
            'monte_carlo_simulation': self.run_monte_carlo_simulation(), # Renomeado
            # 'generated_recommendations': self.generate_improvement_recommendations(calculated_metrics) # Renomeado
        }
        return report_dict


    def get_summary_statistics(self, perf_metrics: PerformanceMetrics) -> Dict[str, Any]: # Renomeado
        """Retorna estatísticas resumidas chave da performance."""
        summary: Dict[str, Any] = { # Adicionada tipagem
            'period_start_date': self.results.start_date.isoformat() if self.results.start_date else "N/A",
            'period_end_date': self.results.end_date.isoformat() if self.results.end_date else "N/A",
            'initial_balance': self.results.initial_balance,
            'final_balance': perf_metrics.final_balance, # Usar de perf_metrics
            'total_net_return_pct': ((perf_metrics.final_balance - self.results.initial_balance) / self.results.initial_balance * 100) if self.results.initial_balance > 0 else 0.0,
            'annualized_return_pct': self._calculate_annualized_return(perf_metrics.final_balance, self.results.initial_balance, self.results.start_date, self.results.end_date) * 100,
            'total_trades_executed': perf_metrics.total_trades, # Renomeado
            'overall_win_rate_pct': perf_metrics.win_rate * 100, # Renomeado
            'average_trade_pnl': perf_metrics.total_pnl / perf_metrics.total_trades if perf_metrics.total_trades > 0 else 0.0,
            'best_trade_pnl': perf_metrics.largest_win,
            'worst_trade_pnl': -perf_metrics.largest_loss, # Já é absoluto, então - para mostrar como perda
            'avg_win_to_avg_loss_ratio': perf_metrics.average_win / perf_metrics.average_loss if perf_metrics.average_loss > 0 else np.inf if perf_metrics.average_win > 0 else 0.0,
            'total_commissions_paid': perf_metrics.total_commission, # Renomeado
            'net_profit_after_comm': perf_metrics.net_pnl, # Renomeado
            'sharpe_ratio_annualized': perf_metrics.sharpe_ratio, # Renomeado
            'max_drawdown_observed_pct': perf_metrics.max_drawdown_pct * 100, # Renomeado
            'profit_factor': perf_metrics.profit_factor
        }
        return summary

    # calculate_performance_metrics foi substituído pelo uso de StrategyScorer
    # calculate_risk_metrics foi substituído pelo uso de StrategyScorer

    def calculate_risk_analysis_metrics(self, perf_metrics: PerformanceMetrics) -> Dict[str, Any]: # Renomeado
        """Calcula métricas de risco adicionais ou específicas não cobertas pelo Scorer geral."""
        risk_metrics_dict: Dict[str, Any] = {} # Renomeado
        if self.trades_as_df.empty: return risk_metrics_dict

        # Value at Risk (VaR) - percentual do saldo inicial
        # Usar retornos percentuais por trade em relação ao saldo inicial para VaR
        trade_returns_pct = self.trades_as_df['return_on_initial_balance_pct'].dropna().values
        if len(trade_returns_pct) > 10: # Suficiente para percentis
            risk_metrics_dict['var_95_pct_per_trade'] = np.percentile(trade_returns_pct, 5)
            risk_metrics_dict['var_99_pct_per_trade'] = np.percentile(trade_returns_pct, 1)
            # Conditional VaR (CVaR) or Expected Shortfall
            risk_metrics_dict['cvar_95_pct_per_trade'] = trade_returns_pct[trade_returns_pct <= risk_metrics_dict['var_95_pct_per_trade']].mean()
        
        # Ulcer Index (calculado sobre a curva de equity)
        risk_metrics_dict['ulcer_index_val'] = self._calculate_ulcer_index_from_equity(perf_metrics.equity_curve) # Renomeado

        # Máximo de Perdas Consecutivas (já em PerformanceMetrics)
        risk_metrics_dict['max_consecutive_losses_count'] = perf_metrics.max_consecutive_losses # Renomeado

        # Omega Ratio (já em PerformanceMetrics se o Scorer o calcular)
        # Se não, calcular aqui. Exige retornos periódicos (ex: diários)
        # if perf_metrics.daily_returns:
        #     daily_rets = np.array(perf_metrics.daily_returns)
        #     threshold_ret = 0.0 # Retorno alvo
        #     gain_sum = np.sum(daily_rets[daily_rets > threshold_ret] - threshold_ret)
        #     loss_sum = abs(np.sum(daily_rets[daily_rets <= threshold_ret] - threshold_ret))
        #     risk_metrics_dict['omega_ratio_daily_thresh_0'] = gain_sum / (loss_sum + 1e-9)

        return risk_metrics_dict

    def analyze_individual_trades(self) -> Dict[str, Any]: # Renomeado
        """Análise detalhada dos trades individuais (distribuições, MAE/MFE)."""
        if self.trades_as_df.empty: return {}
        analysis_dict: Dict[str, Any] = {} # Renomeado

        # Distribuição de PnL por Trade
        analysis_dict['pnl_per_trade_distribution'] = { # Renomeado
            'mean': self.trades_as_df['pnl'].mean(),
            'median': self.trades_as_df['pnl'].median(),
            'std_dev': self.trades_as_df['pnl'].std(),
            'skewness': self.trades_as_df['pnl'].skew(),
            'kurtosis': self.trades_as_df['pnl'].kurtosis(),
            'percentiles': self.trades_as_df['pnl'].quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).to_dict()
        }

        # Análise de Duração de Trades
        analysis_dict['trade_duration_stats_hours'] = { # Renomeado
            'mean': self.trades_as_df['duration_hours'].mean(),
            'median': self.trades_as_df['duration_hours'].median(),
            'std_dev': self.trades_as_df['duration_hours'].std(),
            'min': self.trades_as_df['duration_hours'].min(),
            'max': self.trades_as_df['duration_hours'].max()
        }

        # Análise MAE/MFE (Excursão Adversa/Favorável Máxima)
        if 'mae_value' in self.trades_as_df.columns and 'mfe_value' in self.trades_as_df.columns:
            analysis_dict['mae_mfe_analysis'] = {
                'avg_mae_currency': self.trades_as_df['mae_value'].mean(), # Renomeado
                'avg_mfe_currency': self.trades_as_df['mfe_value'].mean(), # Renomeado
                # Edge Ratio: Média MFE / Média MAE (para trades vencedores)
                'avg_mfe_of_winners': self.trades_as_df[self.trades_as_df['is_win_trade']]['mfe_value'].mean(),
                'avg_mae_of_losers': self.trades_as_df[self.trades_as_df['is_loss_trade']]['mae_value'].mean(),
                # Eficiência de Saída (PnL / MFE para ganhos, (StopPrice - EntryPrice) / MAE para perdas)
            }
            # Calcular eficiência de saída
            wins_df = self.trades_as_df[self.trades_as_df['is_win_trade']]
            losses_df = self.trades_as_df[self.trades_as_df['is_loss_trade']]
            if not wins_df.empty and 'mfe_value' in wins_df.columns and (wins_df['mfe_value'] > 0).any() :
                analysis_dict['mae_mfe_analysis']['win_exit_efficiency_avg'] = (wins_df['pnl'] / (wins_df['mfe_value'] + 1e-9)).mean()
            if not losses_df.empty and 'mae_value' in losses_df.columns and (losses_df['mae_value'] > 0).any():
                 # Para perdas, PnL é negativo. SL - Entry (para compra) ou Entry - SL (venda) é o risco planejado.
                 # MAE é o quanto foi contra. Se PnL final == MAE, saiu no pior ponto.
                 # Se PnL final < MAE (em valor abs), saiu melhor que o pior ponto.
                 # Eficiência de Stop: 1 - (abs(PnL) / MAE)
                 analysis_dict['mae_mfe_analysis']['loss_stop_efficiency_avg'] = (1 - (losses_df['pnl'].abs() / (losses_df['mae_value'] + 1e-9))).mean()

        return analysis_dict

    def analyze_temporal_patterns(self) -> Dict[str, Any]: # Renomeado
        """Analisa padrões de performance por hora, dia da semana, mês."""
        if self.trades_as_df.empty or 'entry_time' not in self.trades_as_df.columns:
            return {}
        analysis_t: Dict[str, Any] = {} # Renomeado

        # Garantir que entry_time seja datetime
        self.trades_as_df['entry_time'] = pd.to_datetime(self.trades_as_df['entry_time'], utc=True)

        # Performance por Hora do Dia (UTC)
        self.trades_as_df['entry_hour_of_day'] = self.trades_as_df['entry_time'].dt.hour # Renomeado
        hourly_perf_df = self.trades_as_df.groupby('entry_hour_of_day').agg( # Renomeado
            total_pnl_sum=('pnl', 'sum'),
            avg_pnl_mean=('pnl', 'mean'),
            trade_count=('pnl', 'count'),
            win_rate_mean=('is_win_trade', 'mean')
        ).reset_index()
        analysis_t['hourly_performance_utc'] = hourly_perf_df.to_dict('records')

        # Performance por Dia da Semana (0=Seg, 6=Dom)
        self.trades_as_df['entry_day_of_week'] = self.trades_as_df['entry_time'].dt.dayofweek # Renomeado
        daily_perf_df = self.trades_as_df.groupby('entry_day_of_week').agg( # Renomeado
            total_pnl_sum=('pnl', 'sum'),
            avg_pnl_mean=('pnl', 'mean'),
            trade_count=('pnl', 'count'),
            win_rate_mean=('is_win_trade', 'mean')
        ).reset_index()
        analysis_t['weekday_performance'] = daily_perf_df.to_dict('records')
        
        # Performance por Mês
        self.trades_as_df['entry_month_year'] = self.trades_as_df['entry_time'].dt.to_period('M').astype(str) # Renomeado
        monthly_perf_df = self.trades_as_df.groupby('entry_month_year').agg( # Renomeado
            total_pnl_sum=('pnl', 'sum'),
            trade_count=('pnl', 'count'),
            win_rate_mean=('is_win_trade', 'mean')
        ).reset_index()
        analysis_t['monthly_performance'] = monthly_perf_df.to_dict('records')

        return analysis_t


    def analyze_trade_exits(self) -> Dict[str, Any]: # Renomeado
        """Analisa performance por razão de saída dos trades."""
        if self.trades_as_df.empty or 'exit_reason' not in self.trades_as_df.columns:
            return {'status': 'Coluna "exit_reason" não encontrada ou sem trades.'}
        
        exit_reason_stats_df = self.trades_as_df.groupby('exit_reason').agg( # Renomeado
            total_pnl_sum=('pnl', 'sum'),
            avg_pnl_mean=('pnl', 'mean'),
            trade_count=('pnl', 'count'),
            win_rate_mean=('is_win_trade', 'mean'),
            avg_duration_hours_mean=('duration_hours', 'mean') # Renomeado
        ).sort_values(by=('trade_count'), ascending=False).reset_index() # Ordenar por contagem

        most_profit_exit = exit_reason_stats_df.loc[exit_reason_stats_df[('pnl','sum')].idxmax()] if not exit_reason_stats_df.empty else None
        most_common_exit_reason = exit_reason_stats_df.loc[exit_reason_stats_df[('pnl','count')].idxmax()] if not exit_reason_stats_df.empty else None # Renomeado

        return {
            'exit_reason_summary_stats': exit_reason_stats_df.to_dict('records'), # Renomeado
            'most_profitable_exit_type': most_profit_exit.to_dict() if most_profit_exit is not None else None, # Renomeado
            'most_frequent_exit_type': most_common_exit_reason.to_dict() if most_common_exit_reason is not None else None # Renomeado
        }


    def run_monte_carlo_simulation(self, num_simulations: int = 1000, # Renomeado
                                  num_trades_to_sample: Optional[int] = None) -> Dict[str, Any]:
        """Executa simulação de Monte Carlo sobre os retornos dos trades."""
        if self.trades_as_df.empty or 'pnl' not in self.trades_as_df.columns:
            return {'status': 'Trades insuficientes ou PnL ausente para Monte Carlo.'}

        trade_pnls_mc = self.trades_as_df['pnl'].dropna().values # Renomeado
        if len(trade_pnls_mc) < 10: # Mínimo de trades para simulação ter algum sentido
            return {'status': 'Trades insuficientes para Monte Carlo significativo.'}

        initial_bal_mc = self.results.initial_balance # Renomeado
        num_trades_in_sim = num_trades_to_sample if num_trades_to_sample else len(trade_pnls_mc) # Renomeado

        sim_final_balances: List[float] = [] # Renomeado
        sim_max_drawdowns_pct: List[float] = [] # Renomeado

        for _ in range(num_simulations):
            # Amostrar com reposição da série de PnLs de trades
            simulated_trade_pnls = np.random.choice(trade_pnls_mc, size=num_trades_in_sim, replace=True) # Renomeado

            # Simular curva de equity para esta sequência de trades
            current_equity_mc = initial_bal_mc # Renomeado
            equity_path_mc: List[float] = [initial_bal_mc] # Renomeado
            
            for pnl_trade in simulated_trade_pnls: # Renomeado
                current_equity_mc += pnl_trade
                equity_path_mc.append(current_equity_mc)
            
            sim_final_balances.append(current_equity_mc)
            max_dd_sim_pct, _, _ = calculate_max_drawdown(equity_path_mc) # Usar helper
            sim_max_drawdowns_pct.append(max_dd_sim_pct)

        return {
            'num_simulations_run': num_simulations,
            'num_trades_per_simulation': num_trades_in_sim,
            'median_final_balance_sim': np.median(sim_final_balances),
            'percentile_5_final_balance_sim': np.percentile(sim_final_balances, 5),
            'percentile_95_final_balance_sim': np.percentile(sim_final_balances, 95),
            'probability_of_profit_sim_pct': (np.sum(np.array(sim_final_balances) > initial_bal_mc) / num_simulations) * 100,
            'median_max_drawdown_sim_pct': np.median(sim_max_drawdowns_pct) * 100,
            'percentile_95_max_drawdown_sim_pct': np.percentile(sim_max_drawdowns_pct, 95) * 100 # "Pior" drawdown esperado em 95% dos casos
        }

    # generate_improvement_recommendations foi removido, pois a versão original estava vazia.
    # Requereria lógica complexa para dar recomendações úteis.

    def plot_backtest_analysis_charts(self, save_to_dir: Optional[str] = None): # Renomeado
        """Cria e opcionalmente salva um conjunto de visualizações da análise do backtest."""
        if self.trades_as_df.empty and self.results.total_trades == 0:
            logger.info("Nenhum trade para plotar na análise.")
            return

        num_charts = 9 # Número de subplots
        fig, axes = plt.subplots(3, 3, figsize=(22, 18)) # Ajustado tamanho
        fig.suptitle(f"Análise de Backtest: {self.results.trades[0].strategy if self.results.trades else 'Estratégia Desconhecida'}", fontsize=16) # Adicionado título geral

        # 1. Distribuição de PnL por Trade
        if 'pnl' in self.trades_as_df.columns:
            sns.histplot(self.trades_as_df['pnl'], bins=50, kde=True, ax=axes[0, 0], color='skyblue')
            axes[0, 0].axvline(x=0, color='red', linestyle='--')
            axes[0, 0].set_title('Distribuição de PnL por Trade')
            axes[0, 0].set_xlabel('PnL ($)')
            axes[0, 0].set_ylabel('Frequência')

        # 2. Curva de Equity Detalhada
        if self.results.equity_curve:
            equity_arr = np.array(self.results.equity_curve) # Renomeado
            axes[0, 1].plot(equity_arr, label='Equity', color='blue')
            axes[0, 1].fill_between(range(len(equity_arr)), self.results.initial_balance, equity_arr,
                                   where=(equity_arr >= self.results.initial_balance),
                                   alpha=0.3, color='green', interpolate=True)
            axes[0, 1].fill_between(range(len(equity_arr)), self.results.initial_balance, equity_arr,
                                   where=(equity_arr < self.results.initial_balance),
                                   alpha=0.3, color='red', interpolate=True)
            axes[0, 1].set_title('Curva de Equity')
            axes[0, 1].set_xlabel('Número do Trade (ou Período)')
            axes[0, 1].set_ylabel('Equity ($)')
            axes[0, 1].legend()

        # 3. Performance por Hora do Dia (PnL Total)
        if 'entry_hour_of_day' in self.trades_as_df.columns:
            hourly_pnl_sum = self.trades_as_df.groupby('entry_hour_of_day')['pnl'].sum() # Renomeado
            if not hourly_pnl_sum.empty:
                hourly_pnl_sum.plot(kind='bar', ax=axes[0, 2], color='dodgerblue')
                axes[0, 2].set_title('PnL Total por Hora de Entrada (UTC)')
                axes[0, 2].set_xlabel('Hora do Dia (UTC)')
                axes[0, 2].set_ylabel('PnL Total ($)')

        # 4. MAE vs MFE Scatter Plot
        if 'mae_value' in self.trades_as_df.columns and 'mfe_value' in self.trades_as_df.columns:
            wins_df_plot = self.trades_as_df[self.trades_as_df['is_win_trade']] # Renomeado
            losses_df_plot = self.trades_as_df[self.trades_as_df['is_loss_trade']] # Renomeado
            if not wins_df_plot.empty:
                axes[1, 0].scatter(wins_df_plot['mae_value'], wins_df_plot['mfe_value'], alpha=0.5, color='green', label='Ganhos', s=15)
            if not losses_df_plot.empty:
                axes[1, 0].scatter(losses_df_plot['mae_value'], losses_df_plot['mfe_value'], alpha=0.5, color='red', label='Perdas', s=15)
            axes[1, 0].set_title('MAE vs MFE por Trade')
            axes[1, 0].set_xlabel('Excursão Adversa Máxima (MAE $)')
            axes[1, 0].set_ylabel('Excursão Favorável Máxima (MFE $)')
            axes[1, 0].legend()
            axes[1, 0].grid(True, linestyle=':', alpha=0.7)


        # 5. Duração do Trade vs PnL Scatter Plot
        if 'duration_hours' in self.trades_as_df.columns and 'pnl' in self.trades_as_df.columns:
            axes[1, 1].scatter(self.trades_as_df['duration_hours'], self.trades_as_df['pnl'],
                               c=self.trades_as_df['is_win_trade'].map({True: 'green', False: 'red'}),
                               alpha=0.5, s=15)
            axes[1, 1].axhline(y=0, color='black', linestyle='--')
            axes[1, 1].set_title('Duração do Trade vs. PnL')
            axes[1, 1].set_xlabel('Duração (Horas)')
            axes[1, 1].set_ylabel('PnL ($)')
            axes[1, 1].set_xscale('log') # Escala log para duração se houver grande variação


        # 6. PnL Acumulado (Rolling Sum de PnL)
        if 'pnl' in self.trades_as_df.columns:
            self.trades_as_df['cumulative_pnl'] = self.trades_as_df['pnl'].cumsum() + self.results.initial_balance
            axes[1, 2].plot(self.trades_as_df.index, self.trades_as_df['cumulative_pnl'], label='PnL Acumulado', color='purple')
            axes[1, 2].set_title('PnL Acumulado por Trade')
            axes[1, 2].set_xlabel('Número do Trade')
            axes[1, 2].set_ylabel('Balanço ($)')
            axes[1, 2].legend()

        # 7. Heatmap de Retornos Mensais (se dados suficientes)
        if 'entry_time' in self.trades_as_df.columns:
            monthly_returns_matrix = self._calculate_monthly_returns_as_matrix() # Renomeado
            if not monthly_returns_matrix.empty:
                sns.heatmap(monthly_returns_matrix, annot=True, fmt=".1f", cmap="RdYlGn", center=0, ax=axes[2, 0], linewidths=.5)
                axes[2, 0].set_title('Retornos Mensais (%)')
                axes[2, 0].set_xlabel('Mês')
                axes[2, 0].set_ylabel('Ano')


        # 8. Curva Underwater (Drawdown ao longo do tempo/trades)
        if self.results.drawdown_curve: # Esta é a curva de DD percentual
            dd_percent_curve = -1 * np.array(self.results.drawdown_curve) * 100 # Inverter para plotar abaixo de zero
            axes[2, 1].fill_between(range(len(dd_percent_curve)), dd_percent_curve, 0, color='red', alpha=0.4, interpolate=True)
            axes[2, 1].set_title('Curva Underwater (Drawdown %)')
            axes[2, 1].set_xlabel('Número do Trade (ou Período)')
            axes[2, 1].set_ylabel('Drawdown (%)')
            axes[2, 1].grid(True, linestyle=':', alpha=0.7)


        # 9. Distribuição de Razões de Saída (Pie Chart)
        if 'exit_reason' in self.trades_as_df.columns:
            exit_reason_counts = self.trades_as_df['exit_reason'].value_counts() # Renomeado
            if not exit_reason_counts.empty:
                axes[2, 2].pie(exit_reason_counts, labels=exit_reason_counts.index, autopct='%1.1f%%', startangle=90,
                               wedgeprops=dict(width=0.4)) # Donut chart
                axes[2, 2].set_title('Distribuição por Razão de Saída')
                axes[2, 2].axis('equal') # Para círculo


        plt.tight_layout(rect=[0, 0, 1, 0.96]) # Ajustar para o suptitle
        
        if save_to_dir:
            fig_path = Path(save_to_dir) / f"backtest_analysis_charts_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.png"
            fig_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                plt.savefig(str(fig_path))
                logger.info(f"Gráficos da análise de backtest salvos em: {fig_path}")
            except Exception as e_save_fig: # Renomeado
                logger.error(f"Erro ao salvar gráficos da análise: {e_save_fig}")

        else:
            # plt.show() # Comentado para ambientes sem GUI. Descomente se rodar localmente.
            logger.info("Geração de plots concluída. Use save_to_dir para salvar ou descomente plt.show() para visualização interativa.")
        plt.close(fig) # Fechar figura para liberar memória


    # --- Métodos Auxiliares Privados (renomeados e alguns ajustados) ---

    def _calculate_annualized_return(self, final_bal: float, initial_bal: float, # Renomeado
                                    start_dt: Optional[datetime], end_dt: Optional[datetime], # Renomeado
                                    trading_days_year: int = 252) -> float:
        """Calcula o retorno anualizado."""
        if initial_bal == 0: return 0.0
        if not start_dt or not end_dt: return 0.0

        total_days_in_period = (end_dt - start_dt).days # Renomeado
        if total_days_in_period <= 0: return 0.0 # Evitar divisão por zero ou raiz negativa

        total_return_pct = (final_bal - initial_bal) / initial_bal
        # ((1 + R_total) ^ (dias_neg_ano / dias_totais_periodo)) - 1
        # Usar trading_days_year aqui pode ser mais para mercados de ações.
        # Para Forex (24/5), (365 / total_days_in_period) ou (dias_uteis_no_ano / dias_uteis_no_periodo)
        # Se total_days_in_period for < 1 ano, isso projeta. Se > 1 ano, anualiza.
        annualization_factor = 365.25 / total_days_in_period # Usar 365.25 para dias calendário
        
        annualized_ret = ( (1.0 + total_return_pct) ** annualization_factor ) - 1.0
        return annualized_ret


    def _calculate_profit_factor_for_subset(self, trades_subset_df: pd.DataFrame) -> float: # Renomeado
        """Calcula o profit factor para um subconjunto de trades (DataFrame)."""
        if trades_subset_df.empty: return 0.0
        gross_profit_sub = trades_subset_df[trades_subset_df['pnl'] > 0]['pnl'].sum() # Renomeado
        gross_loss_sub = abs(trades_subset_df[trades_subset_df['pnl'] < 0]['pnl'].sum()) # Renomeado
        return gross_profit_sub / (gross_loss_sub + 1e-9) # Evitar divisão por zero


    def _calculate_monthly_returns_as_series(self) -> pd.Series: # Renomeado
        """Calcula os retornos percentuais mensais como uma pd.Series."""
        if self.trades_as_df.empty or 'exit_time' not in self.trades_as_df.columns:
            return pd.Series(dtype=float)

        # Garantir que 'exit_time' é datetime
        df_copy = self.trades_as_df.copy() # Trabalhar com cópia
        df_copy['exit_time'] = pd.to_datetime(df_copy['exit_time'], utc=True)
        
        # Calcular PnL mensal
        monthly_pnl_series = df_copy.set_index('exit_time')['pnl'].resample('M').sum() # 'M' para fim do mês # Renomeado
        
        # Converter PnL mensal para retorno percentual em relação ao saldo inicial do backtest
        if self.results.initial_balance > 0:
            monthly_returns_pct_series = (monthly_pnl_series / self.results.initial_balance) * 100 # Renomeado
            return monthly_returns_pct_series
        return pd.Series(dtype=float)


    def _calculate_monthly_returns_as_matrix(self) -> pd.DataFrame: # Renomeado
        """Cria uma matriz (DataFrame) de retornos mensais (Ano x Mês)."""
        if self.trades_as_df.empty or 'exit_time' not in self.trades_as_df.columns:
            return pd.DataFrame()
        
        df_copy_matrix = self.trades_as_df.copy() # Renomeado
        df_copy_matrix['exit_time'] = pd.to_datetime(df_copy_matrix['exit_time'], utc=True)
        df_copy_matrix['year'] = df_copy_matrix['exit_time'].dt.year
        df_copy_matrix['month'] = df_copy_matrix['exit_time'].dt.month

        monthly_pnl_matrix = df_copy_matrix.groupby(['year', 'month'])['pnl'].sum() # Renomeado
        if self.results.initial_balance > 0:
            monthly_returns_matrix_pct = (monthly_pnl_matrix / self.results.initial_balance * 100).unstack(fill_value=0.0) # Renomeado
            # Renomear colunas de mês para nomes (Jan, Feb, etc.)
            month_names_map = {i: pd.Timestamp(2000, i, 1).strftime('%b') for i in range(1, 13)} # Renomeado
            monthly_returns_matrix_pct.rename(columns=month_names_map, inplace=True)
            return monthly_returns_matrix_pct
        return pd.DataFrame()


    def _calculate_ulcer_index_from_equity(self, equity_curve_vals: List[float]) -> float: # Renomeado
        """Calcula o Ulcer Index a partir da curva de equity."""
        if not equity_curve_vals or len(equity_curve_vals) < 2:
            return 0.0
        
        # Calcular drawdowns percentuais em cada ponto
        equity_arr_ui = np.array(equity_curve_vals, dtype=float) # Renomeado
        running_max_equity = np.maximum.accumulate(equity_arr_ui) # Renomeado
        # Drawdown como (Pico - Atual) / Pico. Se Pico=0, DD=0.
        drawdowns_pct_ui = (running_max_equity - equity_arr_ui) / (running_max_equity + 1e-9) # Renomeado
        drawdowns_pct_ui[running_max_equity <= 1e-9] = 0 # Lidar com picos zero ou negativos

        # Ulcer Index = sqrt(sum(DD_pct[i]^2) / N)
        ulcer_idx = np.sqrt(np.mean(drawdowns_pct_ui ** 2)) * 100 # Multiplicar por 100 para percentual
        return ulcer_idx if not np.isnan(ulcer_idx) else 0.0


    # _calculate_max_consecutive_losses foi removido pois PerformanceMetrics já o calcula.
    # _calculate_risk_of_ruin foi removido, pois a versão simplificada pode não ser muito útil.
    # _find_best_period e _find_worst_period foram removidos por complexidade e podem ser adicionados depois.