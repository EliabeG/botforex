# optimization/scoring.py
"""Sistema de scoring para avaliação de estratégias e resultados de backtest"""
import numpy as np
import pandas as pd # Adicionado pandas para possível uso futuro em métricas mais complexas
from typing import Dict, List, Optional, Tuple, Any # Adicionado Any
from dataclasses import dataclass, asdict # Adicionado asdict
import statistics # Não usado no código original, mas pode ser útil
from datetime import datetime, timedelta # Não usado diretamente, mas comum em performance

from utils.logger import setup_logger
from utils.helpers import calculate_sharpe_ratio, calculate_max_drawdown # Usando helpers

logger = setup_logger("strategy_scorer") # Renomeado logger

@dataclass
class PerformanceMetrics:
    """Métricas de performance de uma estratégia"""
    # Métricas básicas de contagem
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    breakeven_trades: int = 0 # Adicionado
    win_rate: float = 0.0 # (Wins / (Wins + Losses))
    
    # Métricas de PnL
    total_pnl: float = 0.0 # PnL bruto total
    gross_profit: float = 0.0 # Soma de todos os PnLs positivos
    gross_loss: float = 0.0 # Soma de todos os PnLs negativos (valor absoluto)
    net_pnl: float = 0.0 # PnL líquido (total_pnl - total_commission)
    total_commission: float = 0.0 # Adicionado

    average_win: float = 0.0 # PnL médio dos trades vencedores
    average_loss: float = 0.0 # PnL médio (absoluto) dos trades perdedores
    largest_win: float = 0.0
    largest_loss: float = 0.0 # Valor absoluto da maior perda
    
    # Métricas de Risco/Retorno e Ratios
    expectancy: float = 0.0 # (WinRate * AvgWin) - (LossRate * AvgLoss)
    profit_factor: float = 0.0 # GrossProfit / GrossLoss
    sharpe_ratio: float = 0.0 # Anualizado, se retornos forem diários/mensais
    sortino_ratio: float = 0.0 # Anualizado
    calmar_ratio: float = 0.0 # Retorno Anualizado / MaxDrawdown
    
    # Métricas de Drawdown
    max_drawdown_abs: float = 0.0 # Drawdown máximo em valor monetário
    max_drawdown_pct: float = 0.0 # Drawdown máximo em percentual da equity
    max_drawdown_duration_days: int = 0 # Duração do maior drawdown em dias
    average_drawdown_pct: float = 0.0 # Drawdown médio em percentual
    recovery_factor: float = 0.0 # NetPnL / MaxDrawdownAbs
    
    # Métricas de Duração e Exposição
    avg_trade_duration_seconds: float = 0.0 # Renomeado para clareza
    avg_bars_in_trade: int = 0 # Se aplicável (para dados de barra)
    market_exposure_time_pct: float = 0.0 # Percentual do tempo total que esteve no mercado
    
    # Métricas de Consistência e Sequência
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    # 'consecutive_wins' e 'consecutive_losses' foram removidos pois representam o estado atual, não uma métrica de resumo.

    # Outras métricas avançadas (placeholders, podem ser complexas de calcular)
    # risk_adjusted_return: float = 0.0 # Ex: Sharpe, ou PnL/VaR
    # information_ratio: float = 0.0 # Comparado a um benchmark
    # omega_ratio: float = 0.0

    # Adicionar campos do backtest original se necessários para compatibilidade
    initial_balance: float = 0.0
    final_balance: float = 0.0


    def to_dict(self) -> Dict[str, Any]: # Usar Any
        """Converte para dicionário usando dataclasses.asdict."""
        return asdict(self)

class StrategyScorer:
    """Calculador de score para estratégias e resultados de backtest."""

    def __init__(self):
        # Pesos para cálculo do score (devem somar 1.0 se for média ponderada direta)
        self.weights: Dict[str, float] = {
            'win_rate': 0.10,
            'expectancy': 0.20,
            'sharpe_ratio': 0.25,
            'profit_factor': 0.15,
            'max_drawdown_pct': 0.15, # Peso para o inverso do drawdown
            'consistency_score': 0.10, # Ex: % meses positivos ou baixa std dev de retornos
            'recovery_factor': 0.05
        }
        # Normalizar pesos para garantir que somem 1
        total_weight = sum(self.weights.values())
        if total_weight > 0 and abs(total_weight - 1.0) > 1e-6 : # Checar se não soma 1
             logger.warning(f"Pesos do Scorer não somam 1 (soma: {total_weight}). Normalizando...")
             self.weights = {k: v / total_weight for k, v in self.weights.items()}


        # Thresholds para normalização de cada métrica (min, target, max)
        # Target é o valor que daria um score "bom" (ex: 0.8 na normalização)
        self.thresholds: Dict[str, Dict[str, float]] = {
            'win_rate': {'min': 0.30, 'target': 0.55, 'max': 0.75},
            'expectancy': {'min': 0.0, 'target': 0.002, 'max': 0.01}, # Assumindo expectancy como % do capital por trade
            'sharpe_ratio': {'min': 0.0, 'target': 1.0, 'max': 2.5},
            'profit_factor': {'min': 1.0, 'target': 1.5, 'max': 3.0},
            'max_drawdown_pct': {'min': 0.02, 'target': 0.10, 'max': 0.25}, # Menor é melhor
            'consistency_score': {'min': 0.3, 'target': 0.6, 'max': 0.9}, # Ex: % de períodos lucrativos
            'recovery_factor': {'min': 0.5, 'target': 2.0, 'max': 5.0}
        }

    def calculate_final_score(self, perf_metrics: PerformanceMetrics) -> float: # Renomeado, recebe PerformanceMetrics
        """
        Calcula score geral da estratégia baseado no objeto PerformanceMetrics.
        Retorna score entre 0 e 100 (ou 0-1, ajustar normalização).
        """
        if not isinstance(perf_metrics, PerformanceMetrics):
            logger.error("Tipo inválido para perf_metrics em calculate_final_score. Esperado PerformanceMetrics.")
            return 0.0

        # Verificar se tem trades suficientes
        if perf_metrics.total_trades < getattr(CONFIG, 'SCORING_MIN_TRADES', 20): # Usar de CONFIG
            logger.info(f"Trades insuficientes ({perf_metrics.total_trades}) para scoring robusto. Score = 0.")
            return 0.0

        try:
            # Normalizar cada métrica principal para uma escala (ex: 0-1)
            norm_win_rate = self._normalize_metric(perf_metrics.win_rate, self.thresholds['win_rate'])
            # Expectancy pode ser em $, pips, ou %. Precisa ser consistente.
            # Se for em $, normalizar pelo capital inicial ou ATR para ter uma base.
            # Assumindo que expectancy em PerformanceMetrics já é um valor comparável (ex: R-múltiplo ou % da conta)
            norm_expectancy = self._normalize_metric(perf_metrics.expectancy, self.thresholds['expectancy'])
            norm_sharpe = self._normalize_metric(perf_metrics.sharpe_ratio, self.thresholds['sharpe_ratio'])
            norm_profit_factor = self._normalize_metric(perf_metrics.profit_factor, self.thresholds['profit_factor'])
            # Para drawdown, menor é melhor, então (1 - normalized_drawdown)
            norm_max_drawdown = 1.0 - self._normalize_metric(perf_metrics.max_drawdown_pct, self.thresholds['max_drawdown_pct'])

            # Consistência pode ser calculada a partir de retornos diários/mensais
            # Aqui, vamos simular um valor de consistência ou calculá-lo se 'daily_returns' estiverem em perf_metrics
            consistency_val = self._calculate_consistency_metric(getattr(perf_metrics, 'daily_returns', []))
            norm_consistency = self._normalize_metric(consistency_val, self.thresholds['consistency_score'])
            
            norm_recovery_factor = self._normalize_metric(perf_metrics.recovery_factor, self.thresholds['recovery_factor'])


            # Calcular score ponderado
            weighted_score = (
                norm_win_rate * self.weights['win_rate'] +
                norm_expectancy * self.weights['expectancy'] +
                norm_sharpe * self.weights['sharpe_ratio'] +
                norm_profit_factor * self.weights['profit_factor'] +
                norm_max_drawdown * self.weights['max_drawdown_pct'] +
                norm_consistency * self.weights['consistency_score'] +
                norm_recovery_factor * self.weights['recovery_factor']
            )

            final_score = np.clip(weighted_score * 100, 0, 100) # Score de 0 a 100

            logger.debug(f"Componentes normalizados do score: WR={norm_win_rate:.2f}, EXP={norm_expectancy:.2f}, SHP={norm_sharpe:.2f}, PF={norm_profit_factor:.2f}, DD={norm_max_drawdown:.2f}, CONS={norm_consistency:.2f}, REC={norm_recovery_factor:.2f}")
            logger.debug(f"Score final calculado: {final_score:.2f}")
            return final_score

        except Exception as e:
            logger.exception("Erro ao calcular score final da estratégia:")
            return 0.0


    def calculate_all_performance_metrics(self, # Renomeado de calculate_metrics
                                         trades_list: List[Dict[str, Any]], # Lista de dicts de trades
                                         initial_balance: float,
                                         total_duration_days: Optional[float] = None, # Duração total do período de backtest/live
                                         risk_free_rate_annual: float = 0.0) -> PerformanceMetrics:
        """
        Calcula todas as métricas de performance a partir de uma lista de trades.
        """
        metrics = PerformanceMetrics(initial_balance=initial_balance)
        if not trades_list:
            metrics.final_balance = initial_balance
            return metrics

        # Converter lista de dicts para DataFrame para facilidade de cálculo
        trades_df = pd.DataFrame(trades_list)
        if 'pnl' not in trades_df.columns or trades_df['pnl'].isnull().all():
            logger.warning("Coluna 'pnl' ausente ou todos NaNs nos trades. Não é possível calcular métricas.")
            metrics.final_balance = initial_balance
            return metrics

        # Garantir que 'pnl' seja numérico
        trades_df['pnl'] = pd.to_numeric(trades_df['pnl'], errors='coerce').fillna(0.0)


        # Métricas básicas de contagem
        metrics.total_trades = len(trades_df)
        metrics.winning_trades = len(trades_df[trades_df['pnl'] > 0])
        metrics.losing_trades = len(trades_df[trades_df['pnl'] < 0])
        metrics.breakeven_trades = len(trades_df[trades_df['pnl'] == 0])
        # Win rate sobre trades não-breakeven
        non_be_trades = metrics.winning_trades + metrics.losing_trades
        metrics.win_rate = (metrics.winning_trades / non_be_trades) if non_be_trades > 0 else 0.0

        # Métricas de PnL
        metrics.total_pnl = trades_df['pnl'].sum()
        metrics.gross_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
        metrics.gross_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum()) # Absoluto
        metrics.total_commission = trades_df.get('commission', pd.Series(0.0)).sum() # Usar .get para segurança
        metrics.net_pnl = metrics.total_pnl - metrics.total_commission # PnL líquido

        wins_pnl = trades_df[trades_df['pnl'] > 0]['pnl']
        losses_pnl = trades_df[trades_df['pnl'] < 0]['pnl'] # Manter negativo para avg_loss real

        metrics.average_win = wins_pnl.mean() if not wins_pnl.empty else 0.0
        metrics.average_loss = abs(losses_pnl.mean()) if not losses_pnl.empty else 0.0 # Média da perda absoluta
        metrics.largest_win = wins_pnl.max() if not wins_pnl.empty else 0.0
        metrics.largest_loss = abs(losses_pnl.min()) if not losses_pnl.empty else 0.0


        # Ratios
        metrics.expectancy = (metrics.win_rate * metrics.average_win) - ((1.0 - metrics.win_rate) * metrics.average_loss)
        metrics.profit_factor = metrics.gross_profit / metrics.gross_loss if metrics.gross_loss > 0 else np.inf if metrics.gross_profit > 0 else 0.0


        # Equity Curve e Retornos Diários/Periódicos
        # A coluna 'close_time' é necessária para calcular retornos diários/periódicos
        equity_curve_values: List[float] = [initial_balance] # Renomeado
        current_equity = initial_balance
        for pnl_val in trades_df['pnl']: # Renomeado pnl para pnl_val
            current_equity += pnl_val
            equity_curve_values.append(current_equity)
        metrics.equity_curve = equity_curve_values
        metrics.final_balance = equity_curve_values[-1]

        # Calcular retornos para Sharpe/Sortino
        # Se 'close_time' disponível, calcular retornos diários/semanais. Senão, por trade.
        # Aqui, usaremos retornos por trade normalizados pelo capital no início do trade (aproximação)
        returns_per_trade_pct: List[float] = []
        temp_balance = initial_balance
        for pnl_val in trades_df['pnl']:
            if temp_balance > 0: # Evitar divisão por zero
                returns_per_trade_pct.append(pnl_val / temp_balance)
            else: # Se o balanço for a zero, não se pode calcular % de retorno
                returns_per_trade_pct.append(0.0)
            temp_balance += pnl_val
        metrics.daily_returns = returns_per_trade_pct # Armazenando retornos por trade como 'daily_returns'

        # Assumir 252 períodos de trading por ano (ajustar se os retornos não forem diários)
        # Se total_duration_days for fornecido, pode-se calcular um fator de anualização mais preciso
        periods_per_year = 252
        if total_duration_days and total_duration_days > 0:
            if len(returns_per_trade_pct) > 1: # Precisa de mais de um retorno
                # Fator de anualização: (num_trades / total_dias_negociacao) * dias_negociacao_ano
                # Se total_duration_days é o total de dias do backtest:
                trades_per_day = len(returns_per_trade_pct) / total_duration_days
                periods_per_year = trades_per_day * 252 # Estimativa de trades por ano

        metrics.sharpe_ratio = calculate_sharpe_ratio(returns_per_trade_pct, risk_free_rate=risk_free_rate_annual, periods_per_year=periods_per_year)
        metrics.sortino_ratio = self._calculate_sortino_ratio(returns_per_trade_pct, target_return_pct_per_period=0.0, periods_per_year=periods_per_year)


        # Métricas de Drawdown
        metrics.max_drawdown_pct, peak_idx, trough_idx = calculate_max_drawdown(equity_curve_values)
        metrics.max_drawdown_abs = metrics.max_drawdown_pct * equity_curve_values[peak_idx] if peak_idx < len(equity_curve_values) else metrics.max_drawdown_pct * initial_balance
        # Duração do drawdown (aproximada em número de trades)
        metrics.max_drawdown_duration_days = trough_idx - peak_idx # Isso é em número de trades/pontos na equity curve
        
        # metrics.average_drawdown_pct = self._calculate_average_drawdown_pct(equity_curve_values) # Renomeado

        metrics.recovery_factor = metrics.net_pnl / metrics.max_drawdown_abs if metrics.max_drawdown_abs > 0 else np.inf if metrics.net_pnl > 0 else 0.0
        metrics.calmar_ratio = self._calculate_calmar_ratio(returns_per_trade_pct, metrics.max_drawdown_pct, periods_per_year)


        # Duração e Exposição
        if 'duration' in trades_df.columns: # 'duration' deve ser em segundos
             metrics.avg_trade_duration_seconds = trades_df['duration'].mean()

        # max_consecutive_wins/losses
        metrics = self._calculate_consecutive_trade_stats(trades_df['pnl'].tolist(), metrics) # Renomeado

        return metrics


    def _normalize_metric(self, value: float, threshold_config: Dict[str, float]) -> float: # Renomeado
        """Normaliza métrica para score entre 0 e 1, usando config de thresholds."""
        min_val = threshold_config['min']
        target_val = threshold_config['target']
        max_val = threshold_config['max']

        if not (min_val <= target_val <= max_val): # Validação dos thresholds
            logger.error(f"Thresholds inválidos para normalização: {threshold_config}. Min deve ser <= Target <= Max.")
            return 0.0 # Ou levantar erro

        if value <= min_val: return 0.0
        if value >= max_val: return 1.0

        # Escala de min_val para target_val (mapeia para 0.0 - 0.8)
        if value <= target_val:
            if (target_val - min_val) == 0: return 0.8 # Se min == target, e value é target
            return 0.8 * (value - min_val) / (target_val - min_val)
        # Escala de target_val para max_val (mapeia para 0.8 - 1.0)
        else:
            if (max_val - target_val) == 0: return 1.0 # Se target == max, e value é max
            return 0.8 + 0.2 * (value - target_val) / (max_val - target_val)


    def _calculate_sortino_ratio(self, returns_pct_per_period: List[float], # Renomeado
                                target_return_pct_per_period: float = 0.0,
                                periods_per_year: int = 252) -> float:
        """Calcula Sortino ratio. `returns_pct_per_period` são retornos percentuais por período."""
        if not returns_pct_per_period or len(returns_pct_per_period) < 2:
            return 0.0

        returns_arr = np.array(returns_pct_per_period) # Renomeado
        excess_returns_arr = returns_arr - target_return_pct_per_period # Renomeado
        
        # Filtrar retornos abaixo do alvo (downside deviation)
        downside_returns_arr = excess_returns_arr[excess_returns_arr < 0] # Renomeado

        if len(downside_returns_arr) == 0: # Sem retornos negativos (ou abaixo do alvo)
            return np.inf if np.mean(excess_returns_arr) > 0 else 0.0 # Infinito se média de retornos é positiva

        downside_std_dev = np.std(downside_returns_arr) # Renomeado
        if downside_std_dev == 0: # Sem variação no downside (todos os retornos negativos iguais)
            return 0.0 # Ou Inf se avg_excess > 0 e downside_std_dev = 0 (raro)

        avg_excess_return_period = np.mean(excess_returns_arr) # Renomeado
        sortino_period = avg_excess_return_period / downside_std_dev
        
        return sortino_period * np.sqrt(periods_per_year) # Anualizar


    def _calculate_calmar_ratio(self, returns_pct_per_period: List[float], # Renomeado
                               max_drawdown_pct_val: float, # Renomeado
                               periods_per_year: int = 252) -> float:
        """Calcula Calmar ratio. `max_drawdown_pct_val` deve ser positivo (ex: 0.2 para 20% DD)."""
        if not returns_pct_per_period or max_drawdown_pct_val == 0.0:
            return 0.0

        # Retorno médio anualizado
        # (1 + R_total)^(periodos_no_ano / total_periodos) - 1
        # Se returns_pct_per_period são retornos por trade, e não temos a duração total em períodos,
        # uma aproximação do retorno anualizado é mais complexa.
        # Se assumirmos que os retornos são diários e `periods_per_year` é 252:
        avg_period_return = np.mean(returns_pct_per_period)
        annualized_return = avg_period_return * periods_per_year # Aproximação simples
        # Ou, (1 + avg_period_return)**periods_per_year - 1 para retornos compostos
        # Se 'returns_pct_per_period' são retornos de um período de backtest completo:
        # total_return_compound = np.prod([1 + r for r in returns_pct_per_period]) - 1
        # num_years = len(returns_pct_per_period) / periods_per_year
        # annualized_return = (1 + total_return_compound)**(1/num_years) - 1 if num_years > 0 else total_return_compound


        return annualized_return / max_drawdown_pct_val if max_drawdown_pct_val > 0 else 0.0


    # _build_equity_curve foi removido pois a curva de equity é construída em calculate_all_performance_metrics
    # _calculate_average_drawdown_pct foi removido, pode ser re-adicionado se necessário. MaxDD é mais comum.

    def _calculate_consecutive_trade_stats(self, pnl_list: List[float], # Renomeado
                                   metrics_obj: PerformanceMetrics) -> PerformanceMetrics: # Renomeado
        """Calcula estatísticas de trades consecutivos (wins/losses)."""
        if not pnl_list:
            return metrics_obj

        current_wins_streak = 0 # Renomeado
        current_losses_streak = 0 # Renomeado

        for pnl_val_streak in pnl_list: # Renomeado
            if pnl_val_streak > 0:
                current_wins_streak += 1
                current_losses_streak = 0 # Resetar perdas
            elif pnl_val_streak < 0:
                current_losses_streak += 1
                current_wins_streak = 0 # Resetar ganhos
            # Breakeven trades não quebram nem continuam streaks, mas podem ser tratados como um reset.
            # else: # PnL == 0
            #     current_wins_streak = 0
            #     current_losses_streak = 0


            metrics_obj.max_consecutive_wins = max(metrics_obj.max_consecutive_wins, current_wins_streak)
            metrics_obj.max_consecutive_losses = max(metrics_obj.max_consecutive_losses, current_losses_streak)

        # Os atributos 'consecutive_wins' e 'consecutive_losses' na dataclass original
        # pareciam ser para o *estado atual* da sequência, não o máximo.
        # Se for para o estado atual, esta função não os atualizaria corretamente após o loop.
        # Removidos da dataclass por ora, pois max_consecutive é mais comum em resumos.

        return metrics_obj

    def _calculate_consistency_metric(self, periodic_returns: List[float], min_periods: int = 3) -> float:
        """Calcula uma métrica de consistência (ex: % de períodos lucrativos)."""
        if not periodic_returns or len(periodic_returns) < min_periods:
            return 0.0 # Não confiável com poucos períodos
        
        positive_periods = sum(1 for r_val in periodic_returns if r_val > 0) # Renomeado r para r_val
        return positive_periods / len(periodic_returns)


    def compare_multiple_strategies(self, # Renomeado de compare_strategies
                                 strategy_performances: Dict[str, PerformanceMetrics] # Recebe dict de PerformanceMetrics
                                 ) -> List[Tuple[str, float]]:
        """
        Compara múltiplas estratégias com base em seus scores.
        Retorna lista ordenada de (nome_da_estrategia, score_final).
        """
        strategy_scores_list: List[Tuple[str, float]] = [] # Renomeado

        for strategy_name, perf_metrics in strategy_performances.items(): # Renomeado name, performance
            final_score = self.calculate_final_score(perf_metrics) # Renomeado
            strategy_scores_list.append((strategy_name, final_score))

        # Ordenar por score decrescente
        strategy_scores_list.sort(key=lambda x_item: x_item[1], reverse=True) # Renomeado x para x_item
        return strategy_scores_list


    def update_scoring_weights(self, new_weights: Dict[str, float]): # Renomeado
        """Atualiza pesos para o cálculo do score e normaliza para somar 1."""
        if not new_weights:
            logger.warning("Tentativa de atualizar pesos do scorer com dicionário vazio.")
            return

        total_new_weight = sum(new_weights.values()) # Renomeado

        if total_new_weight <= 0: # Checar se a soma é positiva
            logger.error("Soma dos novos pesos do scorer é zero ou negativa. Pesos não atualizados.")
            return

        self.weights = {k_weight: v_weight / total_new_weight for k_weight, v_weight in new_weights.items()} # Renomeado k,v
        logger.info(f"Pesos do scorer atualizados e normalizados: {self.weights}")

    def update_scoring_thresholds(self, metric_name: str, new_threshold_config: Dict[str, float]): # Novo método
        """Atualiza os thresholds para uma métrica específica."""
        if metric_name not in self.thresholds:
            logger.warning(f"Métrica '{metric_name}' não encontrada nos thresholds do scorer.")
            return
        if not all(k_thr in new_threshold_config for k_thr in ['min', 'target', 'max']): # Renomeado k para k_thr
            logger.error(f"Configuração de threshold inválida para '{metric_name}'. Deve conter 'min', 'target', 'max'.")
            return
        if not (new_threshold_config['min'] <= new_threshold_config['target'] <= new_threshold_config['max']):
            logger.error(f"Valores de threshold inválidos para '{metric_name}'. Min deve ser <= Target <= Max.")
            return

        self.thresholds[metric_name] = new_threshold_config
        logger.info(f"Thresholds para a métrica '{metric_name}' atualizados para: {new_threshold_config}")