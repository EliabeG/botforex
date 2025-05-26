# risk/position_sizing.py
"""Sistema de dimensionamento de posições com múltiplos métodos e ajustes dinâmicos."""
import numpy as np
from typing import Dict, List, Optional, Tuple, Any # Adicionado Any
from dataclasses import dataclass, field # Adicionado field
from datetime import datetime, timedelta # Não usado diretamente, mas comum em contextos de trading

# Usar CONFIG para alguns limites globais se RiskConfig não os tiver
from config.settings import CONFIG
from config.risk_config import RISK_LIMITS, RISK_PARAMS, POSITION_SIZING # POSITION_SIZING já é uma instância
from utils.logger import setup_logger
from utils.helpers import calculate_pip_value # calculate_position_size não deve ser importado aqui, pois esta classe o implementa

logger = setup_logger("position_sizer") # Renomeado

@dataclass
class PositionSizeResult:
    """Resultado do cálculo de tamanho de posição."""
    lot_size: float
    risk_amount_currency: float # Renomeado de risk_amount
    risk_percent_of_balance: float # Renomeado de risk_percent
    pip_value_per_lot_at_calc_size: float # Renomeado de pip_value e mais específico
    stop_distance_pips: float
    potential_loss_currency: float # Renomeado de potential_loss
    margin_required_currency: float # Renomeado de margin_required
    method_used: str
    adjustments_applied: List[str] = field(default_factory=list) # Default para lista vazia
    error_message: Optional[str] = None # Para registrar erros

class PositionSizer:
    """Calculador de tamanho de posição com múltiplos métodos e ajustes dinâmicos."""

    def __init__(self, method: str = POSITION_SIZING.FIXED_RISK, initial_balance_for_equity_calc: float = 10000.0): # Adicionado initial_balance
        if method not in POSITION_SIZING.METHOD_CONFIG:
            logger.warning(f"Método de dimensionamento '{method}' não encontrado em METHOD_CONFIG. Usando FIXED_RISK como padrão.")
            self.method = POSITION_SIZING.FIXED_RISK
        else:
            self.method = method
        
        self.method_config: Dict[str, Any] = POSITION_SIZING.METHOD_CONFIG.get(self.method, {}) # Adicionada tipagem
        
        # Histórico de trades e curva de equity para métodos que dependem disso
        self.trade_history: List[Dict[str, Any]] = [] # Lista de dicionários representando trades (pnl, etc.)
        # Equity curve deve ser inicializada com o saldo inicial se usada desde o começo
        self.equity_curve: List[float] = [initial_balance_for_equity_calc] # Inicializar com saldo
        self.current_win_streak: int = 0 # Renomeado de current_streak
        self.current_loss_streak: int = 0 # Adicionado


    def calculate_position_size(self,
                              account_balance: float,
                              entry_price: float, # Preço de entrada do sinal
                              stop_loss_price: float, # Preço do stop loss do sinal # Renomeado de stop_loss
                              symbol: str = CONFIG.SYMBOL, # Usar de CONFIG
                              leverage: int = CONFIG.LEVERAGE, # Usar de CONFIG
                              market_conditions: Optional[Dict[str, Any]] = None
                              ) -> PositionSizeResult:
        """
        Calcula tamanho da posição usando método configurado e aplicando ajustes.
        """
        # Garantir que account_balance é positivo
        if account_balance <= 0:
            logger.error("Saldo da conta é zero ou negativo. Não é possível calcular o tamanho da posição.")
            return PositionSizeResult(0.01, 0,0,0,0,0,0, "error_zero_balance", ["Saldo zero"], "Saldo da conta inválido")

        try:
            stop_dist_pips = abs(entry_price - stop_loss_price) / (0.0001 if "JPY" not in symbol.upper() else 0.01) # Renomeado
            stop_dist_pips = round(stop_dist_pips, 1) # Arredondar pips

            min_stop_pips = RISK_PARAMS.MIN_STOP_DISTANCE_PIPS
            if stop_dist_pips < min_stop_pips:
                logger.warning(f"Distância do Stop ({stop_dist_pips:.1f} pips) é menor que o mínimo ({min_stop_pips} pips). Ajustando para o mínimo.")
                stop_dist_pips = min_stop_pips
            if stop_dist_pips == 0: # Evitar divisão por zero se, mesmo após ajuste, for zero
                logger.error("Distância do Stop é zero. Não é possível calcular o tamanho da posição.")
                return PositionSizeResult(0.01,0,0,0,0,0,0, "error_zero_stop", ["Stop zero"], "Distância do stop é zero")

            # Calcular tamanho base usando o método selecionado
            base_lot_size: float # Adicionada tipagem
            if self.method == POSITION_SIZING.FIXED_LOT:
                base_lot_size = self._calculate_fixed_lot_size() # Renomeado
            elif self.method == POSITION_SIZING.FIXED_RISK:
                base_lot_size = self._calculate_fixed_risk_size(account_balance, stop_dist_pips, symbol) # Renomeado
            elif self.method == POSITION_SIZING.KELLY_CRITERION:
                base_lot_size = self._calculate_kelly_criterion_size(account_balance, stop_dist_pips, symbol) # Renomeado
            elif self.method == POSITION_SIZING.VOLATILITY_BASED:
                base_lot_size = self._calculate_volatility_adjusted_size(account_balance, stop_dist_pips, symbol, market_conditions) # Renomeado
            elif self.method == POSITION_SIZING.EQUITY_CURVE:
                base_lot_size = self._calculate_equity_curve_tracking_size(account_balance, stop_dist_pips, symbol) # Renomeado
            else:
                logger.warning(f"Método de dimensionamento '{self.method}' desconhecido. Usando FIXED_RISK.")
                base_lot_size = self._calculate_fixed_risk_size(account_balance, stop_dist_pips, symbol)

            # Aplicar ajustes dinâmicos
            adjustments_log: List[str] = [] # Renomeado
            adjusted_lot_size = base_lot_size # Renomeado

            perf_multiplier = self._get_performance_adjustment_factor() # Renomeado
            if abs(perf_multiplier - 1.0) > 1e-3: # Se houver ajuste significativo
                adjusted_lot_size *= perf_multiplier
                adjustments_log.append(f"Ajuste Performance: {perf_multiplier:.2f}x")

            if market_conditions:
                market_mult = self._get_market_condition_adjustment_factor(market_conditions) # Renomeado
                if abs(market_mult - 1.0) > 1e-3:
                    adjusted_lot_size *= market_mult
                    adjustments_log.append(f"Ajuste Cond. Mercado: {market_mult:.2f}x")

            # Aplicar limites finais (mínimo, máximo, margem)
            final_lot_size = self._apply_final_lot_limits(adjusted_lot_size, account_balance, leverage, entry_price, symbol) # Renomeado
            if abs(final_lot_size - adjusted_lot_size) > 1e-3 : # Se o limite alterou o tamanho
                 adjustments_log.append(f"Ajuste Limites Finais: de {adjusted_lot_size:.2f} para {final_lot_size:.2f}")


            # Arredondar para precisão do broker (geralmente 0.01 para Forex)
            final_lot_size = round(max(0.01, final_lot_size), 2) # Garantir mínimo de 0.01

            # Calcular valores finais com o tamanho de lote final
            # Usar 1.0 para calcular o valor do pip por lote padrão, depois multiplicar pelo final_lot_size
            pip_value_per_std_lot = calculate_pip_value(symbol, 1.0) # Valor do pip para 1 lote padrão
            risked_amount_curr = stop_dist_pips * pip_value_per_std_lot * final_lot_size # Renomeado
            risk_pct_bal = (risked_amount_curr / account_balance) * 100 if account_balance > 0 else 0.0 # Renomeado
            
            # Margin required: (LotSize * ContractSize * EntryPrice) / Leverage
            # ContractSize é geralmente 100,000 para Forex
            contract_size_val = getattr(CONFIG, 'CONTRACT_SIZE', 100000) # Renomeado
            margin_req_curr = (final_lot_size * contract_size_val * entry_price) / leverage if leverage > 0 else float('inf') # Renomeado

            result = PositionSizeResult(
                lot_size=final_lot_size,
                risk_amount_currency=risked_amount_curr,
                risk_percent_of_balance=risk_pct_bal,
                pip_value_per_lot_at_calc_size=pip_value_per_std_lot * final_lot_size, # Pip value para o tamanho da posição
                stop_distance_pips=stop_dist_pips,
                potential_loss_currency=risked_amount_curr, # Igual ao risk_amount_currency
                margin_required_currency=margin_req_curr,
                method_used=self.method,
                adjustments_applied=adjustments_log
            )
            logger.info(f"Cálculo de Tamanho de Posição: {result}")
            return result

        except Exception as e:
            logger.exception("Erro crítico ao calcular tamanho da posição:")
            # Fallback para tamanho mínimo em caso de erro
            stop_dist_pips_fallback = abs(entry_price - stop_loss_price) / (0.0001 if "JPY" not in symbol.upper() else 0.01)
            return PositionSizeResult(
                lot_size=0.01,
                risk_amount_currency=0, risk_percent_of_balance=0,
                pip_value_per_lot_at_calc_size=calculate_pip_value(symbol, 0.01),
                stop_distance_pips=round(stop_dist_pips_fallback,1), potential_loss_currency=0,
                margin_required_currency=0, method_used="error_fallback",
                adjustments_applied=["Erro no cálculo, usando mínimo."],
                error_message=str(e)
            )

    def _calculate_fixed_lot_size(self) -> float: # Renomeado
        return float(self.method_config.get('lot_size', 0.01))

    def _calculate_fixed_risk_size(self, balance: float, stop_pips_val: float, symbol: str) -> float: # Renomeado
        risk_pct = float(self.method_config.get('risk_percent', RISK_LIMITS.DEFAULT_RISK_PER_TRADE)) # Usar de RISK_LIMITS
        risk_amt = balance * risk_pct # Renomeado
        pip_val_std_lot = calculate_pip_value(symbol, 1.0) # Renomeado

        if stop_pips_val > 0 and pip_val_std_lot > 0:
            return risk_amt / (stop_pips_val * pip_val_std_lot)
        return 0.01 # Fallback

    def _calculate_kelly_criterion_size(self, balance: float, stop_pips_val: float, symbol: str) -> float: # Renomeado
        lookback = int(self.method_config.get('lookback_trades', 100))
        min_trades = int(self.method_config.get('min_trades_for_kelly', 20))
        
        # Usar cópia para evitar modificar original se for mutável
        recent_trades_list = list(self.trade_history[-lookback:]) if self.trade_history else [] # Renomeado

        if len(recent_trades_list) < min_trades:
            logger.debug(f"Trades insuficientes ({len(recent_trades_list)}/{min_trades}) para Kelly Criterion. Usando Fixed Risk.")
            return self._calculate_fixed_risk_size(balance, stop_pips_val, symbol)

        wins_pnl_list = [float(trade.get('pnl', 0.0)) for trade in recent_trades_list if float(trade.get('pnl', 0.0)) > 0] # Renomeado
        losses_pnl_list = [abs(float(trade.get('pnl', 0.0))) for trade in recent_trades_list if float(trade.get('pnl', 0.0)) < 0] # Renomeado

        if not wins_pnl_list or not losses_pnl_list: # Precisa de ganhos e perdas para calcular b
            logger.debug("Não há trades vencedores ou perdedores suficientes no histórico recente para Kelly. Usando Fixed Risk.")
            return self._calculate_fixed_risk_size(balance, stop_pips_val, symbol)

        win_rt = len(wins_pnl_list) / len(recent_trades_list) # Renomeado
        avg_win_amt = np.mean(wins_pnl_list) # Renomeado
        avg_loss_amt = np.mean(losses_pnl_list) # Renomeado

        if avg_loss_amt == 0: # Evitar divisão por zero
             logger.debug("Média de perda é zero para Kelly. Usando Fixed Risk.")
             return self._calculate_fixed_risk_size(balance, stop_pips_val, symbol)

        b_ratio = avg_win_amt / avg_loss_amt # Renomeado
        q_val = 1.0 - win_rt # Renomeado

        kelly_frac = (win_rt * b_ratio - q_val) / b_ratio if b_ratio > 0 else 0.0 # Renomeado

        # Aplicar fração conservadora e limite máximo de risco percentual
        conservative_frac = float(self.method_config.get('kelly_fraction', 0.25)) # Renomeado
        final_kelly_risk_pct = kelly_frac * conservative_frac # Renomeado
        
        max_risk_pct_kelly = float(self.method_config.get('max_kelly_risk_percent', RISK_LIMITS.MAX_RISK_PER_TRADE)) # Renomeado
        final_kelly_risk_pct = min(max(final_kelly_risk_pct, 0.0), max_risk_pct_kelly) # Garantir que está entre 0 e max_risk

        logger.debug(f"Kelly: W={win_rt:.2f}, R(b)={b_ratio:.2f}, Full_K={kelly_frac:.3f}, Adj_K_Risk%={final_kelly_risk_pct:.3%}")

        risk_amt_kelly = balance * final_kelly_risk_pct # Renomeado
        pip_val_std_lot_kelly = calculate_pip_value(symbol, 1.0) # Renomeado

        if stop_pips_val > 0 and pip_val_std_lot_kelly > 0:
            return risk_amt_kelly / (stop_pips_val * pip_val_std_lot_kelly)
        return 0.01


    def _calculate_volatility_adjusted_size(self, balance: float, stop_pips_val: float, # Renomeado
                                   symbol: str, market_conditions: Optional[Dict[str,Any]]) -> float:
        base_risk_pct = float(self.method_config.get('base_risk_percent', RISK_LIMITS.DEFAULT_RISK_PER_TRADE)) # Renomeado

        volatility_mult = 1.0 # Renomeado
        if market_conditions and 'volatility' in market_conditions:
            current_vol = market_conditions['volatility'] # Assumindo que 'volatility' é um valor normalizado ou ATR%
            # Exemplo: RISK_PARAMS.HIGH_VOLATILITY_THRESHOLD pode ser 0.0015 para ATR%
            # Ajustar multiplicador inversamente à volatilidade
            if current_vol > RISK_PARAMS.HIGH_VOLATILITY_THRESHOLD * 1.5: # Muito alta
                volatility_mult = float(RISK_ADJUSTMENTS.VOLATILITY_ADJUSTMENTS.get('extreme', 0.5))
            elif current_vol > RISK_PARAMS.HIGH_VOLATILITY_THRESHOLD: # Alta
                volatility_mult = float(RISK_ADJUSTMENTS.VOLATILITY_ADJUSTMENTS.get('high', 0.75))
            elif current_vol < RISK_PARAMS.HIGH_VOLATILITY_THRESHOLD * 0.5: # Baixa
                volatility_mult = float(RISK_ADJUSTMENTS.VOLATILITY_ADJUSTMENTS.get('low', 1.2))
            # else: normal, multiplicador = 1.0

        adjusted_risk_pct_vol = base_risk_pct * volatility_mult # Renomeado

        risk_amt_vol = balance * adjusted_risk_pct_vol # Renomeado
        pip_val_std_lot_vol = calculate_pip_value(symbol, 1.0) # Renomeado

        if stop_pips_val > 0 and pip_val_std_lot_vol > 0:
            return risk_amt_vol / (stop_pips_val * pip_val_std_lot_vol)
        return 0.01


    def _calculate_equity_curve_tracking_size(self, balance: float, stop_pips_val: float, symbol: str) -> float: # Renomeado
        base_risk_pct_eq = float(self.method_config.get('base_risk_percent', RISK_LIMITS.DEFAULT_RISK_PER_TRADE)) # Renomeado

        if len(self.equity_curve) < int(self.method_config.get('equity_ma_period', 20)):
            logger.debug("Equity curve muito curta para dimensionamento baseado nela. Usando Fixed Risk.")
            return self._calculate_fixed_risk_size(balance, stop_pips_val, symbol)

        ma_period_eq = int(self.method_config.get('equity_ma_period', 20)) # Renomeado
        recent_equity_vals = self.equity_curve[-ma_period_eq:] # Renomeado
        equity_ma_val = np.mean(recent_equity_vals) # Renomeado
        current_equity_val = self.equity_curve[-1] # Renomeado

        multiplier_eq: float # Adicionada tipagem
        if current_equity_val > equity_ma_val: # Ganhando
            multiplier_eq = float(self.method_config.get('increase_factor_on_winning', 1.1))
        elif current_equity_val < equity_ma_val: # Perdendo
            multiplier_eq = float(self.method_config.get('decrease_factor_on_losing', 0.9))
        else: # Exatamente na média
            multiplier_eq = 1.0
        
        adjusted_risk_pct_eq = base_risk_pct_eq * multiplier_eq # Renomeado

        risk_amt_eq = balance * adjusted_risk_pct_eq # Renomeado
        pip_val_std_lot_eq = calculate_pip_value(symbol, 1.0) # Renomeado

        if stop_pips_val > 0 and pip_val_std_lot_eq > 0:
            return risk_amt_eq / (stop_pips_val * pip_val_std_lot_eq)
        return 0.01


    def _get_performance_adjustment_factor(self) -> float: # Renomeado
        """Obtém fator de ajuste baseado em streaks de performance recente."""
        # Usa self.current_win_streak e self.current_loss_streak que são atualizados por update_trade_history
        if self.current_win_streak >= 5:
            return float(RISK_ADJUSTMENTS.PERFORMANCE_ADJUSTMENTS.get('winning_streak_5', 1.2))
        elif self.current_win_streak >= 3:
            return float(RISK_ADJUSTMENTS.PERFORMANCE_ADJUSTMENTS.get('winning_streak_3', 1.1))
        elif self.current_loss_streak >= 3: # Perdas são negativas
            return float(RISK_ADJUSTMENTS.PERFORMANCE_ADJUSTMENTS.get('losing_streak_3', 0.5))
        elif self.current_loss_streak >= 2:
            return float(RISK_ADJUSTMENTS.PERFORMANCE_ADJUSTMENTS.get('losing_streak_2', 0.75))
        return 1.0


    def _get_market_condition_adjustment_factor(self, conditions: Dict[str, Any]) -> float: # Renomeado
        """Obtém fator de ajuste baseado em condições de mercado."""
        factor = 1.0 # Renomeado

        # Volatilidade (exemplo, conditions['volatility_level'] seria 'low', 'normal', 'high', 'extreme')
        if 'volatility_level' in conditions: # Supondo que conditions tenha 'volatility_level'
            factor *= float(RISK_ADJUSTMENTS.VOLATILITY_ADJUSTMENTS.get(conditions['volatility_level'], 1.0))

        # Sessão (exemplo, conditions['session_name'] seria 'asia', 'london', etc.)
        if 'session_name' in conditions:
            factor *= float(RISK_ADJUSTMENTS.SESSION_ADJUSTMENTS.get(conditions['session_name'].lower(), 1.0))
        
        # Adicionar outros ajustes (dia da semana, eventos) se 'conditions' os fornecer
        current_weekday = datetime.now(timezone.utc).weekday() # 0 = Segunda
        factor *= float(RISK_ADJUSTMENTS.WEEKDAY_ADJUSTMENTS.get(current_weekday, 1.0))

        # Exemplo de ajuste por evento de notícia (se 'is_near_high_impact_news' estiver em conditions)
        if conditions.get('is_near_high_impact_news', False):
            factor *= float(RISK_ADJUSTMENTS.EVENT_ADJUSTMENTS.get('high_impact_news_imminent_30min', 0.5))


        return np.clip(factor, 0.1, 2.0) # Limitar o fator total de ajuste


    def _apply_final_lot_limits(self, lot_size_val: float, balance: float, # Renomeado
                     leverage: int, entry_price_val: float, symbol: str) -> float: # Renomeado
        """Aplica limites finais ao tamanho da posição (mín/máx, margem)."""
        # 1. Limite mínimo de lote (geralmente 0.01)
        limited_lot_size = max(getattr(CONFIG, 'MIN_LOT_SIZE', 0.01), lot_size_val) # Renomeado

        # 2. Limite de risco máximo por trade em relação ao saldo
        # (Ex: não arriscar mais que MAX_RISK_PER_TRADE do saldo, mesmo que o método sugira)
        # Isto já é indiretamente controlado pela maioria dos métodos que usam risk_percent.
        # Mas para Fixed Lot ou Kelly muito agressivo, é uma salvaguarda.
        # Refazendo o cálculo do risco com o limited_lot_size:
        # stop_pips_val é necessário aqui. Se não disponível, este limite é mais difícil de aplicar.
        # Este limite é mais sobre o risco percentual *resultante*.
        # Se a lógica já calcula lot_size baseado em risco percentual, este é redundante.
        # No entanto, pode ser uma checagem final.

        # 3. Limite por margem disponível
        # (Ex: não usar mais que X% da margem disponível ou Y% do saldo como margem)
        contract_sz = getattr(CONFIG, 'CONTRACT_SIZE', 100000) # Renomeado
        margin_per_std_lot = (1.0 * contract_sz * entry_price_val) / leverage if leverage > 0 else float('inf')
        
        # Não usar mais que, por exemplo, 20% do saldo como margem para uma única trade
        max_margin_usage_for_trade = balance * getattr(CONFIG, 'MAX_MARGIN_PER_TRADE_PCT', 0.20)
        max_lots_by_margin_limit = max_margin_usage_for_trade / margin_per_std_lot if margin_per_std_lot > 0 else 0.01
        
        limited_lot_size = min(limited_lot_size, max_lots_by_margin_limit)


        # 4. Limite máximo absoluto de lote (ex: 5 lotes, vindo de CONFIG ou RISK_LIMITS)
        max_abs_lot = getattr(CONFIG, 'MAX_ABSOLUTE_LOT_SIZE', 5.0)
        limited_lot_size = min(limited_lot_size, max_abs_lot)

        return max(0.01, limited_lot_size) # Garantir que nunca seja menor que 0.01


    def update_trade_history(self, trade_result_dict: Dict[str, Any]): # Renomeado e tipado
        """Atualiza histórico de trades e curva de equity para cálculos futuros."""
        if not isinstance(trade_result_dict, dict) or 'pnl' not in trade_result_dict:
            logger.warning(f"Resultado de trade inválido para PositionSizer: {trade_result_dict}")
            return

        self.trade_history.append(trade_result_dict)
        max_hist = getattr(CONFIG, 'POSITION_SIZER_TRADE_HISTORY_MAX', 1000) # Renomeado
        if len(self.trade_history) > max_hist:
            self.trade_history = self.trade_history[-max_hist:]

        # Atualizar equity curve
        if self.equity_curve: # Se já tem elementos
            new_equity_val = self.equity_curve[-1] + float(trade_result_dict.get('pnl',0.0)) # Renomeado
        else: # Se estiver vazia (não deveria se inicializada no __init__)
            # Assumir capital inicial se não houver equity_curve, ou logar erro.
            # Este caso deve ser raro se o __init__ for chamado corretamente.
            initial_bal_fallback = getattr(CONFIG, 'INITIAL_BALANCE', 10000.0)
            logger.warning("Equity curve estava vazia ao atualizar histórico de trades. Recriando com saldo inicial.")
            self.equity_curve = [initial_bal_fallback]
            new_equity_val = initial_bal_fallback + float(trade_result_dict.get('pnl',0.0))


        self.equity_curve.append(new_equity_val)
        if len(self.equity_curve) > max_hist + 1: # +1 por causa do saldo inicial
            self.equity_curve = [self.equity_curve[0]] + self.equity_curve[-(max_hist):] # Manter primeiro e últimos N


        # Atualizar streaks
        if float(trade_result_dict.get('pnl',0.0)) > 0:
            self.current_win_streak += 1
            self.current_loss_streak = 0
        elif float(trade_result_dict.get('pnl',0.0)) < 0:
            self.current_loss_streak += 1
            self.current_win_streak = 0
        # Breakeven trades não alteram streaks


    def get_sizing_module_stats(self) -> Dict[str, Any]: # Renomeado e tipado
        """Retorna estatísticas do módulo de position sizing."""
        if not self.trade_history:
            return {'method': self.method, 'status': 'Nenhum trade no histórico do sizer.'}

        # Tentar obter 'lot_size' e 'risk_amount_currency' do histórico
        # O histórico precisa ser populado com o PositionSizeResult completo ou campos relevantes.
        # Assumindo que trade_result_dict em update_trade_history pode conter esses campos.
        lot_sizes_hist = [float(tr.get('calculated_lot_size', 0.0)) for tr in self.trade_history if tr.get('calculated_lot_size') is not None] # Renomeado
        risk_amounts_hist = [float(tr.get('calculated_risk_amount', 0.0)) for tr in self.trade_history if tr.get('calculated_risk_amount') is not None] # Renomeado

        return {
            'current_method': self.method,
            'total_trades_considered_by_sizer': len(self.trade_history),
            'avg_calculated_lot_size': round(np.mean(lot_sizes_hist),2) if lot_sizes_hist else 0.0,
            'max_calculated_lot_size': round(max(lot_sizes_hist),2) if lot_sizes_hist else 0.0,
            'min_calculated_lot_size': round(min(lot_sizes_hist),2) if lot_sizes_hist else 0.01,
            'avg_calculated_risk_amount_currency': round(np.mean(risk_amounts_hist),2) if risk_amounts_hist else 0.0,
            'current_win_streak': self.current_win_streak,
            'current_loss_streak': self.current_loss_streak,
            'equity_curve_length': len(self.equity_curve),
            'current_equity_for_sizer': self.equity_curve[-1] if self.equity_curve else 0.0
        }