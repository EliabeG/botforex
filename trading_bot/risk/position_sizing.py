# risk/position_sizing.py
"""Sistema de dimensionamento de posições"""
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta

from config.risk_config import RISK_LIMITS, RISK_PARAMS, POSITION_SIZING
from utils.logger import setup_logger
from utils.helpers import calculate_position_size, calculate_pip_value

logger = setup_logger("position_sizing")

@dataclass
class PositionSizeResult:
    """Resultado do cálculo de tamanho de posição"""
    lot_size: float
    risk_amount: float
    risk_percent: float
    pip_value: float
    stop_distance_pips: float
    potential_loss: float
    margin_required: float
    method_used: str
    adjustments_applied: List[str]
    
class PositionSizer:
    """Calculador de tamanho de posição com múltiplos métodos"""
    
    def __init__(self, method: str = POSITION_SIZING.FIXED_RISK):
        self.method = method
        self.method_config = POSITION_SIZING.METHOD_CONFIG.get(method, {})
        self.trade_history = []
        self.equity_curve.append(new_equity)
        
        # Atualizar streak
        if trade_result['pnl'] > 0:
            self.current_streak = max(0, self.current_streak) + 1
        else:
            self.current_streak = min(0, self.current_streak) - 1
    
    def get_sizing_stats(self) -> Dict:
        """Retorna estatísticas do position sizing"""
        if not self.trade_history:
            return {}
        
        lot_sizes = [t.get('lot_size', 0) for t in self.trade_history if 'lot_size' in t]
        risk_amounts = [t.get('risk_amount', 0) for t in self.trade_history if 'risk_amount' in t]
        
        return {
            'method': self.method,
            'total_trades': len(self.trade_history),
            'avg_lot_size': np.mean(lot_sizes) if lot_sizes else 0,
            'max_lot_size': max(lot_sizes) if lot_sizes else 0,
            'min_lot_size': min(lot_sizes) if lot_sizes else 0,
            'avg_risk_amount': np.mean(risk_amounts) if risk_amounts else 0,
            'current_streak': self.current_streak,
            'equity_curve_length': len(self.equity_curve)
        } = []
        self.current_streak = 0
        
    def calculate_position_size(self,
                              account_balance: float,
                              entry_price: float,
                              stop_loss: float,
                              symbol: str = "EURUSD",
                              leverage: int = 500,
                              market_conditions: Optional[Dict] = None) -> PositionSizeResult:
        """
        Calcula tamanho da posição usando método configurado
        
        Args:
            account_balance: Saldo atual da conta
            entry_price: Preço de entrada
            stop_loss: Preço do stop loss
            symbol: Símbolo do par
            leverage: Alavancagem disponível
            market_conditions: Condições atuais do mercado
        
        Returns:
            PositionSizeResult com detalhes do cálculo
        """
        try:
            # Calcular distância do stop em pips
            stop_distance_pips = abs(entry_price - stop_loss) * 10000
            
            if stop_distance_pips < RISK_PARAMS.MIN_STOP_DISTANCE * 10000:
                logger.warning(f"Stop muito próximo: {stop_distance_pips:.1f} pips")
                stop_distance_pips = RISK_PARAMS.MIN_STOP_DISTANCE * 10000
            
            # Calcular tamanho base usando método selecionado
            if self.method == POSITION_SIZING.FIXED_LOT:
                lot_size = self._calculate_fixed_lot()
                
            elif self.method == POSITION_SIZING.FIXED_RISK:
                lot_size = self._calculate_fixed_risk(
                    account_balance,
                    stop_distance_pips,
                    symbol
                )
                
            elif self.method == POSITION_SIZING.KELLY_CRITERION:
                lot_size = self._calculate_kelly(
                    account_balance,
                    stop_distance_pips,
                    symbol
                )
                
            elif self.method == POSITION_SIZING.VOLATILITY_BASED:
                lot_size = self._calculate_volatility_based(
                    account_balance,
                    stop_distance_pips,
                    symbol,
                    market_conditions
                )
                
            elif self.method == POSITION_SIZING.EQUITY_CURVE:
                lot_size = self._calculate_equity_curve_based(
                    account_balance,
                    stop_distance_pips,
                    symbol
                )
                
            else:
                # Default para fixed risk
                lot_size = self._calculate_fixed_risk(
                    account_balance,
                    stop_distance_pips,
                    symbol
                )
            
            # Aplicar ajustes dinâmicos
            adjustments = []
            original_lot_size = lot_size
            
            # Ajuste por performance
            performance_multiplier = self._get_performance_multiplier()
            if performance_multiplier != 1.0:
                lot_size *= performance_multiplier
                adjustments.append(f"Performance: {performance_multiplier:.2f}x")
            
            # Ajuste por condições de mercado
            if market_conditions:
                market_multiplier = self._get_market_conditions_multiplier(market_conditions)
                if market_multiplier != 1.0:
                    lot_size *= market_multiplier
                    adjustments.append(f"Market: {market_multiplier:.2f}x")
            
            # Aplicar limites
            lot_size = self._apply_limits(lot_size, account_balance, leverage)
            
            # Arredondar para precisão do broker
            lot_size = round(lot_size, 2)
            
            # Calcular valores finais
            pip_value = calculate_pip_value(symbol, lot_size)
            risk_amount = stop_distance_pips * pip_value
            risk_percent = (risk_amount / account_balance) * 100
            potential_loss = risk_amount
            margin_required = (lot_size * 100000 * entry_price) / leverage
            
            # Log do cálculo
            logger.info(f"Position size calculado: {lot_size} lotes | "
                       f"Risco: ${risk_amount:.2f} ({risk_percent:.1f}%) | "
                       f"Método: {self.method}")
            
            if adjustments:
                logger.info(f"Ajustes aplicados: {', '.join(adjustments)}")
            
            return PositionSizeResult(
                lot_size=lot_size,
                risk_amount=risk_amount,
                risk_percent=risk_percent,
                pip_value=pip_value,
                stop_distance_pips=stop_distance_pips,
                potential_loss=potential_loss,
                margin_required=margin_required,
                method_used=self.method,
                adjustments_applied=adjustments
            )
            
        except Exception as e:
            logger.error(f"Erro ao calcular position size: {e}")
            # Retornar tamanho mínimo em caso de erro
            return PositionSizeResult(
                lot_size=0.01,
                risk_amount=0,
                risk_percent=0,
                pip_value=calculate_pip_value(symbol, 0.01),
                stop_distance_pips=stop_distance_pips,
                potential_loss=0,
                margin_required=0,
                method_used="error_fallback",
                adjustments_applied=["Erro - usando mínimo"]
            )
    
    def _calculate_fixed_lot(self) -> float:
        """Calcula tamanho fixo de lote"""
        return self.method_config.get('lot_size', 0.01)
    
    def _calculate_fixed_risk(self, balance: float, stop_pips: float, 
                             symbol: str) -> float:
        """Calcula baseado em risco fixo"""
        risk_percent = self.method_config.get('risk_percent', 0.01)
        risk_amount = balance * risk_percent
        pip_value = calculate_pip_value(symbol, 1.0)
        
        if stop_pips > 0:
            lot_size = risk_amount / (stop_pips * pip_value)
            return lot_size
        
        return 0.01
    
    def _calculate_kelly(self, balance: float, stop_pips: float, 
                        symbol: str) -> float:
        """Calcula usando Kelly Criterion"""
        # Calcular win rate e profit factor dos últimos trades
        lookback = self.method_config.get('lookback_trades', 100)
        recent_trades = self.trade_history[-lookback:] if self.trade_history else []
        
        if len(recent_trades) < 20:  # Mínimo de trades para Kelly
            # Usar fixed risk como fallback
            return self._calculate_fixed_risk(balance, stop_pips, symbol)
        
        # Calcular estatísticas
        wins = [t['pnl'] for t in recent_trades if t['pnl'] > 0]
        losses = [abs(t['pnl']) for t in recent_trades if t['pnl'] < 0]
        
        if not wins or not losses:
            return self._calculate_fixed_risk(balance, stop_pips, symbol)
        
        win_rate = len(wins) / len(recent_trades)
        avg_win = np.mean(wins)
        avg_loss = np.mean(losses)
        
        # Kelly formula: f = (p*b - q) / b
        # onde p = win_rate, q = 1-p, b = avg_win/avg_loss
        b = avg_win / avg_loss if avg_loss > 0 else 1
        q = 1 - win_rate
        
        kelly_fraction = (win_rate * b - q) / b if b > 0 else 0
        
        # Aplicar fração conservadora do Kelly
        conservative_fraction = self.method_config.get('kelly_fraction', 0.25)
        kelly_fraction *= conservative_fraction
        
        # Limitar ao máximo configurado
        max_kelly = self.method_config.get('max_kelly', 0.02)
        kelly_fraction = min(max(kelly_fraction, 0), max_kelly)
        
        # Calcular tamanho
        risk_amount = balance * kelly_fraction
        pip_value = calculate_pip_value(symbol, 1.0)
        
        if stop_pips > 0:
            lot_size = risk_amount / (stop_pips * pip_value)
            return lot_size
        
        return 0.01
    
    def _calculate_volatility_based(self, balance: float, stop_pips: float,
                                   symbol: str, market_conditions: Dict) -> float:
        """Calcula baseado em volatilidade"""
        base_risk = self.method_config.get('base_risk', 0.01)
        
        if market_conditions and 'volatility' in market_conditions:
            current_volatility = market_conditions['volatility']
            
            # Ajustar risco inversamente à volatilidade
            # Alta volatilidade = menor posição
            if current_volatility > RISK_PARAMS.HIGH_VOLATILITY_THRESHOLD:
                volatility_multiplier = 0.5
            elif current_volatility > 0.015:
                volatility_multiplier = 0.75
            elif current_volatility < 0.005:
                volatility_multiplier = 1.25
            else:
                volatility_multiplier = 1.0
            
            adjusted_risk = base_risk * volatility_multiplier
        else:
            adjusted_risk = base_risk
        
        # Calcular tamanho
        risk_amount = balance * adjusted_risk
        pip_value = calculate_pip_value(symbol, 1.0)
        
        if stop_pips > 0:
            lot_size = risk_amount / (stop_pips * pip_value)
            return lot_size
        
        return 0.01
    
    def _calculate_equity_curve_based(self, balance: float, stop_pips: float,
                                     symbol: str) -> float:
        """Calcula baseado na curva de equity"""
        base_risk = self.method_config.get('base_risk', 0.01)
        
        if len(self.equity_curve) < 20:
            # Sem dados suficientes
            return self._calculate_fixed_risk(balance, stop_pips, symbol)
        
        # Calcular média móvel da equity
        ma_period = self.method_config.get('equity_ma_period', 20)
        recent_equity = self.equity_curve[-ma_period:]
        equity_ma = np.mean(recent_equity)
        
        # Se equity atual > MA, aumentar risco
        # Se equity atual < MA, diminuir risco
        current_equity = self.equity_curve[-1]
        
        if current_equity > equity_ma:
            multiplier = self.method_config.get('increase_on_winning', 1.1)
        else:
            multiplier = self.method_config.get('decrease_on_losing', 0.9)
        
        adjusted_risk = base_risk * multiplier
        
        # Calcular tamanho
        risk_amount = balance * adjusted_risk
        pip_value = calculate_pip_value(symbol, 1.0)
        
        if stop_pips > 0:
            lot_size = risk_amount / (stop_pips * pip_value)
            return lot_size
        
        return 0.01
    
    def _get_performance_multiplier(self) -> float:
        """Obtém multiplicador baseado em performance recente"""
        if not self.trade_history:
            return 1.0
        
        # Verificar streak
        recent_trades = self.trade_history[-5:]
        
        # Winning streak
        consecutive_wins = 0
        for trade in reversed(recent_trades):
            if trade['pnl'] > 0:
                consecutive_wins += 1
            else:
                break
        
        # Losing streak
        consecutive_losses = 0
        for trade in reversed(recent_trades):
            if trade['pnl'] < 0:
                consecutive_losses += 1
            else:
                break
        
        # Aplicar ajustes
        if consecutive_wins >= 5:
            return 1.2
        elif consecutive_wins >= 3:
            return 1.1
        elif consecutive_losses >= 3:
            return 0.5
        elif consecutive_losses >= 2:
            return 0.75
        
        return 1.0
    
    def _get_market_conditions_multiplier(self, conditions: Dict) -> float:
        """Obtém multiplicador baseado em condições de mercado"""
        multiplier = 1.0
        
        # Volatilidade
        if 'volatility' in conditions:
            if conditions['volatility'] > RISK_PARAMS.HIGH_VOLATILITY_THRESHOLD:
                multiplier *= 0.7
            elif conditions['volatility'] < 0.005:
                multiplier *= 1.2
        
        # Sessão
        if 'session' in conditions:
            session_multipliers = {
                'Asia': 0.8,
                'London': 1.0,
                'NewYork': 1.0,
                'Overlap': 1.1,
                'Closed': 0.5
            }
            multiplier *= session_multipliers.get(conditions['session'], 1.0)
        
        # Spread
        if 'spread' in conditions:
            if conditions['spread'] > RISK_PARAMS.MAX_ALLOWED_SLIPPAGE:
                multiplier *= 0.5
        
        return multiplier
    
    def _apply_limits(self, lot_size: float, balance: float, 
                     leverage: int) -> float:
        """Aplica limites ao tamanho da posição"""
        # Limite mínimo
        lot_size = max(lot_size, 0.01)
        
        # Limite por risco máximo
        max_risk_lots = (balance * RISK_LIMITS.MAX_RISK_PER_TRADE) / 100
        lot_size = min(lot_size, max_risk_lots)
        
        # Limite por margem disponível
        margin_per_lot = 100000 / leverage
        max_lots_by_margin = (balance * 0.8) / margin_per_lot  # 80% da margem
        lot_size = min(lot_size, max_lots_by_margin)
        
        # Limite absoluto
        lot_size = min(lot_size, 5.0)
        
        return lot_size
    
    def update_trade_history(self, trade_result: Dict):
        """Atualiza histórico de trades para cálculos futuros"""
        self.trade_history.append(trade_result)
        
        # Manter apenas últimos N trades
        max_history = 1000
        if len(self.trade_history) > max_history:
            self.trade_history = self.trade_history[-max_history:]
        
        # Atualizar equity curve
        if self.equity_curve:
            new_equity = self.equity_curve[-1] + trade_result['pnl']
        else:
            new_equity = 10000 + trade_result['pnl']  # Assumir capital inicial
        
        self.equity_curve