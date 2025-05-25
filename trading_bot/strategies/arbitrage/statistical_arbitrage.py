# strategies/arbitrage/statistical_arbitrage.py
import numpy as np
import pandas as pd
from typing import Dict, Optional, Any, List, Tuple
from datetime import datetime
from statsmodels.tsa.stattools import coint
from sklearn.linear_model import LinearRegression

from strategies.base_strategy import BaseStrategy, Signal, Position, ExitSignal
from core.market_regime import MarketRegime
from utils.logger import setup_logger

logger = setup_logger("statistical_arbitrage")

class StatisticalArbitrageStrategy(BaseStrategy):
    """
    Estratégia de arbitragem estatística EURUSD vs correlatos
    
    Pares:
    - EURUSD vs DXY (Dollar Index)
    - EURUSD vs GBPUSD
    - EURUSD vs EURJPY/USDJPY
    """
    
    def __init__(self):
        super().__init__("StatArb_EURUSD_DXY")
        self.suitable_regimes = [MarketRegime.RANGE, MarketRegime.TREND]
        self.min_time_between_signals = 600  # 10 minutos
        
        # Parâmetros de cointegração
        self.hedge_ratio = 1.0
        self.spread_mean = 0
        self.spread_std = 1
        self.half_life = 20
        
        # Histórico
        self.spread_history = []
        self.cointegration_valid = False
        
    def get_default_parameters(self) -> Dict[str, Any]:
        return {
            # Cointegration
            'lookback_period': 1000,
            'cointegration_pvalue': 0.05,
            'min_half_life': 5,
            'max_half_life': 50,
            
            # Entry/Exit
            'entry_zscore': 2.0,
            'exit_zscore': 0.5,
            'stop_zscore': 3.0,
            
            # Risk
            'position_size_pct': 0.02,  # 2% do capital
            'rebalance_threshold': 0.1,  # 10% desvio
            'max_holding_periods': 100,
            
            # Pairs
            'use_dxy': True,
            'use_gbpusd': True,
            'use_eurjpy': False,
            
            # Filters
            'min_spread_quality': 0.8,
            'check_stationarity': True,
            'rolling_window': 100
        }
    
    async def calculate_indicators(self, market_context: Dict) -> Dict[str, Any]:
        """Calcula spread e indicadores de arbitragem"""
        try:
            eurusd_ticks = market_context.get('recent_ticks', [])
            
            # Simular dados de outros pares (em produção, buscar dados reais)
            dxy_data = self._simulate_dxy_data(eurusd_ticks)
            
            if len(eurusd_ticks) < self.parameters['lookback_period']:
                return {}
            
            # Preços
            eurusd_prices = np.array([t.mid for t in eurusd_ticks])
            dxy_prices = dxy_data
            
            # Verificar cointegração
            if not self.cointegration_valid or len(self.spread_history) % 100 == 0:
                self._check_cointegration(eurusd_prices, dxy_prices)
            
            if not self.cointegration_valid:
                return {}
            
            # Calcular spread
            spread = eurusd_prices[-1] - (self.hedge_ratio * dxy_prices[-1])
            
            # Atualizar histórico
            self.spread_history.append(spread)
            if len(self.spread_history) > self.parameters['lookback_period']:
                self.spread_history.pop(0)
            
            # Estatísticas do spread
            if len(self.spread_history) >= 20:
                spread_array = np.array(self.spread_history)
                self.spread_mean = np.mean(spread_array[-self.parameters['rolling_window']:])
                self.spread_std = np.std(spread_array[-self.parameters['rolling_window']:])
                
                # Z-Score
                zscore = (spread - self.spread_mean) / self.spread_std if self.spread_std > 0 else 0
                
                # Half-life (mean reversion speed)
                half_life = self._calculate_half_life(spread_array)
            else:
                zscore = 0
                half_life = self.half_life
            
            # Outros indicadores
            indicators = {
                'spread': spread,
                'spread_mean': self.spread_mean,
                'spread_std': self.spread_std,
                'zscore': zscore,
                'half_life': half_life,
                'hedge_ratio': self.hedge_ratio,
                'cointegration_valid': self.cointegration_valid,
                'eurusd_price': eurusd_prices[-1],
                'dxy_price': dxy_prices[-1],
                'spread_history_len': len(self.spread_history)
            }
            
            return indicators
            
        except Exception as e:
            logger.error(f"Erro ao calcular indicadores de arbitragem: {e}")
            return {}
    
    def _simulate_dxy_data(self, eurusd_ticks: List) -> np.ndarray:
        """Simula dados DXY (em produção, usar dados reais)"""
        # DXY tem correlação negativa com EURUSD
        eurusd_prices = np.array([t.mid for t in eurusd_ticks])
        
        # Simular com correlação negativa + ruído
        dxy_base = 100  # Base do índice
        correlation = -0.85
        
        # Normalizar EURUSD
        eurusd_returns = np.diff(np.log(eurusd_prices))
        
        # Gerar DXY returns
        noise = np.random.normal(0, 0.0001, len(eurusd_returns))
        dxy_returns = correlation * eurusd_returns + np.sqrt(1 - correlation**2) * noise
        
        # Reconstruir preços
        dxy_prices = np.exp(np.cumsum(np.concatenate([[np.log(dxy_base)], dxy_returns])))
        
        return dxy_prices
    
    def _check_cointegration(self, series1: np.ndarray, series2: np.ndarray):
        """Verifica cointegração entre séries"""
        try:
            # Teste de cointegração Engle-Granger
            score, pvalue, _ = coint(series1[-self.parameters['lookback_period']:],
                                    series2[-self.parameters['lookback_period']:])
            
            if pvalue < self.parameters['cointegration_pvalue']:
                # Calcular hedge ratio via regressão
                X = series2[-self.parameters['lookback_period']:].reshape(-1, 1)
                y = series1[-self.parameters['lookback_period']:]
                
                model = LinearRegression()
                model.fit(X, y)
                
                self.hedge_ratio = model.coef_[0]
                self.cointegration_valid = True
                
                logger.info(f"Cointegração válida. P-value: {pvalue:.4f}, Hedge ratio: {self.hedge_ratio:.4f}")
            else:
                self.cointegration_valid = False
                logger.warning(f"Cointegração inválida. P-value: {pvalue:.4f}")
                
        except Exception as e:
            logger.error(f"Erro ao verificar cointegração: {e}")
            self.cointegration_valid = False
    
    def _calculate_half_life(self, spread: np.ndarray) -> float:
        """Calcula half-life do spread (velocidade de mean reversion)"""
        if len(spread) < 20:
            return self.parameters['max_half_life']
        
        try:
            # Regressão do spread em seu lag
            spread_lag = spread[:-1]
            spread_diff = spread[1:] - spread[:-1]
            
            model = LinearRegression()
            model.fit(spread_lag.reshape(-1, 1), spread_diff)
            
            theta = model.coef_[0]
            
            if theta >= 0:  # Não estacionário
                return self.parameters['max_half_life']
            
            half_life = -np.log(2) / theta
            
            # Limitar ao range permitido
            return np.clip(half_life, 
                          self.parameters['min_half_life'],
                          self.parameters['max_half_life'])
            
        except:
            return self.parameters['max_half_life']
    
    async def generate_signal(self, market_context: Dict) -> Optional[Signal]:
        """Gera sinais de arbitragem"""
        indicators = self.indicators
        
        if not indicators or not indicators.get('cointegration_valid', False):
            return None
        
        zscore = indicators['zscore']
        half_life = indicators['half_life']
        
        # Verificar half-life
        if half_life < self.parameters['min_half_life'] or \
           half_life > self.parameters['max_half_life']:
            return None
        
        signal_type = None
        
        # Sinais baseados em Z-Score
        if abs(zscore) >= self.parameters['entry_zscore']:
            if zscore >= self.parameters['entry_zscore']:
                # Spread muito alto - vender EURUSD, comprar DXY (hedge)
                signal_type = 'sell'
            elif zscore <= -self.parameters['entry_zscore']:
                # Spread muito baixo - comprar EURUSD, vender DXY (hedge)
                signal_type = 'buy'
        
        if signal_type:
            return self._create_arbitrage_signal(signal_type, indicators, market_context)
        
        return None
    
    def _create_arbitrage_signal(self, signal_type: str, indicators: Dict,
                                market_context: Dict) -> Signal:
        """Cria sinal de arbitragem"""
        price = indicators['eurusd_price']
        
        # Stops baseados em Z-Score
        if signal_type == 'buy':
            # Stop se Z-Score ficar ainda mais negativo
            stop_spread = indicators['spread_mean'] - (indicators['spread_std'] * self.parameters['stop_zscore'])
            current_dxy = indicators['dxy_price']
            stop_price = stop_spread + (self.hedge_ratio * current_dxy)
            
            # Target no mean
            target_spread = indicators['spread_mean']
            take_profit = target_spread + (self.hedge_ratio * current_dxy)
            
        else:  # sell
            # Stop se Z-Score ficar ainda mais positivo
            stop_spread = indicators['spread_mean'] + (indicators['spread_std'] * self.parameters['stop_zscore'])
            current_dxy = indicators['dxy_price']
            stop_price = stop_spread + (self.hedge_ratio * current_dxy)
            
            # Target no mean
            target_spread = indicators['spread_mean']
            take_profit = target_spread + (self.hedge_ratio * current_dxy)
        
        # Ajustar stops para serem mais conservadores
        if signal_type == 'buy':
            stop_loss = min(stop_price, price * 0.997)  # Max 0.3% loss
            take_profit = max(take_profit, price * 1.001)  # Min 0.1% profit
        else:
            stop_loss = max(stop_price, price * 1.003)
            take_profit = min(take_profit, price * 0.999)
        
        # Confiança baseada em half-life e Z-Score
        confidence = self._calculate_arbitrage_confidence(indicators)
        
        signal = Signal(
            timestamp=datetime.now(),
            strategy_name=self.name,
            side=signal_type,
            confidence=confidence,
            stop_loss=stop_loss,
            take_profit=take_profit,
            entry_price=price,
            reason=f"Statistical Arbitrage {signal_type.upper()} - Z-Score: {indicators['zscore']:.2f}",
            metadata={
                'zscore': indicators['zscore'],
                'spread': indicators['spread'],
                'half_life': indicators['half_life'],
                'hedge_ratio': indicators['hedge_ratio'],
                'dxy_price': indicators['dxy_price']
            }
        )
        
        return signal
    
    def _calculate_arbitrage_confidence(self, indicators: Dict) -> float:
        """Calcula confiança do sinal de arbitragem"""
        confidence = 0.6  # Base
        
        # Z-Score extremo
        zscore_abs = abs(indicators['zscore'])
        if zscore_abs > 3:
            confidence += 0.15
        elif zscore_abs > 2.5:
            confidence += 0.1
        
        # Half-life ideal (rápida mean reversion)
        half_life = indicators['half_life']
        if half_life < 15:
            confidence += 0.15
        elif half_life < 25:
            confidence += 0.1
        
        # Histórico de spread suficiente
        if indicators['spread_history_len'] >= self.parameters['lookback_period']:
            confidence += 0.05
        
        return min(confidence, 0.95)
    
    async def calculate_exit_conditions(self, position: Position,
                                       current_price: float) -> Optional[ExitSignal]:
        """Condições de saída para arbitragem"""
        indicators = self.indicators
        
        if not indicators:
            return None
        
        zscore = indicators['zscore']
        
        # Saída principal: Z-Score retornou ao normal
        if position.side == 'buy':
            if zscore >= -self.parameters['exit_zscore']:
                return ExitSignal(
                    position_id=position.id,
                    reason=f"Z-Score normalized: {zscore:.2f}",
                    exit_price=current_price
                )
        else:  # sell
            if zscore <= self.parameters['exit_zscore']:
                return ExitSignal(
                    position_id=position.id,
                    reason=f"Z-Score normalized: {zscore:.2f}",
                    exit_price=current_price
                )
        
        # Saída se cointegração quebrar
        if not indicators.get('cointegration_valid', False):
            return ExitSignal(
                position_id=position.id,
                reason="Cointegration broken",
                exit_price=current_price
            )
        
        # Saída por tempo máximo
        if 'entry_time' in position.metadata:
            entry_time = position.metadata['entry_time']
            periods_held = (datetime.now() - entry_time).total_seconds() / 60  # minutos
            
            expected_exit_time = indicators['half_life'] * 2
            
            if periods_held > expected_exit_time:
                return ExitSignal(
                    position_id=position.id,
                    reason=f"Max holding time exceeded ({periods_held:.0f} min)",
                    exit_price=current_price
                )
        
        return None