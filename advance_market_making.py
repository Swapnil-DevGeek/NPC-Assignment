import logging
import statistics
import numpy as np
from decimal import Decimal
from collections import deque
from typing import Dict, List, Tuple
from enum import Enum

from hummingbot.core.data_type.common import OrderType, TradeType
from hummingbot.core.data_type.order_candidate import OrderCandidate
from hummingbot.core.event.events import OrderFilledEvent
from hummingbot.strategy.script_strategy_base import ScriptStrategyBase


class TrendState(Enum):
    STRONG_UPTREND = "STRONG_UPTREND"
    WEAK_UPTREND = "WEAK_UPTREND"
    NEUTRAL = "NEUTRAL"
    WEAK_DOWNTREND = "WEAK_DOWNTREND"
    STRONG_DOWNTREND = "STRONG_DOWNTREND"


class AdvancedMarketMakingStrategy(ScriptStrategyBase):
    trading_pair = "SHIB-USDT"
    exchange = "binance_paper_trade"
    
    # Base order size in units (adjustable as needed)
    base_order_size = Decimal("10000000")
    lookback_period = 20
    order_refresh_time = 15  # seconds

    # Volatility filter thresholds (ATR values); adjust based on asset characteristics
    volatility_threshold_low = Decimal("0.0005")
    volatility_threshold_high = Decimal("0.005")
    
    # Risk control: maximum allowed drawdown (realized P&L) before halting orders.
    max_drawdown_threshold = Decimal("-1000")
    
    # Trend Detection Parameters
    short_ema_period = 8
    long_ema_period = 21
    rsi_period = 14
    rsi_oversold = 30
    rsi_overbought = 70
    macd_fast = 12
    macd_slow = 26
    macd_signal = 9
    
    markets = {exchange: {trading_pair}}

    def __init__(self, connectors: Dict[str, any]):
        super().__init__(connectors)
        self.price_history = deque(maxlen=max(self.lookback_period, 
                                            self.short_ema_period, 
                                            self.long_ema_period, 
                                            self.rsi_period,
                                            self.macd_slow + self.macd_signal))
        self.volume_history = deque(maxlen=self.lookback_period)
        self.create_timestamp = 0

        # P&L tracking and inventory monitoring
        self.realized_pnl = Decimal("0")  # cumulative cash flow from fills
        self.net_position = Decimal("0")  # net position in base asset
        self.max_realized_pnl = Decimal("0")  # highest reached pnl, for drawdown calculations

        # Trend state tracking
        self.trend_state = TrendState.NEUTRAL
        self.detected_trends = []  # Store recent trend signals for consensus
        
        self.logger().info("Advanced Market Making Strategy Initialized")

    def on_tick(self):
        # Check if risk limits are breached
        if self.realized_pnl < self.max_drawdown_threshold:
            self.logger().warning(
                f"Risk limit reached: Realized PnL {self.realized_pnl} below threshold {self.max_drawdown_threshold}. Halting new orders."
            )
            return

        if self.current_timestamp < self.create_timestamp or not self.ready_to_trade:
            return

        # Get current mid price and market data
        mid_price = self.connectors[self.exchange].get_mid_price(self.trading_pair)
        if mid_price is None:
            self.logger().warning("Mid price unavailable.")
            return

        # Get recent trading volume if available (this would need to be adapted to connector capabilities)
        try:
            recent_volume = self.connectors[self.exchange].get_last_traded_volume(self.trading_pair)
            self.volume_history.append(float(recent_volume))
        except:
            # If volume data is unavailable, use a placeholder
            self.volume_history.append(1.0)

        price = float(mid_price)
        self.price_history.append(price)

        # Ensure we have enough historical data before proceeding
        if len(self.price_history) < max(self.lookback_period, self.long_ema_period, self.rsi_period, self.macd_slow + self.macd_signal):
            self.logger().info(f"Collecting price history: {len(self.price_history)}/{max(self.lookback_period, self.long_ema_period, self.rsi_period)}")
            self.create_timestamp = self.current_timestamp + 1
            return

        self.cancel_all_orders()

        try:
            # Calculate core technical indicators for trend detection
            atr = self.calculate_atr()
            sma = self.calculate_sma()
            
            # Advanced trend detection methods
            trend_signals = self.detect_trend_consensus()
            self.trend_state = self.determine_trend_state(trend_signals)
            
            # Calculate spread based on volatility and trend state
            volatility_spread = atr * Decimal("0.5")  # Base spread from ATR
            
            # Adjust spread based on trend strength and direction
            trend_bias = self.calculate_trend_bias()
            
            # Inventory management bias
            inventory_bias = self.calculate_inventory_bias()
            
            # Combine biases with different weights based on trend strength
            if self.trend_state in [TrendState.STRONG_UPTREND, TrendState.STRONG_DOWNTREND]:
                # Prioritize trend in strong trends
                final_bias = (trend_bias * Decimal("0.7")) + (inventory_bias * Decimal("0.3"))
            else:
                # Prioritize inventory management in weak or neutral trends
                final_bias = (trend_bias * Decimal("0.3")) + (inventory_bias * Decimal("0.7"))

            # Dynamic order sizing based on volatility and trend:
            order_size = self.calculate_dynamic_order_size(atr)

            # Calculate bid and ask prices incorporating the volatility spread and combined bias.
            bid_price = Decimal(str(price)) - volatility_spread * (Decimal("1") - final_bias)
            ask_price = Decimal(str(price)) + volatility_spread * (Decimal("1") + final_bias)

            self.logger().info(
                f"Placing MM orders | Trend: {self.trend_state.name} | Bias: {final_bias} | "
                f"Bid: {bid_price:.10f} ({order_size}) | Ask: {ask_price:.10f} ({order_size})"
            )
            
            # Place orders with sizes adjusted by trend direction
            bid_size = order_size
            ask_size = order_size
            
            # Optionally skew sizes based on strong trend signals
            if self.trend_state == TrendState.STRONG_UPTREND:
                bid_size = order_size * Decimal("1.2")  # More aggressive buys in uptrend
                ask_size = order_size * Decimal("0.8")  # Less aggressive sells in uptrend
            elif self.trend_state == TrendState.STRONG_DOWNTREND:
                bid_size = order_size * Decimal("0.8")  # Less aggressive buys in downtrend
                ask_size = order_size * Decimal("1.2")  # More aggressive sells in downtrend
            
            self.place_order(TradeType.BUY, bid_price, bid_size)
            self.place_order(TradeType.SELL, ask_price, ask_size)

        except Exception as e:
            self.logger().error(f"Market making logic error: {e}")

        self.create_timestamp = self.current_timestamp + self.order_refresh_time

    def detect_trend_consensus(self) -> dict:
        """
        Detect trend using multiple indicators and return consensus signals
        """
        signals = {}
        
        # 1. EMA Crossover Analysis
        short_ema = self.calculate_ema(self.short_ema_period)
        long_ema = self.calculate_ema(self.long_ema_period)
        signals["ema_signal"] = 1 if short_ema > long_ema else (-1 if short_ema < long_ema else 0)
        
        # 2. RSI Analysis
        rsi = self.calculate_rsi(self.rsi_period)
        if rsi > self.rsi_overbought:
            signals["rsi_signal"] = -1  # Overbought, potential downtrend
        elif rsi < self.rsi_oversold:
            signals["rsi_signal"] = 1   # Oversold, potential uptrend
        else:
            signals["rsi_signal"] = 0   # Neutral
        
        # 3. MACD Analysis
        macd, signal_line, histogram = self.calculate_macd()
        signals["macd_signal"] = 1 if macd > signal_line else (-1 if macd < signal_line else 0)
        
        # 4. Volume-Weighted Price Direction
        vwpd = self.calculate_volume_weighted_price_direction()
        signals["volume_trend"] = 1 if vwpd > 0 else (-1 if vwpd < 0 else 0)
        
        # 5. Price momentum (recent price action trend)
        momentum = self.calculate_price_momentum()
        signals["momentum"] = 1 if momentum > 0 else (-1 if momentum < 0 else 0)
        
        # Store detected signals for later reference
        self.detected_trends = [signals]
        if len(self.detected_trends) > 5:  # Keep only recent signal history
            self.detected_trends.pop(0)
            
        return signals

    def determine_trend_state(self, signals: dict) -> TrendState:
        """
        Combine multiple trend signals into an overall trend state assessment
        """
        # Calculate a weighted signal score
        weights = {
            "ema_signal": 0.25,
            "rsi_signal": 0.15,
            "macd_signal": 0.25,
            "volume_trend": 0.2,
            "momentum": 0.15
        }
        
        score = sum(signals[k] * weights[k] for k in signals)
        
        # Determine trend state based on score
        if score > 0.5:
            return TrendState.STRONG_UPTREND
        elif score > 0.1:
            return TrendState.WEAK_UPTREND
        elif score < -0.5:
            return TrendState.STRONG_DOWNTREND
        elif score < -0.1:
            return TrendState.WEAK_DOWNTREND
        else:
            return TrendState.NEUTRAL

    def calculate_trend_bias(self) -> Decimal:
        """
        Calculate bid/ask bias based on trend state
        """
        trend_biases = {
            TrendState.STRONG_UPTREND: Decimal("0.4"),    # Strong bullish bias
            TrendState.WEAK_UPTREND: Decimal("0.2"),      # Weak bullish bias
            TrendState.NEUTRAL: Decimal("0.0"),           # No bias
            TrendState.WEAK_DOWNTREND: Decimal("-0.2"),   # Weak bearish bias  
            TrendState.STRONG_DOWNTREND: Decimal("-0.4"), # Strong bearish bias
        }
        return trend_biases[self.trend_state]

    def calculate_inventory_bias(self) -> Decimal:
        """Calculate bias based on current inventory position"""
        skew = self.inventory_skew()
        
        # More aggressive bias for extreme inventory positions
        if skew > 0.8:
            return Decimal("-0.5")  # Strongly discourage buys
        elif skew > 0.6:
            return Decimal("-0.3")  # Discourage buys
        elif skew < 0.2:
            return Decimal("0.5")   # Strongly encourage buys
        elif skew < 0.4:
            return Decimal("0.3")   # Encourage buys
        else:
            return Decimal("0.0")   # Neutral position

    def calculate_dynamic_order_size(self, atr: Decimal) -> Decimal:
        """Calculate order size based on volatility and trend strength"""
        # Base volatility adjustment
        if atr > self.volatility_threshold_high:
            vol_size_factor = Decimal("0.5")  # Reduce size in high volatility
        elif atr < self.volatility_threshold_low:
            vol_size_factor = Decimal("1.5")  # Increase size in low volatility
        else:
            vol_size_factor = Decimal("1.0")  # Default size
            
        # Trend strength adjustment
        trend_strength_factor = Decimal("1.0")
        if self.trend_state in [TrendState.STRONG_UPTREND, TrendState.STRONG_DOWNTREND]:
            trend_strength_factor = Decimal("1.2")  # Larger orders in strong trends
        elif self.trend_state in [TrendState.WEAK_UPTREND, TrendState.WEAK_DOWNTREND]:
            trend_strength_factor = Decimal("1.1")  # Slightly larger orders in weak trends
            
        return self.base_order_size * vol_size_factor * trend_strength_factor

    def calculate_ema(self, period: int) -> Decimal:
        """Calculate Exponential Moving Average"""
        prices = list(self.price_history)[-period:]
        if not prices:
            return Decimal("0")
            
        alpha = 2 / (period + 1)
        ema = prices[0]
        
        for price in prices[1:]:
            ema = alpha * price + (1 - alpha) * ema
            
        return Decimal(str(ema))

    def calculate_rsi(self, period: int) -> Decimal:
        """Calculate Relative Strength Index"""
        if len(self.price_history) <= period:
            return Decimal("50")  # Default to neutral if not enough data
            
        prices = list(self.price_history)[-period-1:]
        deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        
        gains = [delta if delta > 0 else 0 for delta in deltas]
        losses = [-delta if delta < 0 else 0 for delta in deltas]
        
        avg_gain = sum(gains) / period
        avg_loss = sum(losses) / period
        
        if avg_loss == 0:
            return Decimal("100")
            
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return Decimal(str(rsi))

    def calculate_macd(self) -> Tuple[Decimal, Decimal, Decimal]:
        """Calculate MACD, Signal Line and Histogram"""
        fast_ema = self.calculate_ema(self.macd_fast)
        slow_ema = self.calculate_ema(self.macd_slow)
        
        macd_line = fast_ema - slow_ema
        
        # Signal line is the EMA of the MACD line
        # For simplicity, we'll approximate it here
        macd_values = []
        for i in range(self.macd_signal):
            fast_ema = self.calculate_ema(self.macd_fast)
            slow_ema = self.calculate_ema(self.macd_slow)
            macd_values.append(float(fast_ema - slow_ema))
            
        alpha = 2 / (self.macd_signal + 1)
        signal_line_value = macd_values[0]
        
        for value in macd_values[1:]:
            signal_line_value = alpha * value + (1 - alpha) * signal_line_value
            
        signal_line = Decimal(str(signal_line_value))
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram

    def calculate_volume_weighted_price_direction(self) -> float:
        """Calculate volume-weighted price direction"""
        if len(self.price_history) < 2 or len(self.volume_history) < 1:
            return 0
            
        prices = list(self.price_history)[-len(self.volume_history):]
        volumes = list(self.volume_history)
        
        direction = 0
        total_volume = sum(volumes)
        
        if total_volume == 0:
            return 0
            
        for i in range(1, len(prices)):
            price_change = prices[i] - prices[i-1]
            vol_weight = volumes[i-1] / total_volume if i < len(volumes) else 0
            direction += price_change * vol_weight
            
        return direction

    def calculate_price_momentum(self) -> float:
        """Calculate recent price momentum"""
        if len(self.price_history) < 5:
            return 0
            
        recent_prices = list(self.price_history)[-5:]
        return recent_prices[-1] - recent_prices[0]

    def place_order(self, side: TradeType, price: Decimal, amount: Decimal):
        try:
            order = OrderCandidate(
                trading_pair=self.trading_pair,
                is_maker=True,
                order_type=OrderType.LIMIT,
                order_side=side,
                amount=amount,
                price=price
            )
            if side == TradeType.BUY:
                self.buy(self.exchange, self.trading_pair, order.amount, order.order_type, order.price)
            else:
                self.sell(self.exchange, self.trading_pair, order.amount, order.order_type, order.price)
        except Exception as e:
            self.logger().error(f"Order placement failed: {e}")

    def cancel_all_orders(self):
        try:
            for order in self.get_active_orders(self.exchange):
                self.logger().info(f"Cancelling order {order.client_order_id}")
                self.cancel(self.exchange, order.trading_pair, order.client_order_id)
        except Exception as e:
            self.logger().error(f"Order cancellation failed: {e}")

    def did_fill_order(self, event: OrderFilledEvent):
        # Log the fill and update the P&L and net position
        msg = f"{event.trade_type.name} fill: {event.amount} {event.trading_pair} at {event.price}"
        self.log_with_clock(logging.INFO, msg)
        self.notify_hb_app_with_timestamp(msg)

        # Update P&L based on the fill event.
        fill_value = event.amount * event.price
        if event.trade_type == TradeType.BUY:
            self.realized_pnl -= fill_value
            self.net_position += event.amount
        elif event.trade_type == TradeType.SELL:
            self.realized_pnl += fill_value
            self.net_position -= event.amount

        # Update maximum reached P&L (for drawdown monitoring)
        if self.realized_pnl > self.max_realized_pnl:
            self.max_realized_pnl = self.realized_pnl

        self.logger().info(
            f"Updated PnL: {self.realized_pnl}, Net Position: {self.net_position}, Max PnL: {self.max_realized_pnl}"
        )

    def calculate_atr(self):
        if len(self.price_history) < 2:
            return Decimal("0.0001")
        diffs = [
            abs(self.price_history[i] - self.price_history[i - 1])
            for i in range(1, len(self.price_history))
        ]
        return Decimal(str(statistics.mean(diffs)))

    def calculate_sma(self):
        if not self.price_history:
            return Decimal("0")
        return Decimal(str(statistics.mean(self.price_history)))

    def inventory_skew(self):
        # Returns a value between 0 and 1 indicating the proportion of base asset relative to total asset value.
        try:
            base, quote = self.trading_pair.split("-")
            base_balance = self.connectors[self.exchange].get_available_balance(base)
            quote_balance = self.connectors[self.exchange].get_available_balance(quote)

            mid_price = self.connectors[self.exchange].get_mid_price(self.trading_pair)
            if mid_price is None or mid_price == Decimal("0"):
                return 0.5  # Neutral if price is unavailable

            base_value = base_balance * mid_price
            total_value = base_value + quote_balance
            if total_value == Decimal("0"):
                return 0.5
            return float(base_value / total_value)
        except Exception as e:
            self.logger().error(f"Inventory skew calculation failed: {e}")
            return 0.5

    def format_status(self) -> str:
        lines = ["Strategy: Advanced Market Making with Multi-Factor Trend Detection"]
        lines.append(f"Pair: {self.trading_pair}")
        try:
            mid_price = self.connectors[self.exchange].get_mid_price(self.trading_pair)
            if mid_price:
                lines.append(f"Mid Price: {mid_price:.10f}")
            lines.append(f"Data Points: {len(self.price_history)}/{self.lookback_period}")
            if len(self.price_history) >= self.lookback_period:
                lines.append(f"SMA: {self.calculate_sma():.10f}")
                lines.append(f"ATR: {self.calculate_atr():.10f}")
                
                # Technical indicators
                if len(self.price_history) >= max(self.long_ema_period, self.rsi_period):
                    short_ema = self.calculate_ema(self.short_ema_period)
                    long_ema = self.calculate_ema(self.long_ema_period)
                    rsi = self.calculate_rsi(self.rsi_period)
                    macd, signal, histogram = self.calculate_macd()
                    
                    lines.append(f"EMA ({self.short_ema_period}/{self.long_ema_period}): {short_ema:.10f}/{long_ema:.10f}")
                    lines.append(f"RSI ({self.rsi_period}): {rsi:.2f}")
                    lines.append(f"MACD: {macd:.10f}, Signal: {signal:.10f}, Hist: {histogram:.10f}")
                
                lines.append(f"Trend State: {self.trend_state.name}")
                lines.append(f"Inventory Skew: {self.inventory_skew():.2f}")
            
            active_orders = self.get_active_orders(self.exchange)
            lines.append(f"Active Orders ({len(active_orders)}):")
            for order in active_orders:
                lines.append(f"  {order.trading_pair} {order.order_side.name} {order.amount} @ {order.price}")
            
            lines.append(f"Realized PnL: {self.realized_pnl:.2f}")
            lines.append(f"Net Position: {self.net_position:.2f}")
            lines.append(f"Max Realized PnL: {self.max_realized_pnl:.2f}")
        except Exception as e:
            lines.append(f"Status error: {e}")
        return "\n".join(lines)