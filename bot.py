"""
Hibachi BTC Market Making Bot
Based on HL-market-making strategy: 5-tier spreads, ATR volatility, inventory skew
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from decimal import Decimal, ROUND_DOWN, ROUND_HALF_UP

from hibachi_xyz import (
    HibachiApiClient, CreateOrder, CancelOrder, Side, OrderFlags,
)
from hibachi_xyz.env_setup import setup_environment
from hibachi_xyz.types import Interval

logger = logging.getLogger("hibachi_mm")


# ─── Data Classes ─────────────────────────────────────────────

@dataclass
class TrackedOrder:
    order_id: int
    nonce: int
    side: str       # "BUY" or "SELL"
    price: float
    quantity: float
    tier: int       # 0-4
    placed_at: float  # timestamp


@dataclass
class PnLTracker:
    total_buys: int = 0
    total_sells: int = 0
    total_buy_notional: Decimal = Decimal("0")
    total_sell_notional: Decimal = Decimal("0")

    @property
    def realized_pnl(self) -> Decimal:
        return self.total_sell_notional - self.total_buy_notional

    @property
    def completed_cycles(self) -> int:
        return min(self.total_buys, self.total_sells)

    def record_fill(self, side: str, notional: Decimal):
        if side == "BUY":
            self.total_buys += 1
            self.total_buy_notional += notional
        else:
            self.total_sells += 1
            self.total_sell_notional += notional

    def summary(self) -> str:
        return (
            f"PnL: ${self.realized_pnl:.4f} | "
            f"Cycles: {self.completed_cycles} | "
            f"Buys: {self.total_buys} | Sells: {self.total_sells}"
        )


# ─── Market Making Bot ───────────────────────────────────────

class GridBot:
    """Market making bot with 5-tier spreads, ATR volatility adjustment, and inventory skew."""

    def __init__(self, config: dict, dry_run: bool = False):
        self.config = config
        self.dry_run = dry_run
        self.symbol: str = config["symbol"]

        # Order params
        o = config["order"]
        self.order_size_usd: float = o["order_size_usd"]
        self.max_fees_pct: float = o["max_fees_percent"]
        self.use_post_only: bool = o.get("use_post_only", True)
        self.max_open_orders: int = o.get("max_open_orders", 50)
        self.order_expiry_min: int = o.get("order_expiry_minutes", 15)

        # Spread params (5 tiers)
        sp = config["spread"]
        self.buy_spreads: list[float] = [s / 100 for s in sp["buy_spreads"]]   # % → decimal
        self.sell_spreads: list[float] = [s / 100 for s in sp["sell_spreads"]]
        self.order_ratios: list[float] = sp["order_ratios"]

        # Volatility params
        v = config["volatility"]
        self.atr_interval: str = v["atr_interval"]
        self.atr_period: int = v["atr_period"]
        self.base_spread: float = v["base_spread"]
        self.vol_mult_min: float = v["vol_multiplier_min"]
        self.vol_mult_max: float = v["vol_multiplier_max"]

        # Inventory skew
        inv = config["inventory"]
        self.skew_multiplier: float = inv["skew_multiplier"]
        self.max_position_usd: float = inv["max_position_usd"]

        # Safety
        s = config["safety"]
        self.max_position_btc: float = s["max_position_btc"]
        self.max_unrealized_loss: float = s["max_unrealized_loss_usd"]
        self.max_balance_usage_pct: float = s["max_balance_usage_pct"] / 100

        # Timing
        t = config["timing"]
        self.check_interval: int = t["check_interval"]
        self.pnl_report_interval: int = t.get("pnl_report_interval", 300)

        # State
        self.client: HibachiApiClient | None = None
        self.tracked_orders: dict[int, TrackedOrder] = {}  # order_id -> TrackedOrder
        self.pnl = PnLTracker()
        self._initialized = False
        self._killed = False
        self._initial_balance: float = 0
        self._usable_balance: float = 0

        # Interval mapping for SDK
        self._interval_map = {
            "1m": Interval.ONE_MINUTE,
            "5m": Interval.FIVE_MINUTES,
            "15m": Interval.FIFTEEN_MINUTES,
            "1h": Interval.ONE_HOUR,
            "4h": Interval.FOUR_HOURS,
            "1d": Interval.ONE_DAY,
        }

    # ─── Initialization ───────────────────────────────────────

    async def initialize(self):
        logger.info("Initializing Hibachi API client...")

        api_endpoint, data_api_endpoint, api_key, account_id, private_key, public_key, _ = setup_environment()

        self.client = HibachiApiClient(
            api_url=api_endpoint,
            data_api_url=data_api_endpoint,
            api_key=api_key,
            account_id=account_id,
            private_key=private_key,
        )

        # Must call first to cache contract metadata
        exch_info = self.client.get_exchange_info()
        logger.info(f"Exchange info loaded. Status: {exch_info.status}")

        # Account
        account = self.client.get_account_info()
        self._initial_balance = float(account.balance)
        self._usable_balance = self._initial_balance * self.max_balance_usage_pct
        logger.info(f"Balance: ${self._initial_balance:.2f} (usable: ${self._usable_balance:.2f})")

        # Existing positions
        if account.positions:
            for pos in account.positions:
                if pos.symbol == self.symbol:
                    logger.info(f"Existing position: direction='{pos.direction}' qty={pos.quantity} @ ${pos.openPrice}")

        # Current price
        prices = self.client.get_prices(self.symbol)
        mid = float(prices.markPrice)
        logger.info(f"{self.symbol} mark price: ${mid:,.2f}")

        if self.dry_run:
            logger.info("[DRY-RUN] Simulation mode - no real orders")

        self._initialized = True

    # ─── Market Data ──────────────────────────────────────────

    def _get_mid_price(self) -> float | None:
        try:
            prices = self.client.get_prices(self.symbol)
            return float(prices.markPrice)
        except Exception as e:
            logger.error(f"get_mid_price failed: {e}")
            return None

    def _get_position(self) -> dict:
        """Returns {'size': float, 'unrealized_pnl': float, 'entry_price': float}"""
        try:
            account = self.client.get_account_info()
            for pos in (account.positions or []):
                if pos.symbol == self.symbol:
                    size = float(pos.quantity)
                    direction_raw = str(getattr(pos, 'direction', 'UNKNOWN'))
                    logger.debug(f"Raw position: direction={direction_raw}, qty={pos.quantity}, symbol={pos.symbol}")
                    if direction_raw.upper() == "SHORT":
                        size = -size
                    return {
                        "size": size,
                        "entry_price": float(pos.openPrice),
                        "unrealized_pnl": float(pos.unrealizedTradingPnl),
                    }
        except Exception as e:
            logger.error(f"get_position failed: {e}")
        return {"size": 0, "entry_price": 0, "unrealized_pnl": 0}

    # ─── ATR Volatility ──────────────────────────────────────

    def _calculate_volatility_multiplier(self, mid_price: float) -> float:
        """Calculate spread multiplier from ATR. High vol = wider spreads."""
        try:
            interval = self._interval_map.get(self.atr_interval, Interval.FIVE_MINUTES)
            klines_resp = self.client.get_klines(self.symbol, interval)
            candles = klines_resp.klines
            if not candles or len(candles) < self.atr_period + 1:
                return 1.0

            # Calculate True Range
            tr_list = []
            for i in range(1, len(candles)):
                high = float(candles[i].high)
                low = float(candles[i].low)
                prev_close = float(candles[i - 1].close)
                tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
                tr_list.append(tr)

            # Wilder's EMA for ATR
            atr = sum(tr_list[:self.atr_period]) / self.atr_period
            for tr in tr_list[self.atr_period:]:
                atr = (tr / self.atr_period) + (atr * (1 - 1 / self.atr_period))

            if mid_price <= 0:
                return 1.0

            vol_mult = (atr / mid_price) / self.base_spread
            return max(self.vol_mult_min, min(self.vol_mult_max, vol_mult))

        except Exception as e:
            logger.error(f"ATR calculation failed: {e}")
            return 1.0

    # ─── Inventory Skew ──────────────────────────────────────

    def _calculate_adjusted_spreads(
        self, pos_ratio: float, vol_mult: float
    ) -> tuple[list[float], list[float]]:
        """Adjust spreads based on position imbalance and volatility.

        When LONG-heavy (pos_ratio > 0):
          - BUY spreads widen  → harder to accumulate more longs
          - SELL spreads tighten → easier to reduce position

        When SHORT-heavy (pos_ratio < 0): opposite.
        """
        adj = pos_ratio * self.skew_multiplier
        buy_sp = [max(0.0001, s * (1 + adj) * vol_mult) for s in self.buy_spreads]
        sell_sp = [max(0.0001, s * (1 - adj) * vol_mult) for s in self.sell_spreads]
        return buy_sp, sell_sp

    # ─── Order Placement ─────────────────────────────────────

    def _get_open_order_exposure(self) -> tuple[float, float]:
        """Calculate total BTC exposure from open orders.
        Returns (buy_exposure, sell_exposure) both as positive values."""
        buy_exp = sum(t.quantity for t in self.tracked_orders.values() if t.side == "BUY")
        sell_exp = sum(t.quantity for t in self.tracked_orders.values() if t.side == "SELL")
        return buy_exp, sell_exp

    async def _place_tier_orders(
        self,
        side: str,
        mid: float,
        spreads: list[float],
        open_count: int,
        pos_value: float,
    ) -> int:
        """Place 5-tier orders for one side. Returns number of orders placed."""
        num_tiers = len(self.order_ratios)
        side_name = "BUY" if side == "BUY" else "SELL"

        # Check slots
        if open_count + num_tiers > self.max_open_orders:
            logger.debug(f"Skip {side_name} (need {num_tiers} slots, {self.max_open_orders - open_count} available)")
            return 0

        # Position limits (USD)
        if side == "BUY" and pos_value >= self.max_position_usd:
            logger.info(f"Skip {side_name} (position ${pos_value:,.0f} >= limit ${self.max_position_usd:,.0f})")
            return 0
        if side == "SELL" and pos_value <= -self.max_position_usd:
            logger.info(f"Skip {side_name} (position ${pos_value:,.0f} <= limit -${self.max_position_usd:,.0f})")
            return 0

        # BTC position limit — include open order exposure
        pos = self._get_position()
        pos_btc = pos["size"]
        buy_exp, sell_exp = self._get_open_order_exposure()

        # Effective exposure = current position + open orders that could fill
        if side == "BUY":
            effective = pos_btc + buy_exp  # if all open BUYs fill, this is max long
            if effective >= self.max_position_btc:
                logger.info(
                    f"[SAFETY] Skip {side_name} (pos {pos_btc:.4f} + open_buy {buy_exp:.4f} "
                    f"= {effective:.4f} >= {self.max_position_btc} BTC)"
                )
                return 0
        if side == "SELL":
            effective = pos_btc - sell_exp  # if all open SELLs fill, this is max short
            if effective <= -self.max_position_btc:
                logger.info(
                    f"[SAFETY] Skip {side_name} (pos {pos_btc:.4f} - open_sell {sell_exp:.4f} "
                    f"= {effective:.4f} <= -{self.max_position_btc} BTC)"
                )
                return 0

        if self.dry_run:
            for ratio, spread in zip(self.order_ratios, spreads):
                usd = self.order_size_usd * ratio
                if side == "BUY":
                    price = mid * (1 - spread)
                else:
                    price = mid * (1 + spread)
                qty = usd / price
                logger.info(f"  [DRY-RUN] {side_name} {qty:.4f} BTC @ ${price:,.2f} ({spread*100:.2f}%)")
            return num_tiers

        # Build orders — skip tiers with quantity < 0.0001 BTC
        MIN_QTY = 0.0001
        orders = []
        tier_indices = []  # track which tier each order belongs to
        order_flags = OrderFlags.PostOnly if self.use_post_only else None
        for i, (ratio, spread) in enumerate(zip(self.order_ratios, spreads)):
            usd = self.order_size_usd * ratio
            if side == "BUY":
                price = round(mid * (1 - spread), 2)
            else:
                price = round(mid * (1 + spread), 2)
            qty = round(usd / price, 4)

            if qty < MIN_QTY:
                logger.debug(f"  Skip tier {i+1}: ${usd:.1f} → {qty} BTC < {MIN_QTY}")
                continue

            orders.append(
                CreateOrder(
                    symbol=self.symbol,
                    side=Side.BUY if side == "BUY" else Side.SELL,
                    quantity=qty,
                    max_fees_percent=self.max_fees_pct,
                    price=price,
                    order_flags=order_flags,
                )
            )
            tier_indices.append(i)

        if not orders:
            logger.warning(f"No valid {side_name} orders (all tiers below {MIN_QTY} BTC)")
            return 0

        try:
            response = self.client.batch_orders(orders)
            placed = 0
            now = time.time()
            spread_log = []
            for i, result in enumerate(response.orders):
                tier_idx = tier_indices[i]
                if hasattr(result, "orderId"):
                    tracked = TrackedOrder(
                        order_id=result.orderId,
                        nonce=result.nonce,
                        side=side,
                        price=float(orders[i].price),
                        quantity=float(orders[i].quantity),
                        tier=tier_idx,
                        placed_at=now,
                    )
                    self.tracked_orders[result.orderId] = tracked
                    placed += 1
                    sign = "-" if side == "BUY" else "+"
                    spread_log.append(f"{sign}{spreads[tier_idx]*100:.2f}%@{int(orders[i].price):,}")
                elif hasattr(result, "errorCode"):
                    logger.warning(f"  Tier {tier_idx+1} error: {result.errorCode} - {result.message}")

            if spread_log:
                logger.info(f"[ORDER] {side_name}({placed}/{num_tiers}): {'  '.join(spread_log)}")
            return placed

        except Exception as e:
            logger.error(f"{side_name} batch failed: {e}", exc_info=True)
            return 0

    # ─── Order Cancellation ──────────────────────────────────

    async def _cancel_expired_orders(self) -> tuple[int, int]:
        """Cancel orders older than expiry. Returns (buy_cancelled, sell_cancelled)."""
        if self.dry_run:
            return 0, 0

        try:
            pending = self.client.get_pending_orders()
            if not pending.orders:
                return 0, 0

            now = time.time()
            to_cancel = []
            for order in pending.orders:
                # Check if tracked and expired
                tracked = self.tracked_orders.get(order.orderId)
                if tracked and (now - tracked.placed_at) > self.order_expiry_min * 60:
                    to_cancel.append(order)

            if not to_cancel:
                return 0, 0

            cancel_ops = [CancelOrder(order_id=o.orderId) for o in to_cancel]
            self.client.batch_orders(cancel_ops)

            buy_c = sell_c = 0
            for order in to_cancel:
                tracked = self.tracked_orders.pop(order.orderId, None)
                if tracked:
                    if tracked.side == "BUY":
                        buy_c += 1
                    else:
                        sell_c += 1

            if buy_c or sell_c:
                logger.info(f"[CANCEL] Expired: {buy_c} BUY, {sell_c} SELL (>{self.order_expiry_min}min)")
            return buy_c, sell_c

        except Exception as e:
            logger.error(f"Cancel expired failed: {e}", exc_info=True)
            return 0, 0

    async def _cancel_all_orders(self):
        """Cancel ALL open orders."""
        if self.dry_run:
            logger.info("[DRY-RUN] Would cancel all orders")
            return

        try:
            pending = self.client.get_pending_orders()
            if not pending.orders:
                return

            cancel_ops = [CancelOrder(order_id=o.orderId) for o in pending.orders]
            logger.info(f"Cancelling {len(cancel_ops)} orders...")
            self.client.batch_orders(cancel_ops)
            self.tracked_orders.clear()
            logger.info("All orders cancelled")
        except Exception as e:
            logger.error(f"Cancel all failed: {e}", exc_info=True)

    # ─── Fill Detection ──────────────────────────────────────

    async def _detect_fills(self):
        """Detect filled orders by comparing tracked vs pending."""
        if self.dry_run:
            return

        try:
            pending = self.client.get_pending_orders()
            pending_ids = {o.orderId for o in pending.orders} if pending.orders else set()

            for oid in list(self.tracked_orders.keys()):
                if oid not in pending_ids:
                    tracked = self.tracked_orders.pop(oid)
                    notional = Decimal(str(tracked.price)) * Decimal(str(tracked.quantity))
                    self.pnl.record_fill(tracked.side, notional)
                    logger.info(
                        f"FILL: {tracked.side} {tracked.quantity:.4f} BTC "
                        f"@ ${tracked.price:,.2f} (${float(notional):,.2f})"
                    )
        except Exception as e:
            logger.error(f"Fill detection failed: {e}", exc_info=True)

    # ─── Kill Switch ─────────────────────────────────────────

    async def _check_kill_switch(self) -> bool:
        if self._killed:
            return True
        pos = self._get_position()
        if pos["unrealized_pnl"] < -self.max_unrealized_loss:
            logger.critical(
                f"[KILL-SWITCH] Unrealized loss ${pos['unrealized_pnl']:.2f} "
                f"exceeds -${self.max_unrealized_loss}!"
            )
            self._killed = True
            await self._cancel_all_orders()
            return True
        return False

    # ─── Single Iteration ────────────────────────────────────

    async def _run_iteration(self):
        """One cycle: get data → detect fills → place orders → cancel expired."""

        # 1. Mid price
        mid = self._get_mid_price()
        if not mid:
            logger.warning("No mid price data, skipping iteration")
            return

        # 2. Volatility multiplier
        vol_mult = self._calculate_volatility_multiplier(mid)

        # 3. Position
        position = self._get_position()
        pos_value = position["size"] * mid
        pos_ratio = pos_value / self.max_position_usd if self.max_position_usd > 0 else 0
        inv_adj = pos_ratio * self.skew_multiplier

        pos_status = "Neutral"
        if abs(position["size"]) > 0.0001:
            pos_status = "Long" if position["size"] > 0 else "Short"

        # 4. Detect fills from last cycle
        await self._detect_fills()

        # 5. Cancel ALL existing orders before placing fresh ones
        #    This prevents order accumulation across cycles
        await self._cancel_all_orders()

        # 6. Log status
        logger.info(f"{'='*55}")
        logger.info(
            f"{self.symbol} | Mid: ${mid:,.0f} | Vol: {vol_mult:.2f}x | "
            f"Skew: {inv_adj:+.2f} ({pos_ratio:+.1%})"
        )
        logger.info(
            f"Pos: {position['size']:.4f} BTC (${abs(pos_value):,.0f}) {pos_status} | "
            f"Entry: ${position['entry_price']:,.0f} | "
            f"PnL: ${position['unrealized_pnl']:+.2f}"
        )

        # 7. Calculate adjusted spreads
        buy_spreads, sell_spreads = self._calculate_adjusted_spreads(pos_ratio, vol_mult)

        # 8. Place BUY orders (fresh, no existing orders)
        await self._place_tier_orders("BUY", mid, buy_spreads, 0, pos_value)

        # 9. Place SELL orders (fresh, no existing orders)
        await self._place_tier_orders("SELL", mid, sell_spreads, 0, pos_value)

        # 10. Count orders after placement
        buy_cnt = sum(1 for t in self.tracked_orders.values() if t.side == "BUY")
        sell_cnt = sum(1 for t in self.tracked_orders.values() if t.side == "SELL")
        logger.info(f"[ORDERS] Open: {buy_cnt} buys, {sell_cnt} sells")

    # ─── Main Loop ────────────────────────────────────────────

    async def run(self, shutdown_event: asyncio.Event):
        if not self._initialized:
            raise RuntimeError("Call initialize() first")

        buy_sp_str = " / ".join([f"-{s*100:.2f}%" for s in self.buy_spreads])
        sell_sp_str = " / ".join([f"+{s*100:.2f}%" for s in self.sell_spreads])
        ratio_str = " / ".join([f"{r*100:.0f}%" for r in self.order_ratios])

        logger.info("=" * 55)
        logger.info(f"Hibachi Market Making Bot | {self.symbol}")
        logger.info(f"  Order size: ${self.order_size_usd} | Interval: {self.check_interval}s")
        logger.info(f"  Buy spreads:  {buy_sp_str}")
        logger.info(f"  Sell spreads: {sell_sp_str}")
        logger.info(f"  Ratios: {ratio_str}")
        logger.info(f"  Skew: {self.skew_multiplier}x | ATR: {self.atr_interval}/{self.atr_period}")
        logger.info(f"  Max position: {self.max_position_btc} BTC / ${self.max_position_usd:,}")
        logger.info(f"  Kill-switch: -${self.max_unrealized_loss}")
        logger.info(f"  Balance: ${self._usable_balance:.2f} / ${self._initial_balance:.2f}")
        logger.info(f"  Dry-run: {self.dry_run}")
        logger.info("=" * 55)

        last_pnl_report = time.time()

        while not shutdown_event.is_set():
            try:
                # Kill switch
                if not self.dry_run and await self._check_kill_switch():
                    logger.critical("[KILL-SWITCH] Bot stopped.")
                    break

                # Run one iteration
                await self._run_iteration()

                # PnL report
                now = time.time()
                if now - last_pnl_report >= self.pnl_report_interval:
                    pos = self._get_position()
                    logger.info(f"[REPORT] {self.pnl.summary()}")
                    logger.info(
                        f"[REPORT] Position: {pos['size']:.4f} BTC | "
                        f"Unrealized: ${pos['unrealized_pnl']:.2f}"
                    )
                    last_pnl_report = now

                # Wait
                try:
                    await asyncio.wait_for(shutdown_event.wait(), timeout=self.check_interval)
                    break
                except asyncio.TimeoutError:
                    pass

            except Exception as e:
                logger.error(f"Main loop error: {e}", exc_info=True)
                await asyncio.sleep(5)

        logger.info("Main loop ended")

    # ─── Shutdown ─────────────────────────────────────────────

    async def shutdown(self):
        logger.info("Shutting down...")
        await self._cancel_all_orders()

        if not self.dry_run and self.client:
            pos = self._get_position()
            logger.info(f"[FINAL] Position: {pos['size']:.4f} BTC (unrealized: ${pos['unrealized_pnl']:.2f})")

        logger.info("=" * 55)
        logger.info(f"[FINAL] {self.pnl.summary()}")
        if self._killed:
            logger.info("[FINAL] Bot was stopped by KILL-SWITCH")
        logger.info("=" * 55)
