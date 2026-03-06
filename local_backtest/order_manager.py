# order_manager.py
import json
from collections import deque, defaultdict
from datetime import datetime, timedelta

class OrderManager:
    def __init__(self, capital=100000, orders_per_minute_limit=60, per_asset_limit=10000, audit_file="orders.log"):
        self.capital = capital
        self.orders_per_minute_limit = orders_per_minute_limit
        self.per_asset_limit = per_asset_limit
        self.audit_file = audit_file
        self.order_timestamps = deque()
        self.net_exposure = defaultdict(int)  # symbol -> net position

    def can_place(self, order):
        """
        order: dict {symbol, side, qty, price, timestamp}
        checks:
          - orders per minute
          - per-asset limit if executed fully (approx)
          - capital sufficiency for buys (naive)
        """
        now = datetime.utcnow()
        # clean timestamps older than 60 sec
        while self.order_timestamps and (now - self.order_timestamps[0]) > timedelta(seconds=60):
            self.order_timestamps.popleft()

        if len(self.order_timestamps) >= self.orders_per_minute_limit:
            return False, "orders_per_minute_limit_exceeded"

        est_cost = order["qty"] * order["price"]
        if order["side"] == "buy" and est_cost > self.capital:
            return False, "insufficient_capital"

        # per asset limit check (exposure after order)
        current = self.net_exposure[order["symbol"]]
        new = current + order["qty"] if order["side"] == "buy" else current - order["qty"]
        if abs(new) > self.per_asset_limit:
            return False, "per_asset_limit_exceeded"

        return True, "ok"

    def record_order(self, order):
        # append timestamp, write to audit
        self.order_timestamps.append(datetime.utcnow())
        with open(self.audit_file, "a") as f:
            f.write(json.dumps({"ts": datetime.utcnow().isoformat(), **order}) + "\n")

    def on_fill(self, execution):
        # update exposure and capital
        symbol = execution["symbol"]
        qty = execution["qty"]
        price = execution["price"]
        # assume match has buy and sell sides: for this simplified model, we track net exposure
        self.net_exposure[symbol] += qty  # assume buy; for realism, you'd pass side
        # naive capital update:
        self.capital -= qty * price
