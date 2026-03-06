# backtester.py
from gateway import Gateway
from strategy import MACrossoverStrategy
from order_manager import OrderManager
from matching_engine import MatchingEngineSimulator
import pandas as pd
import time

class Backtester:
    def __init__(self, market_csv, strategy, capital=100000):
        self.gateway = Gateway(market_csv)
        self.strategy = strategy
        self.order_manager = OrderManager(capital=capital)
        self.matching = MatchingEngineSimulator()
        self.trades = []  # executed trade records
        self.equity_curve = []

    def on_tick(self, timestamp, row):
        # feed row into strategy
        # build hist up to this timestamp
        # Gateway's df is index-based
        hist = self.gateway.df.loc[:timestamp]
        signal = self.strategy.generate_signal(row, hist)
        # create order if signal exists
        if signal:
            order = {"symbol":"SYMBOL", "side": "buy" if signal["action"]=="buy" else "sell",
                     "qty": signal["size"], "price": row["Close"], "timestamp": timestamp.isoformat()}
            ok, reason = self.order_manager.can_place(order)
            if not ok:
                # log rejection
                print("Order rejected:", reason)
            else:
                self.order_manager.record_order(order)
                executions = self.matching.execute(order)
                for ex in executions:
                    if ex["status"] in ("filled", "partial"):
                        exec_rec = {"timestamp": timestamp, "symbol": ex["symbol"], "qty": ex["qty"], "price": ex["price"], "side": ex["side"]}
                        self.trades.append(exec_rec)
                        self.order_manager.on_fill(exec_rec)
                        self.strategy.on_fill(exec_rec)
        # snapshot equity (naive): capital + mark-to-market
        # very simple: equity = capital + sum(position * last_price)
        total_pos_value = 0
        for sym, pos in self.order_manager.net_exposure.items():
            last_price = row["Close"]  # naive assume single symbol
            total_pos_value += pos * last_price
        equity = self.order_manager.capital + total_pos_value
        self.equity_curve.append((timestamp, equity))

    def run(self):
        self.gateway.stream(self.on_tick)
