# matching_engine.py
import random

class MatchingEngineSimulator:
    def __init__(self, book=None, partial_fill_prob=0.2, cancel_prob=0.05):
        self.book = book
        self.partial_fill_prob = partial_fill_prob
        self.cancel_prob = cancel_prob

    def execute(self, order):
        """
        order: dict {symbol, side, qty, price, timestamp}
        Returns execution result(s): list of dicts {"symbol","qty","price","side","status"}
        """
        r = random.random()
        if r < self.cancel_prob:
            return [{"status":"canceled", **order}]
        elif r < self.cancel_prob + self.partial_fill_prob:
            # partial fill
            filled = max(1, int(order["qty"] * random.uniform(0.1, 0.9)))
            remain = order["qty"] - filled
            return [{"status":"partial", "symbol":order["symbol"], "qty":filled, "price":order["price"], "side":order["side"]},
                    {"status":"remaining", "symbol":order["symbol"], "qty":remain, "price":order["price"], "side":order["side"]}]
        else:
            # full fill
            return [{"status":"filled", "symbol":order["symbol"], "qty":order["qty"], "price":order["price"], "side":order["side"]}]
