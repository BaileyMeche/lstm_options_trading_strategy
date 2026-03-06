# orderbook.py
import heapq
import itertools
from dataclasses import dataclass

@dataclass
class Order:
    id: int
    symbol: str
    side: str  # 'buy' or 'sell'
    price: float
    qty: int
    timestamp: float

class OrderBook:
    def __init__(self):
        self._bids = []  # max-heap via negatives (price, time, order)
        self._asks = []  # min-heap (price, time, order)
        self._id_counter = itertools.count(1)
        self._orders = {}  # id -> Order

    def add_order(self, symbol, side, price, qty, timestamp):
        oid = next(self._id_counter)
        o = Order(oid, symbol, side, price, qty, timestamp)
        entry = ( -price, timestamp, o ) if side == 'buy' else ( price, timestamp, o )
        if side == 'buy':
            heapq.heappush(self._bids, entry)
        else:
            heapq.heappush(self._asks, entry)
        self._orders[oid] = o
        return o

    def cancel_order(self, oid):
        # lazy removal: mark qty=0
        o = self._orders.get(oid)
        if o:
            o.qty = 0
            del self._orders[oid]
            return True
        return False

    def match_best(self):
        """
        Very simple matching: look at top bid and ask—if bid.price >= ask.price, produce a trade
        Return execution dict or None
        """
        while self._bids and self._bids[0][2].qty == 0:
            heapq.heappop(self._bids)
        while self._asks and self._asks[0][2].qty == 0:
            heapq.heappop(self._asks)

        if not self._bids or not self._asks:
            return None

        bid = self._bids[0][2]
        ask = self._asks[0][2]
        if bid.price >= ask.price:
            # match
            exec_qty = min(bid.qty, ask.qty)
            price = (bid.price + ask.price) / 2.0
            # reduce quantities
            bid.qty -= exec_qty
            ask.qty -= exec_qty
            if bid.qty == 0:
                heapq.heappop(self._bids)
                del self._orders[bid.id]
            if ask.qty == 0:
                heapq.heappop(self._asks)
                del self._orders[ask.id]
            return {"symbol": bid.symbol, "qty": exec_qty, "price": price, "buy_id": bid.id, "sell_id": ask.id}
        return None
