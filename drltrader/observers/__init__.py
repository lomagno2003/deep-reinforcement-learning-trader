class Sides:
    Buy = 'buy'
    Sell = 'sell'


class Order:
    def __init__(self,
                 symbol: str,
                 qty: float,
                 price: float,
                 side: Sides):
        self.symbol = symbol
        self.qty = qty,
        self.price = price
        self.side = side


class Observer:
    def notify_order(self, order: Order):
        pass
