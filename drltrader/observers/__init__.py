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
    # TODO: This method makes low-sense
    def notify_new_data(self):
        pass

    def notify_order(self, order: Order):
        pass

    def notify_portfolio_change(self, portfolio: dict):
        pass

    def notify_begin_of_observation(self, portfolio: dict):
        pass
