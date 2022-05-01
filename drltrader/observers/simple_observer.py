from drltrader.observers import Observer, Order


class PrintEnvObserver(Observer):
    def notify_new_data(self):
        print(f"There's new data available")

    def notify_order(self, order: Order):
        print(f"There was an order to {order.side} {order.qty} stocks of {order.symbol}")

    def notify_portfolio_change(self, portfolio: dict):
        print(f"There was a portfolio change: {str(portfolio)}")


class CallbackObserver(Observer):
    def __init__(self,
                 new_data_callback_function,
                 order_callback_function,
                 portfolio_callback_function):
        self._new_data_callback_function = new_data_callback_function
        self._order_callback_function = order_callback_function
        self._portfolio_callback_function = portfolio_callback_function

    def notify_new_data(self):
        self._new_data_callback_function

    def notify_order(self, order: Order):
        self._callback_function(order)

    def notify_portfolio_change(self, portfolio: dict):
        self._portfolio_callback_function(portfolio)


class CompositeObserver(Observer):
    def __init__(self, observers: list):
        self._observers = observers

    def notify_new_data(self):
        for observer in self._observers:
            observer.notify_new_data()

    def notify_order(self, order: Order):
        for observer in self._observers:
            observer.notify_order(order)

    def notify_portfolio_change(self, portfolio: dict):
        for observer in self._observers:
            observer.notify_portfolio_change(portfolio)
