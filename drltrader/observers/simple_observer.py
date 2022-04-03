from drltrader.observers import Observer, Order


class PrintEnvObserver(Observer):
    def notify_order(self, order: Order):
        print(f"There was an order to {order.side} {order.qty} stocks of {order.symbol}")


class CallbackObserver(Observer):
    def __init__(self, callback_function):
        self._callback_function = callback_function

    def notify_order(self, order: Order):
        self._callback_function(order)


class CompositeObserver(Observer):
    def __init__(self, observers: list):
        self._observers = observers

    def notify_order(self, order: Order):
        for observer in self._observers:
            observer.notify_order(order)
