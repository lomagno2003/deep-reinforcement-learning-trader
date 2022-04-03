from drltrader.envs.observers import EnvObserver


class PrintEnvObserver(EnvObserver):
    def notify_stock_buy(self, symbol, qty, price):
        print(f"The stock {symbol} was bought")

    def notify_stock_sell(self, symbol, qty, price):
        print(f"The stock {symbol} was sold")
