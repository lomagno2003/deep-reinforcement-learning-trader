

class EnvObserver:
    def notify_stock_buy(self, symbol):
        pass

    def notify_stock_sell(self, symbol):
        pass


class PrintEnvObserver(EnvObserver):
    def notify_stock_buy(self, symbol):
        print(f"The stock {symbol} was bought")

    def notify_stock_sell(self, symbol):
        print(f"The stock {symbol} was sold")
