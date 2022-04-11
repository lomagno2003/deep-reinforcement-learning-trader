from datetime import datetime


class Scenario:
    def __init__(self,
                 start_date: datetime,
                 end_date: datetime = None,
                 interval: str = '5m',
                 symbol: str = None,
                 symbols: list = None):
        if symbol is None and symbols is None:
            raise ValueError("Either symbol or symbols needs to be provided")
        if symbol is not None and symbols is not None:
            raise ValueError("Either symbol or symbols needs to be provided, not both")

        self.symbol = symbol
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.interval = interval

    def __str__(self):
        if self.symbol is not None:
            symbols_space = f"{self.symbol}"
        else:
            symbols_space = "_".join(self.symbols)

        return f"{self.interval}" \
               f"_{symbols_space}" \
               f"_{self.start_date.strftime('%Y-%m-%d-%H-%M-%S')}" \
               f"_{self.end_date.strftime('%Y-%m-%d-%H-%M-%S')}"

    def clone_with_symbol(self, symbol):
        return Scenario(start_date=self.start_date,
                        end_date=self.end_date,
                        interval=self.interval,
                        symbol=symbol)

    def copy_with_end_date(self, end_date: datetime):
        return Scenario(start_date=self.start_date,
                        end_date=end_date,
                        interval=self.interval,
                        symbol=self.symbol,
                        symbols=self.symbols)


class DataRepository:
    def retrieve_datas(self, scenario: Scenario):
        pass
