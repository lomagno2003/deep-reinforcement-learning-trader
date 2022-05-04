from datetime import datetime


class Scenario:
    def __init__(self,
                 start_date: datetime,
                 end_date: datetime = None,
                 interval: str = None,
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

    def copy_with_interval(self, interval: str):
        return Scenario(start_date=self.start_date,
                        end_date=self.end_date,
                        interval=interval,
                        symbol=self.symbol,
                        symbols=self.symbols)

    @staticmethod
    def empty_scenario():
        return Scenario(symbols=['FOO'],
                        start_date=datetime.fromtimestamp(0),
                        end_date=datetime.fromtimestamp(0),
                        interval='5m')


class DataRepository:
    def retrieve_datas(self, scenario: Scenario):
        pass

    def get_columns_per_symbol(self):
        empty_dataframe = next(iter(self.retrieve_datas(Scenario.empty_scenario()).values()))

        return empty_dataframe.columns
