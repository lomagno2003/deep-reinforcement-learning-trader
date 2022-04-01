from datetime import datetime


class Scenario:
    def __init__(self,
                 symbol: str,
                 start_date: datetime,
                 end_date: datetime,
                 interval: str = '5m'):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.interval = interval

    def __str__(self):
        return f"{self.interval}" \
               f"_{self.symbol}" \
               f"_{self.start_date.strftime('%Y-%m-%d-%H-%M-%S')}" \
               f"_{self.end_date.strftime('%Y-%m-%d-%H-%M-%S')}"
