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
