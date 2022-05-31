from datetime import datetime, time
from datetime import timedelta
from dateutil.relativedelta import relativedelta, SA, MO
from pytz import timezone

from drltrader.data import Scenario


class Scenarios:
    @staticmethod
    def market_week_before_date(reference_datetime: datetime, symbols: list, interval: str):
        est_timezone = timezone('EST')
        market_open_hour = time(hour=9, minute=30)
        market_close_hour = time(hour=16, minute=00)

        end_date = datetime.combine(
            reference_datetime - timedelta(weeks=1) + relativedelta(weekday=SA(-1)) - timedelta(days=1),
            market_close_hour,
            est_timezone)
        start_date = datetime.combine(
            end_date + relativedelta(weekday=MO(-1)),
            market_open_hour,
            est_timezone)

        return Scenario(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            interval=interval
        )

    @staticmethod
    def last_market_week(symbols: list = None, interval: str = None):
        return Scenarios.market_week_before_date(
            reference_datetime=datetime.now(),
            symbols=symbols,
            interval=interval
        )
