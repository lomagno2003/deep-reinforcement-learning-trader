from datetime import datetime, time
from datetime import timedelta
from dateutil.relativedelta import relativedelta, SA, MO
from pytz import timezone

from drltrader.data import Scenario


class Scenarios:
    @staticmethod
    def market_weeks_before_date(reference_datetime: datetime,
                                 symbols: list,
                                 interval: str,
                                 start_week: int = 0,
                                 end_week: int = 0):
        est_timezone = timezone('EST')
        market_open_hour = time(hour=9, minute=30)
        market_close_hour = time(hour=16, minute=00)

        end_date = datetime.combine(
            reference_datetime - timedelta(weeks=1) + relativedelta(weekday=SA(-1)) - timedelta(days=1) - timedelta(weeks=end_week),
            market_close_hour,
            est_timezone)
        start_date = datetime.combine(
            end_date + relativedelta(weekday=MO(-1)) - timedelta(weeks=start_week),
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
        return Scenarios.market_weeks_before_date(
            reference_datetime=datetime.now(),
            symbols=symbols,
            interval=interval
        )

    @staticmethod
    def last_market_weeks(symbols: list = None,
                          interval: str = None,
                          start_week: int = 0,
                          end_week: int = 0):
        return Scenarios.market_weeks_before_date(
            reference_datetime=datetime.now(),
            symbols=symbols,
            interval=interval,
            start_week=start_week,
            end_week=end_week
        )

    @staticmethod
    def last_months(months: int = 1,
                    symbols: list = None,
                    interval: str = None):
        est_timezone = timezone('EST')

        today = datetime.now(est_timezone)
        first = today.replace(day=1, hour=0, minute=0, second=0)
        end_of_last_month = first - timedelta(days=1)
        beginning_of_last_month = end_of_last_month.replace(day=1) - relativedelta(month=months - 1)

        return Scenario(
            symbols=symbols,
            start_date=beginning_of_last_month,
            end_date=end_of_last_month,
            interval=interval
        )

    @staticmethod
    def months_of_this_year(symbols: list = None,
                            interval: str = None,
                            start_month: int = 1,
                            end_month: int = 12):
        est_timezone = timezone('EST')

        start_date = datetime.now(est_timezone).replace(month=start_month, day=1, hour=0, minute=0, second=0)
        end_date = datetime.now(est_timezone).replace(month=end_month, day=1, hour=0, minute=0, second=0)

        return Scenario(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            interval=interval
        )
