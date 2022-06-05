import yfinance as yf
import pandas as pd
from datetime import datetime
from datetime import timedelta
from pytz import timezone
import logging
import logging.config

import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import TimeFrame, TimeFrameUnit
import json

from drltrader.data import DataRepository, Scenario

logging.config.fileConfig('log.ini', disable_existing_loggers=False)
logger = logging.getLogger(__name__)


class OHLCVDataRepository(DataRepository):
    def __init__(self,
                 cache_enabled: bool = True):
        self.indicator_column_names = None
        self._cache = {}
        self._cache_enabled = cache_enabled

    def get_repository_name(self):
        return "D"

    def retrieve_datas(self, scenario: Scenario):
        dataframe_per_symbol = {}

        intersection_index = None

        for symbol in scenario.symbols:
            dataframe_per_symbol[symbol] = self.retrieve_data(scenario.copy_with_symbol(symbol))

            if intersection_index is None:
                intersection_index = dataframe_per_symbol[symbol].index
            else:
                intersection_index = intersection_index.intersection(dataframe_per_symbol[symbol].index)

        for symbol in dataframe_per_symbol:
            dataframe_per_symbol[symbol] = dataframe_per_symbol[symbol].loc[intersection_index]

        return dataframe_per_symbol

    def retrieve_data(self, scenario: Scenario):
        if scenario.end_date is None:
            scenario = scenario.copy_with_end_date(datetime.now())

        if self._cache_enabled and str(scenario) in self._cache:
            logger.debug(f"Data for scenario {scenario} available on cache. Returning saved version...")
            return self._cache[str(scenario)]
        else:
            logger.info(f"Data for scenario {scenario} not available on cache. Fetching it...")
            df = self._fetch_data(scenario)

            if self._cache_enabled:
                self._cache[str(scenario)] = df

            df.index = df.index.tz_convert('GMT')

            return df

    def _fetch_data(self, scenario: Scenario):
        raise NotImplementedError()


class YahooOHLCVDataRepository(OHLCVDataRepository):
    def _fetch_data(self, scenario: Scenario):
        ticker = yf.Ticker(scenario.symbol)
        df = ticker.history(start=scenario.start_date,
                            end=scenario.end_date,
                            interval=scenario.interval)

        # FIXME: There's some weird bug on Yahoo
        df = df.iloc[:-1, :]

        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

        return df


class AlpacaOHLCVDataRepository(OHLCVDataRepository):
    def __init__(self,
                 cache_enabled: bool = True,
                 config_file_name: str = 'config.json'):
        super().__init__(cache_enabled=cache_enabled)

        with open(config_file_name) as config_file:
            config = json.load(config_file)

        self._alpaca_key = config['alpaca']['key']
        self._alpaca_secret = config['alpaca']['secret']
        self._alpaca_url = config['alpaca']['url']

        self._alpaca_api = tradeapi.REST(self._alpaca_key,
                                         self._alpaca_secret,
                                         self._alpaca_url,
                                         api_version='v2')

    def _fetch_data(self, scenario: Scenario):
        # FIXME: This needs to be done because Alpaca free account does not support real time data up to 15 min
        scenario = scenario.copy_with_end_date(scenario.end_date - timedelta(minutes=30))
        if scenario.start_date > scenario.end_date:
            scenario.start_date = scenario.end_date
        elif scenario.start_date == scenario.end_date:
            return pd.DataFrame(data=[],
                                columns=['Open', 'High', 'Low', 'Close', 'Volume'],
                                index=pd.DatetimeIndex(data=[], tz='GMT'))

        timeframe = AlpacaOHLCVDataRepository.convert_str_interval_to_timeframe(scenario.interval)
        start_date_str = f"{scenario.start_date.astimezone(timezone('GMT')).isoformat()}"
        end_date_str = f"{scenario.end_date.astimezone(timezone('GMT')).isoformat()}"
        data = self._alpaca_api.get_bars(symbol=scenario.symbol,
                                         timeframe=timeframe,
                                         start=start_date_str,
                                         end=end_date_str)
        df = data.df
        df = df.rename(columns={"open": "Open",
                                "high": "High",
                                "low": "Low",
                                "close": "Close",
                                "volume": "Volume"})

        # FIXME: If no data, it returns empty dataframe
        if len(df.columns) == 0:
            df = pd.DataFrame(data=[],
                              columns=['Open', 'High', 'Low', 'Close', 'Volume'],
                              index=pd.DatetimeIndex(data=[], tz='GMT'))
        else:
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

        return df

    @staticmethod
    def convert_str_interval_to_timeframe(str_interval: str):
        return {
            '1m': TimeFrame(1, TimeFrameUnit.Minute),
            '5m': TimeFrame(5, TimeFrameUnit.Minute),
            '15m': TimeFrame(15, TimeFrameUnit.Minute),
            '30m': TimeFrame(30, TimeFrameUnit.Minute),
            '1h': TimeFrame(1, TimeFrameUnit.Hour),
            '1d': TimeFrame(1, TimeFrameUnit.Day),
        }[str_interval]
