import yfinance as yf
import pandas as pd
from datetime import datetime
import logging
import logging.config
from finta import TA

from drltrader.data import DataRepository, Scenario

logging.config.fileConfig('log.ini', disable_existing_loggers=False)
logger = logging.getLogger(__name__)


class OHLCVDataRepository(DataRepository):
    def __init__(self,
                 cache_enabled: bool = True):
        self.indicator_column_names = None
        self._cache = {}
        self._cache_enabled = cache_enabled

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

            # FIXME: There's some weird bug on Yahoo
            df = df.iloc[:-1, :]

            return df

    def _fetch_data(self, scenario):
        ticker = yf.Ticker(scenario.symbol)
        df = ticker.history(start=scenario.start_date,
                            end=scenario.end_date,
                            interval=scenario.interval)
        return df
