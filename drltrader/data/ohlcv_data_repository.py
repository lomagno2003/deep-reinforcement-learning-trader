import yfinance as yf
import pandas as pd
from datetime import datetime
import logging
import logging.config
from finta import TA

from drltrader.data.scenario import Scenario
from drltrader.data import DataRepository

logging.config.fileConfig('log.ini', disable_existing_loggers=False)
logger = logging.getLogger(__name__)


class OHLCVDataRepository(DataRepository):
    def __init__(self,
                 cache_enabled: bool = True):
        self.indicator_column_names = None
        self._cache = {}
        self._cache_enabled = cache_enabled
        self._define_indicators()

    def retrieve_datas(self, scenario: Scenario):
        dataframe_per_symbol = {}

        intersection_index = None

        for symbol in scenario.symbols:
            dataframe_per_symbol[symbol] = self.retrieve_data(scenario.clone_with_symbol(symbol))

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
            self._calculate_indicators(df)

            if self._cache_enabled:
                self._cache[str(scenario)] = df

            # FIXME: There's some weird bug on Yahoo
            df = df.iloc[:-1, :]

            return df

    def _define_indicators(self):
        self.indicators_function = {}
        self.indicators_parameters = {}

        self.indicators_function['SMA'] = TA.SMA
        self.indicators_parameters['SMA'] = {
            "period": [4, 6, 8, 10, 12, 14, 16, 18]
        }
        self.indicators_function['VWAP'] = TA.VWAP
        self.indicators_parameters['VWAP'] = {}
        self.indicators_function['MACD'] = OHLCVDataRepository._extract_column(TA.MACD, 'SIGNAL')
        self.indicators_parameters['MACD'] = {
            "period_fast": [4, 8, 12, 16],
            "period_slow": [8, 16, 24, 32],
            "signal": [3, 6, 9, 12]
        }
        self.indicators_function['PPO'] = OHLCVDataRepository._extract_column(TA.PPO, 'HISTO')
        self.indicators_parameters['PPO'] = {
            "period_fast": [4, 8, 12, 16],
            "period_slow": [8, 16, 24, 32],
            "signal": [3, 6, 9, 12]
        }
        self.indicators_function['VW_MACD'] = OHLCVDataRepository._extract_column(TA.VW_MACD, 'SIGNAL')
        self.indicators_parameters['VW_MACD'] = {
            "period_fast": [4, 8, 12, 16],
            "period_slow": [8, 16, 24, 32],
            "signal": [3, 6, 9, 12]
        }
        self.indicators_function['RSI'] = TA.RSI
        self.indicators_parameters['RSI'] = {
            "period": [4, 6, 8, 10, 12, 14, 16, 18]
        }
        self.indicators_function['OBV'] = TA.OBV
        self.indicators_parameters['OBV'] = {}

        self.indicators_function['MOM'] = TA.MOM
        self.indicators_parameters['MOM'] = {
            "period": [4, 6, 8, 10, 12, 14, 16, 18]
        }
        self.indicators_function['ROC'] = TA.ROC
        self.indicators_parameters['ROC'] = {
            "period": [4, 6, 8, 10, 12, 14, 16, 18]
        }
        self.indicators_function['VBM'] = TA.VBM
        self.indicators_parameters['VBM'] = {
            "roc_period": [6, 8, 10, 12, 14],
            "atr_period": [12, 16, 20, 24, 28]
        }

        # Populate indicator_column_names
        df = pd.DataFrame(columns=['Open', 'High', 'Close', 'Low', 'Volume'])
        self._calculate_indicators(df)

    def _calculate_indicators(self, df):
        logger.debug(f"Calculating indicators...")

        all_indicator_columns_names = []

        for indicator_name in self.indicators_function:
            function = self.indicators_function[indicator_name]
            parameters = self.indicators_parameters[indicator_name]

            indicator_column_names = []
            parsed_parameters = []

            if len(parameters) == 0:
                parsed_parameters.append({})
                indicator_column_names.append(indicator_name)
            else:
                for parameter_name in parameters:
                    for parameter_value_idx in range(0, len(parameters[parameter_name])):
                        if parameter_value_idx >= len(parsed_parameters):
                            parsed_parameters.append({})
                            indicator_column_names.append(indicator_name)

                        parsed_parameters[parameter_value_idx][parameter_name] = parameters[parameter_name][parameter_value_idx]
                        indicator_column_names[parameter_value_idx] = f"{indicator_column_names[parameter_value_idx]}_{parsed_parameters[parameter_value_idx][parameter_name]}"

            for parameter_value_idx in range(0, len(parsed_parameters)):
                df[indicator_column_names[parameter_value_idx]] = function(df, **parsed_parameters[parameter_value_idx])
                all_indicator_columns_names.append(indicator_column_names[parameter_value_idx])

        df.fillna(0.0, inplace=True)

        if self.indicator_column_names is None:
            self.indicator_column_names = all_indicator_columns_names

        logger.debug(f"The following indicators were calculated: {self.indicator_column_names}")

    def _fetch_data(self, scenario):
        ticker = yf.Ticker(scenario.symbol)
        df = ticker.history(start=scenario.start_date,
                            end=scenario.end_date,
                            interval=scenario.interval)
        return df

    @staticmethod
    def _extract_column(function, column_name):
        def _extract(df, **kwargs):
            return function(df, **kwargs)[column_name]

        return _extract
