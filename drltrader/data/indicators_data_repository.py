import pandas as pd
import logging
import logging.config
from finta import TA

from drltrader.data import DataRepository
from drltrader.data import Scenario


logging.config.fileConfig('log.ini', disable_existing_loggers=False)
logger = logging.getLogger(__name__)


class IndicatorsDataRepository(DataRepository):
    def __init__(self, source_data_repository: DataRepository):
        self._source_data_repository = source_data_repository
        self._define_indicators()

    def get_repository_name(self):
        return f"I({self._source_data_repository.get_repository_name()})"

    def retrieve_datas(self, scenario: Scenario):
        dataframe_per_symbol = self._source_data_repository.retrieve_datas(scenario)

        for symbol in dataframe_per_symbol:
            self._calculate_indicators(dataframe_per_symbol[symbol])

        return dataframe_per_symbol

    def _define_indicators(self):
        self.indicator_column_names = None

        self.indicators_function = {}
        self.indicators_parameters = {}

        self.indicators_function['SMA'] = TA.SMA
        self.indicators_parameters['SMA'] = {
            "period": [4, 6, 8, 10, 12, 14, 16, 18]
        }
        self.indicators_function['VWAP'] = TA.VWAP
        self.indicators_parameters['VWAP'] = {}
        self.indicators_function['MACD'] = IndicatorsDataRepository._extract_column(TA.MACD, 'SIGNAL')
        self.indicators_parameters['MACD'] = {
            "period_fast": [4, 8, 12, 16],
            "period_slow": [8, 16, 24, 32],
            "signal": [3, 6, 9, 12]
        }
        self.indicators_function['PPO'] = IndicatorsDataRepository._extract_column(TA.PPO, 'HISTO')
        self.indicators_parameters['PPO'] = {
            "period_fast": [4, 8, 12, 16],
            "period_slow": [8, 16, 24, 32],
            "signal": [3, 6, 9, 12]
        }
        self.indicators_function['VW_MACD'] = IndicatorsDataRepository._extract_column(TA.VW_MACD, 'SIGNAL')
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

    @staticmethod
    def _extract_column(function, column_name):
        def _extract(df, **kwargs):
            return function(df, **kwargs)[column_name]

        return _extract
