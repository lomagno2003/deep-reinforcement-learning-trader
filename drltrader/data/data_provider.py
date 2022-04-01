import yfinance as yf
from finta import TA

from drltrader.data.scenario import Scenario


class DataProvider:
    def __init__(self):
        self.indicator_column_names = None

        self._define_indicators()

    def retrieve_data(self,
                      scenario: Scenario):
        df = self._fetch_data(scenario)

        self._calculate_indicators(df)

        return df

    def _define_indicators(self):
        self.indicators_function = {}
        self.indicators_parameters = {}
        self.indicators_function['SMA'] = TA.SMA
        self.indicators_parameters['SMA'] = {
            "period": [6, 12]
        }
        self.indicators_function['VWAP'] = TA.VWAP
        self.indicators_parameters['VWAP'] = {}
        self.indicators_function['MACD'] = DataProvider._extract_column(TA.MACD, 'SIGNAL')
        self.indicators_parameters['MACD'] = {
            "period_fast": [12],
            "period_slow": [26],
            "signal": [9]
        }
        self.indicators_function['PPO'] = DataProvider._extract_column(TA.PPO, 'HISTO')
        self.indicators_parameters['PPO'] = {
            "period_fast": [12],
            "period_slow": [26],
            "signal": [9]
        }
        self.indicators_function['VW_MACD'] = DataProvider._extract_column(TA.VW_MACD, 'SIGNAL')
        self.indicators_parameters['VW_MACD'] = {
            "period_fast": [12],
            "period_slow": [26],
            "signal": [9]
        }
        self.indicators_function['RSI'] = TA.RSI
        self.indicators_parameters['RSI'] = {
            "period": [14]
        }
        self.indicators_function['OBV'] = TA.OBV
        self.indicators_parameters['OBV'] = {}
        self.indicators_function['MOM'] = TA.MOM
        self.indicators_parameters['MOM'] = {
            "period": [10]
        }
        self.indicators_function['ROC'] = TA.ROC
        self.indicators_parameters['ROC'] = {
            "period": [10]
        }
        self.indicators_function['VBM'] = TA.VBM
        self.indicators_parameters['VBM'] = {
            "roc_period": [12],
            "atr_period": [26]
        }

    def _calculate_indicators(self, df):
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
