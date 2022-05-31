import pandas as pd
import logging
import logging.config
from finta import TA

from drltrader.data import DataRepository
from drltrader.data import Scenario


logging.config.fileConfig('log.ini', disable_existing_loggers=False)
logger = logging.getLogger(__name__)


class ResampleDataRepository(DataRepository):
    def __init__(self, source_data_repository: DataRepository, resampled_interval: str):
        self._source_data_repository = source_data_repository
        self._resampled_interval = resampled_interval

    def get_repository_name(self):
        return f"Resample({self._resampled_interval}, {self._source_data_repository.get_repository_name()})"

    def retrieve_datas(self, scenario: Scenario):
        dataframe_per_symbol = self._source_data_repository.retrieve_datas(scenario)

        for symbol in dataframe_per_symbol:
            dataframe_per_symbol[symbol] = self._resample_dataframe(dataframe_per_symbol[symbol])

        return dataframe_per_symbol

    def _resample_dataframe(self, dataframe: pd.DataFrame):
        resample_rule = ResampleDataRepository._convert_str_interval_to_rule(self._resampled_interval)

        open = dataframe['Open'].resample(resample_rule).first()
        high = dataframe['High'].resample(resample_rule).max()
        low = dataframe['Low'].resample(resample_rule).min()
        close = dataframe['Close'].resample(resample_rule).last()
        volume = dataframe['Volume'].resample(resample_rule).sum()

        return pd.DataFrame({
            'Open': open,
            'High': high,
            'Low': low,
            'Close': close,
            'Volume': volume
        }).dropna()

    @staticmethod
    def _convert_str_interval_to_rule(str_interval: str):
        return {
            '1m': '1min',
            '5m': '5min',
            '15m': '15min',
            '30m': '30min',
            '60m': '60min',
            '1h': '60min',
            '1d': '1d',
        }[str_interval]
