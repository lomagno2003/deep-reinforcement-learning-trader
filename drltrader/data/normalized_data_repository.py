import math

import pandas as pd
import logging.config

from drltrader.data import DataRepository
from drltrader.data import Scenario


logging.config.fileConfig('log.ini', disable_existing_loggers=False)
logger = logging.getLogger(__name__)


class NormalizedDataRepository(DataRepository):
    def __init__(self, source_data_repository: DataRepository, excluded_columns: list = []):
        self._source_data_repository = source_data_repository
        self._excluded_columns = excluded_columns

    def get_repository_name(self):
        return f"N({self._column_prefix}, {self._source_data_repository.get_repository_name()})"

    def retrieve_datas(self, scenario: Scenario):
        dataframe_per_symbol = self._source_data_repository.retrieve_datas(scenario)

        for symbol in dataframe_per_symbol:
            dataframe_per_symbol[symbol] = self._normalize_dataframe(dataframe_per_symbol[symbol])

        return dataframe_per_symbol

    def _normalize_dataframe(self, dataframe: pd.DataFrame):
        # FIXME: This won't work for live data as max/min changes overtime
        for column in dataframe.columns:
            if column in self._excluded_columns:
                continue

            max_value = dataframe[column].max()
            min_value = dataframe[column].min()
            range_value = math.fabs(max_value - min_value)

            if range_value > 0.0:
                dataframe[column] = dataframe[column].apply(lambda x: ((x - (range_value / 2.0 + min_value)) / range_value) * 2.0)
            else:
                dataframe[column] = dataframe[column].apply(lambda x: 0.0)

        return dataframe
