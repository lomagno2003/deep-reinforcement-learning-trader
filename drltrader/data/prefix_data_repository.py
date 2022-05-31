import pandas as pd
import logging
import logging.config
from finta import TA

from drltrader.data import DataRepository
from drltrader.data import Scenario


logging.config.fileConfig('log.ini', disable_existing_loggers=False)
logger = logging.getLogger(__name__)


class PrefixDataRepository(DataRepository):
    def __init__(self, source_data_repository: DataRepository, column_prefix: str):
        self._source_data_repository = source_data_repository
        self._column_prefix = column_prefix

    def get_repository_name(self):
        return f"Prefix({self._column_prefix}, {self._source_data_repository.get_repository_name()})"

    def retrieve_datas(self, scenario: Scenario):
        dataframe_per_symbol = self._source_data_repository.retrieve_datas(scenario)

        for symbol in dataframe_per_symbol:
            dataframe_per_symbol[symbol] = self._prefix_dataframe(dataframe_per_symbol[symbol])

        return dataframe_per_symbol

    def _prefix_dataframe(self, dataframe: pd.DataFrame):
        rename_map = {}

        for column in dataframe:
            rename_map[column] = f"{self._column_prefix}_{column}"

        return dataframe.rename(columns=rename_map)
