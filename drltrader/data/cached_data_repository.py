import os
from pathlib import Path
import pickle
import zlib
import logging.config
from datetime import datetime

from drltrader.data import DataRepository, Scenario

logging.config.fileConfig('log.ini', disable_existing_loggers=False)
logger = logging.getLogger(__name__)


class CachedDataRepository(DataRepository):
    def __init__(self,
                 source_data_repository: DataRepository,
                 cache_directory: str = 'temp/cache_data'):
        self._source_data_repository = source_data_repository
        self._cache_directory = cache_directory

    def get_repository_name(self):
        return f"Ca({self._source_data_repository.get_repository_name()})"

    def retrieve_datas(self, scenario: Scenario):
        self._initialize_directory()

        if self._query_already_cached(scenario):
            return self._load_query_cache(scenario)
        else:
            dataframe_per_symbol = self._source_data_repository.retrieve_datas(scenario)
            self._save_query_cache(scenario, dataframe_per_symbol)
            return dataframe_per_symbol

    def _parse_cache_path(self, scenario: Scenario):
        path = f'{self._cache_directory}/' \
               f'{self.get_repository_name()}_' \
               f'{scenario}_'

        return path

    def _initialize_directory(self):
        directory_exists = (Path.cwd() / self._cache_directory).exists()

        if not directory_exists:
            os.mkdir(self._cache_directory)

    def _query_already_cached(self, scenario: Scenario):
        query_path = self._parse_cache_path(scenario)
        query_cache_exists = (Path.cwd() / query_path).exists()

        return query_cache_exists

    def _load_query_cache(self, scenario: Scenario):
        query_path = self._parse_cache_path(scenario)
        with open(query_path, 'rb') as query_file:
            return pickle.load(query_file)

    def _save_query_cache(self, scenario: Scenario, dataframe_per_symbol: dict):
        query_path = self._parse_cache_path(scenario)

        with open(query_path, 'wb') as query_file:
            pickle.dump(dataframe_per_symbol, query_file)
