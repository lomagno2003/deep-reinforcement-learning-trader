import os
from pathlib import Path
import pickle
import logging.config
from datetime import datetime

from drltrader.media import TickerMediaRepository

logging.config.fileConfig('log.ini', disable_existing_loggers=False)
logger = logging.getLogger(__name__)


class CachedMediaRepository(TickerMediaRepository):
    def __init__(self,
                 source_media_repository: TickerMediaRepository,
                 cache_directory: str = 'temp/cache_media'):
        self._source_media_repository = source_media_repository
        self._cache_directory = cache_directory

    def find_medias(self, ticker: str, from_date: datetime, to_date: datetime):
        self._initialize_directory()

        if self._query_already_cached(ticker, from_date, to_date):
            return self._load_query_cache(ticker, from_date, to_date)
        else:
            articles = self._source_media_repository.find_medias(ticker, from_date, to_date)
            self._save_query_cache(ticker, from_date, to_date, articles)
            return articles

    def _parse_cache_path(self, ticker: str, from_date: datetime, to_date: datetime):
        return f'{self._cache_directory}/{ticker}_{from_date.timestamp()}_{to_date.timestamp()}'

    def _initialize_directory(self):
        directory_exists = (Path.cwd() / self._cache_directory).exists()

        if not directory_exists:
            os.mkdir(self._cache_directory)

    def _query_already_cached(self, ticker: str, from_date: datetime, to_date: datetime):
        query_path = self._parse_cache_path(ticker, from_date, to_date)
        query_cache_exists = (Path.cwd() / query_path).exists()

        return query_cache_exists

    def _load_query_cache(self, ticker: str, from_date: datetime, to_date: datetime):
        query_path = self._parse_cache_path(ticker, from_date, to_date)
        with open(query_path, 'rb') as query_file:
            return pickle.load(query_file)

    def _save_query_cache(self, ticker: str, from_date: datetime, to_date: datetime, articles: list):
        query_path = self._parse_cache_path(ticker, from_date, to_date)

        with open(query_path, 'wb') as query_file:
            pickle.dump(articles, query_file)
