import logging.config

from drltrader.data.sentiment_data_repository import TickerFeedRepository

logging.config.fileConfig('log.ini', disable_existing_loggers=False)
logger = logging.getLogger(__name__)


class StaticFeedRepository(TickerFeedRepository):
    def __init__(self, articles_per_symbol):
        self._articles_per_symbol = articles_per_symbol

    def find_articles(self, ticker: str):
        return self._articles_per_symbol[ticker]
