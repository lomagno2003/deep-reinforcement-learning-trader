import logging.config
from datetime import datetime

from drltrader.media import TickerMediaRepository

logging.config.fileConfig('log.ini', disable_existing_loggers=False)
logger = logging.getLogger(__name__)


class StaticMediaRepository(TickerMediaRepository):
    def __init__(self, medias_per_symbol):
        self._medias_per_symbol = medias_per_symbol

    def find_medias(self, ticker: str, from_date: datetime, to_date: datetime):
        return self._medias_per_symbol[ticker]
