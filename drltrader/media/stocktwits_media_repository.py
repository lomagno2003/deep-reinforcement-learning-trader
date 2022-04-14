import logging.config
import pytwits
from datetime import datetime

from drltrader.media import Media, TickerMediaRepository

logging.config.fileConfig('log.ini', disable_existing_loggers=False)
logger = logging.getLogger(__name__)


class StockTwitsMediaRepository(TickerMediaRepository):
    def __init__(self):
        # FIXME: Currently, stocktwits is not registering new applications. See https://api.stocktwits.com/developers
        self._access_token = None
        self.stocktwits_api = pytwits.StockTwits(access_token=self._access_token)

    def find_medias(self, ticker: str, from_date: datetime, to_date: datetime):
        # TODO: This is kind of stub method. This code is not completed and it does not work
        symbols = self.stocktwits_api.streams(path='symbols', id='AAPL')

        return None
