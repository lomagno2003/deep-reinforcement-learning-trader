import pandas as pd
import logging.config
from transformers import pipeline

from drltrader.data import DataRepository, Scenario
from drltrader.media import TickerMediaRepository

logging.config.fileConfig('log.ini', disable_existing_loggers=False)
logger = logging.getLogger(__name__)


class SentimentDataRepository(DataRepository):
    def __init__(self, ticker_media_repository: TickerMediaRepository):
        self._ticker_media_repository = ticker_media_repository
        self._sentiment_analysis_pipeline = pipeline('sentiment-analysis')

    def get_repository_name(self):
        return f"Sentiment({self._ticker_media_repository.get_column_prefix()})_"

    def retrieve_datas(self, scenario: Scenario):
        logger.info(f"Retrieving sentiment for scenario {scenario}")

        result = {}
        for symbol in scenario.symbols:
            medias = self._ticker_media_repository.find_medias(ticker=symbol,
                                                               from_date=scenario.start_date,
                                                               to_date=scenario.end_date)

            # TODO: Probably there's a better way to do this
            # FIXME: Here we are assuming no collisions
            medias_dates = list(map(lambda a: a.datetime, medias))
            medias_sentiment = list(map(lambda a: [1.0, 0.0] if a.sentiment == 'POSITIVE' else [0.0, 1.0], medias))

            columns = [f"{self._ticker_media_repository.get_column_prefix()}_positive_sentiment",
                       f"{self._ticker_media_repository.get_column_prefix()}_negative_sentiment"]
            symbol_dataframe = pd.DataFrame(columns=columns,
                                            index=medias_dates,
                                            data=medias_sentiment)

            result[symbol] = symbol_dataframe

        return result
