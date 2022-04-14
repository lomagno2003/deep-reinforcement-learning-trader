import logging.config
from datetime import datetime
from transformers import pipeline

from drltrader.media import TickerMediaRepository

logging.config.fileConfig('log.ini', disable_existing_loggers=False)
logger = logging.getLogger(__name__)


class SentimentMediaRepository(TickerMediaRepository):
    def __init__(self, source_media_repository: TickerMediaRepository):
        self._source_media_repository = source_media_repository
        self._sentiment_analysis_pipeline = pipeline('sentiment-analysis')

    def get_column_prefix(self):
        return self._source_media_repository.get_column_prefix()

    def find_medias(self, ticker: str, from_date: datetime, to_date: datetime):
        sourced_medias = self._source_media_repository.find_medias(ticker=ticker,
                                                                   from_date=from_date,
                                                                   to_date=to_date)

        updated_medias = self._add_sentiment(medias=sourced_medias)

        return updated_medias

    def _add_sentiment(self, medias: list):
        logger.debug(f"Adding sentiment to {len(medias)} medias")
        summaries = list(map(lambda a: a.summary, medias))
        scores = self._sentiment_analysis_pipeline(summaries)

        for i in range(len(medias)):
            medias[i].sentiment = scores[i]['label']

        return medias
