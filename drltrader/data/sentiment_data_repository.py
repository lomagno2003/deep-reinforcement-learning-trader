import pandas as pd
from datetime import datetime
import logging
import logging.config
from transformers import pipeline

from drltrader.data import DataRepository, Scenario

logging.config.fileConfig('log.ini', disable_existing_loggers=False)
logger = logging.getLogger(__name__)


class Article:
    def __init__(self,
                 datetime: datetime = None,
                 url: str = None,
                 content: str = None,
                 summary: str = None,
                 sentiment: str = None):
        self.datetime = datetime
        self.url = url
        self.content = content
        self.summary = summary
        self.sentiment = sentiment


class TickerFeedRepository:
    def get_column_prefix(self):
        pass

    def find_articles(self, ticker: str, from_date: datetime, to_date: datetime) -> list:
        pass


class SentimentDataRepository(DataRepository):
    def __init__(self, ticker_feed_repository: TickerFeedRepository):
        self._ticker_feed_repository = ticker_feed_repository
        self._sentiment_analysis_pipeline = pipeline('sentiment-analysis')

    def retrieve_datas(self, scenario: Scenario):
        logger.info(f"Retrieving sentiment for scenario {scenario}")

        result = {}
        for symbol in scenario.symbols:
            articles = self._ticker_feed_repository.find_articles(ticker=symbol,
                                                                  from_date=scenario.start_date,
                                                                  to_date=scenario.end_date)
            articles = self._add_sentiment(articles)

            # TODO: Probably there's a better way to do this
            # FIXME: Here we are assuming no collisions
            articles_dates = list(map(lambda a: a.datetime, articles))
            articles_sentiment = list(map(lambda a: [1.0, 0.0] if a.sentiment == 'POSITIVE' else [0.0, 1.0], articles))

            symbol_dataframe = pd.DataFrame(columns=['positive_sentiment', 'negative_sentiment'],
                                            index=articles_dates,
                                            data=articles_sentiment)

            result[symbol] = symbol_dataframe

        return result

    def _add_sentiment(self, articles: list):
        logger.debug(f"Adding sentiment to {len(articles)} articles")
        summaries = list(map(lambda a: a.summary, articles))
        scores = self._sentiment_analysis_pipeline(summaries)

        for i in range(len(articles)):
            articles[i].sentiment = scores[i]['label']

        return articles
