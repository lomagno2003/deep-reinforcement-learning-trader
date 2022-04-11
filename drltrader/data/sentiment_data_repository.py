import pandas as pd
from datetime import datetime

from transformers import pipeline

from drltrader.data import DataRepository, Scenario


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
    def find_articles(self, ticker: str) -> list:
        pass


class SentimentDataRepository(DataRepository):
    def __init__(self, ticker_feed_repository: TickerFeedRepository):
        self._ticker_feed_repository = ticker_feed_repository
        self._sentiment_analysis_pipeline = pipeline('sentiment-analysis')

    def retrieve_datas(self, scenario: Scenario):
        result = {}
        for symbol in scenario.symbols:
            articles = self._ticker_feed_repository.find_articles(symbol)
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
        summaries = list(map(lambda a: a.summary, articles))
        scores = self._sentiment_analysis_pipeline(summaries)

        for i in range(len(articles)):
            articles[i].sentiment = scores[i]['label']

        return articles
