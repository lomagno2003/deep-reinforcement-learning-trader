import unittest
from datetime import datetime
from datetime import timedelta
import pandas as pd

from drltrader.data import Scenario
from drltrader.data.sentiment_data_repository import SentimentDataRepository
from drltrader.data.sentiment_data_repository import Article
from drltrader.data.static_feed_repository import StaticFeedRepository


class SentimentDataRepositoryTestCase(unittest.TestCase):
    def test_retrieve_datas(self):
        # Arrange
        testing_scenario = Scenario(symbols=['TSLA', 'MELI'],
                                    start_date=datetime.now() - timedelta(days=30),
                                    end_date=datetime.now())
        data_repository: SentimentDataRepository = SentimentDataRepository(
            ticker_feed_repository=self._initialize_feed_repository())

        # Act
        dataframe_per_symbol = data_repository.retrieve_datas(testing_scenario)

        # Assert
        self.assertIsNotNone(dataframe_per_symbol)

        pd.set_option('display.max_columns', None)

    def _initialize_feed_repository(self):
        return StaticFeedRepository(articles_per_symbol={
            'TSLA': [Article(datetime=datetime.now() - timedelta(days=3), summary='Tesla is bad'),
                     Article(datetime=datetime.now() - timedelta(days=5), summary='Tesla is good'),
                     Article(datetime=datetime.now() - timedelta(days=7), summary='Tesla is bad')],
            'MELI': [Article(datetime=datetime.now() - timedelta(days=3), summary='MELI is good'),
                     Article(datetime=datetime.now() - timedelta(days=5), summary='MELI is bad'),
                     Article(datetime=datetime.now() - timedelta(days=7), summary='MELI is good')]
        })

if __name__ == '__main__':
    unittest.main()