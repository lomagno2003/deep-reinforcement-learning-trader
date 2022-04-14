import unittest
from datetime import datetime
from datetime import timedelta

from drltrader.data.sentiment_data_repository import SentimentDataRepository
from drltrader.data.news_feed_repository import NewsMediaRepository
from drltrader.data.twitter_feed_repository import TwitterMediaRepository
from drltrader.data.composite_data_repository import CompositeDataRepository
from drltrader.data.indicators_data_repository import IndicatorsDataRepository
from drltrader.data.ohlcv_data_repository import OHLCVDataRepository
from drltrader.data import DataRepository, Scenario


class DataRepositoryIntegrationTestCase(unittest.TestCase):
    def test_retrieve_data(self):
        # Arrange
        testing_scenario = Scenario(symbols=['TSLA'],
                                    interval='1h',
                                    start_date=datetime.now() - timedelta(days=1),
                                    end_date=datetime.now())

        data_repository: DataRepository = IndicatorsDataRepository(
            CompositeDataRepository([
                OHLCVDataRepository(),
                SentimentDataRepository(NewsMediaRepository()),
                SentimentDataRepository(TwitterMediaRepository())
            ])
        )

        # Act
        dataframe = data_repository.retrieve_datas(testing_scenario)

        # Assert
        self.assertIsNotNone(dataframe)


if __name__ == '__main__':
    unittest.main()