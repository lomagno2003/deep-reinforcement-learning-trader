import unittest
from datetime import datetime
from datetime import timedelta
from datetime import time

from drltrader.media.news_media_repository import NewsMediaRepository
from drltrader.media.sentiment_media_repository import SentimentMediaRepository
from drltrader.media.cached_media_repository import CachedMediaRepository
from drltrader.media.twitter_media_repository import TwitterMediaRepository
from drltrader.data.sentiment_data_repository import SentimentDataRepository
from drltrader.data.composite_data_repository import CompositeDataRepository
from drltrader.data.indicators_data_repository import IndicatorsDataRepository
from drltrader.data.ohlcv_data_repository import AlpacaOHLCVDataRepository
from drltrader.data.data_repositories import DataRepositories
from drltrader.data import DataRepository, Scenario
from drltrader.data.scenarios import Scenarios


class DataRepositoryIntegrationTestCase(unittest.TestCase):
    def test_sentiment_integration_retrieve_data(self):
        # Arrange
        start_date = datetime.combine(datetime.now() - timedelta(days=1), time.min)
        end_date = datetime.combine(datetime.now(), time.max)

        testing_scenario = Scenario(symbols=['TSLA'],
                                    interval='1h',
                                    start_date=start_date,
                                    end_date=end_date)

        news_media_repository = CachedMediaRepository(SentimentMediaRepository(NewsMediaRepository()))
        twitter_media_repository = CachedMediaRepository(SentimentMediaRepository(TwitterMediaRepository()))

        data_repository: DataRepository = IndicatorsDataRepository(
            CompositeDataRepository([
                AlpacaOHLCVDataRepository(),
                SentimentDataRepository(news_media_repository),
                SentimentDataRepository(twitter_media_repository)
            ])
        )

        # Act
        dataframe = data_repository.retrieve_datas(testing_scenario)

        # Assert
        self.assertIsNotNone(dataframe)

    def test_multi_timeframe_integration_retrieve_data(self):
        # Arrange
        testing_scenario = Scenarios.last_market_week(symbols=['TSLA', 'MELI'],
                                                      interval='5m')

        data_repository = DataRepositories.build_multi_time_interval_data_repository()

        # Act
        result = data_repository.retrieve_datas(testing_scenario)

        # Assert
        self.assertIsNotNone(result)
        self.assertIsNotNone(data_repository.get_repository_name())
        self.assertIsNotNone(data_repository.get_columns_per_symbol())


if __name__ == '__main__':
    unittest.main()
