import unittest
from datetime import datetime
from datetime import timedelta

from drltrader.data.stocktwits_feed_repository import StockTwitsFeedRepository


class StockTwitsFeedRepositoryTestCase(unittest.TestCase):
    def test_find_articles(self):
        # Arrange
        data_repository: StockTwitsFeedRepository = StockTwitsFeedRepository()

        # Act
        articles = data_repository.find_articles('TSLA',
                                                 from_date=datetime.now() - timedelta(days=2),
                                                 to_date=datetime.now())

        # Assert
        self.assertIsNotNone(articles)


if __name__ == '__main__':
    unittest.main()