import unittest
from datetime import datetime
from datetime import timedelta

from drltrader.data.news_feed_repository import NewsFeedRepository


class NewsFeedRepositoryTestCase(unittest.TestCase):
    def test_find_articles(self):
        # Arrange
        data_repository: NewsFeedRepository = NewsFeedRepository()

        # Act
        articles = data_repository.find_articles('TSLA',
                                                 from_date=datetime.now() - timedelta(days=2),
                                                 to_date=datetime.now())

        # Assert
        self.assertIsNotNone(articles)


if __name__ == '__main__':
    unittest.main()