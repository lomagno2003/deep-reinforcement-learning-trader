import unittest
from datetime import datetime
from datetime import timedelta

from drltrader.media.news_media_repository import NewsMediaRepository


class NewsMediaRepositoryTestCase(unittest.TestCase):
    def test_find_articles(self):
        # Arrange
        data_repository: NewsMediaRepository = NewsMediaRepository()

        # Act
        articles = data_repository.find_medias('TSLA',
                                               from_date=datetime.now() - timedelta(days=2),
                                               to_date=datetime.now())

        # Assert
        self.assertIsNotNone(articles)


if __name__ == '__main__':
    unittest.main()
