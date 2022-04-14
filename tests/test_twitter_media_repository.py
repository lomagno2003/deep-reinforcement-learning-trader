import unittest
from datetime import datetime
from datetime import timedelta

from drltrader.media.twitter_media_repository import TwitterMediaRepository


class TwitterMediaRepositoryTestCase(unittest.TestCase):
    def test_find_articles(self):
        # Arrange
        data_repository: TwitterMediaRepository = TwitterMediaRepository()

        # Act
        articles = data_repository.find_medias('TSLA',
                                               from_date=datetime.now() - timedelta(days=1),
                                               to_date=datetime.now())

        # Assert
        self.assertIsNotNone(articles)


if __name__ == '__main__':
    unittest.main()
