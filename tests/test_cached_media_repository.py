import unittest
from datetime import datetime
from datetime import timedelta

from drltrader.media import Media
from drltrader.media.static_media_repository import StaticMediaRepository
from drltrader.media.cached_media_repository import CachedMediaRepository


class SentimentDataRepositoryTestCase(unittest.TestCase):
    def __init__(self, method_name: str):
        super(SentimentDataRepositoryTestCase, self).__init__(method_name)

        self._source_media_repository = StaticMediaRepository(medias_per_symbol={
            'TSLA': [Media(datetime=datetime.now() - timedelta(days=3), summary='Tesla is bad'),
                     Media(datetime=datetime.now() - timedelta(days=5), summary='Tesla is good'),
                     Media(datetime=datetime.now() - timedelta(days=7), summary='Tesla is bad')],
            'MELI': [Media(datetime=datetime.now() - timedelta(days=3), summary='MELI is good'),
                     Media(datetime=datetime.now() - timedelta(days=5), summary='MELI is bad'),
                     Media(datetime=datetime.now() - timedelta(days=7), summary='MELI is good')]
        })

    def test_retrieve_cached_data(self):
        # Arrange
        self._sentiment_media_repository = CachedMediaRepository(
            source_media_repository=self._source_media_repository)

        # Act
        articles = self._sentiment_media_repository.find_medias(ticker='TSLA',
                                                                from_date=datetime.fromtimestamp(1500000000),
                                                                to_date=datetime.fromtimestamp(1600000000))

        # Assert
        self.assertIsNotNone(articles)

    def test_retrieve_uncached_data(self):
        # Arrange
        self._sentiment_media_repository = CachedMediaRepository(
            source_media_repository=self._source_media_repository)

        # Act
        articles = self._sentiment_media_repository.find_medias(ticker='TSLA',
                                                                from_date=datetime.fromtimestamp(1500000000),
                                                                to_date=datetime.now())

        # Assert
        self.assertIsNotNone(articles)

if __name__ == '__main__':
    unittest.main()
