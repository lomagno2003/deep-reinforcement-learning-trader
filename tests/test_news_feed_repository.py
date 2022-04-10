import unittest

from drltrader.data.news_feed_repository import NewsFeedRepository


class NewsFeedRepositoryTestCase(unittest.TestCase):
    def test_find_articles(self):
        # Arrange
        data_repository: NewsFeedRepository = NewsFeedRepository()

        # Act
        articles = data_repository.find_articles('TSLA')

        # Assert
        self.assertIsNotNone(articles)


if __name__ == '__main__':
    unittest.main()