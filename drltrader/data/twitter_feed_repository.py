import logging.config
import tweepy
import json
from datetime import datetime

from drltrader.data.sentiment_data_repository import TickerFeedRepository
from drltrader.data.sentiment_data_repository import Article

logging.config.fileConfig('log.ini', disable_existing_loggers=False)
logger = logging.getLogger(__name__)


class TwitterFeedRepository(TickerFeedRepository):
    def __init__(self, config_file_name: str = 'config.json'):
        with open(config_file_name) as config_file:
            config = json.load(config_file)

        self._access_key = config['twitter']['api_key']
        self._access_secret = config['twitter']['api_secret']
        self._bearer_token = config['twitter']['bearer_token']

        auth = tweepy.OAuth2BearerHandler(self._bearer_token)

        self._tweepy_api: tweepy.API = tweepy.API(auth)

    def get_column_prefix(self):
        return 'twitter'

    def find_articles(self, ticker: str, from_date: datetime, to_date: datetime):
        logger.info(f"Searching tweets for {ticker} from {from_date} to {datetime}")
        # TODO: Add logging
        # FIXME: Current search has up to 7 days history. To improve we need premium API:
        # https://docs.tweepy.org/en/stable/api.html#tweepy.API.search_full_archive
        query = ticker
        since = from_date.strftime("%Y-%m-%d")
        until = to_date.strftime("%Y-%m-%d")

        # TODO: There's a limitation of 450 request per 15 min period. Each request can retrieve up to 100 tweets
        # If we hit the limit, the API starts failing for 15 min.
        max_tweets = 40000
        articles = []
        last_id = -1

        while len(articles) < max_tweets:
            count = max_tweets - len(articles)
            max_id = str(last_id - 1)

            logger.debug(f"Querying twitter search API with max-id: {max_id}")
            new_tweets = self._tweepy_api.search_tweets(q=query,
                                                        since=since,
                                                        until=until,
                                                        count=count,
                                                        max_id=max_id)
            if not new_tweets:
                break

            for tweet in new_tweets:
                tweet_creation_date = tweet.created_at
                tweet_text = tweet.text
                articles.append(Article(datetime=tweet_creation_date,
                                        content=tweet_text,
                                        summary=tweet_text))

            last_id = new_tweets[-1].id

        return articles
