import logging.config
import re
import requests
from bs4 import BeautifulSoup
from dateutil import parser
from retry import retry
from transformers import PegasusTokenizer, PegasusForConditionalGeneration

from drltrader.data.sentiment_data_repository import TickerFeedRepository
from drltrader.data.sentiment_data_repository import Article

logging.config.fileConfig('log.ini', disable_existing_loggers=False)
logger = logging.getLogger(__name__)


class NewsFeedRepository(TickerFeedRepository):
    def __init__(self):
        self._init_summarization_model()

    def find_articles(self, ticker: str):
        ticker_urls = self._search_for_ticker_news_urls(ticker)

        summaries = []
        for ticker_url in ticker_urls:
            ticker_article = self._get_article_content(ticker_url)
            ticker_article = self._summarize(ticker_article)

            summaries.append(ticker_article)

        return summaries

    def _init_summarization_model(self):
        model_name = "human-centered-summarization/financial-summarization-pegasus"
        self.tokenizer = PegasusTokenizer.from_pretrained(model_name)
        self.model = PegasusForConditionalGeneration.from_pretrained(model_name)

    def _search_for_ticker_news_urls(self, ticker):
        logger.info(f"Fetching urls for {ticker}")

        search_url = "https://www.google.com/search?q=yahoo+finance+{}&tbs=sbd:1,qdr:w&tbm=nws".format(ticker)
        r = requests.get(search_url)
        soup = BeautifulSoup(r.text, 'html.parser')
        atags = soup.find_all('a')
        hrefs = [link['href'] for link in atags]

        # TODO: Hardcoded value
        exclude_list = ['maps', 'policies', 'preferences', 'accounts', 'support', 'video', 'nasdaq']

        val = []
        for url in hrefs:
            if 'https://' in url and not any(exclude_word in url for exclude_word in exclude_list):
                res = re.findall(r'(https?://\S+)', url)[0].split('&')[0]
                val.append(res)

        logger.debug(f"{len(list(set(val)))} urls found")

        return list(set(val))

    @retry(delay=2, tries=3)
    def _get_article_content(self, url):
        logger.debug(f"Fetching {url}")

        r = requests.get(url)
        soup = BeautifulSoup(r.text, 'html.parser')

        article_datetime = parser.parse(soup.find_all('time')[0]['datetime'])

        paragraphs = soup.find_all('p')
        text = [paragraph.text for paragraph in paragraphs]
        # TODO: Hardcoded value
        words = ' '.join(text).split(' ')[:400]
        article_content = ' '.join(words)

        return Article(datetime=article_datetime,
                       url=url,
                       content=article_content)

    def _summarize(self, article):
        logger.debug(f"Summarizing {article.url}")

        # TODO: Hardcoded Values
        input_ids = self.tokenizer.encode(article.content, return_tensors='pt', truncation=True, max_length=512)
        output = self.model.generate(input_ids, max_length=55, num_beams=5, early_stopping=True)
        article_summary = self.tokenizer.decode(output[0], skip_special_tokens=True)
        article.summary = article_summary

        return article



