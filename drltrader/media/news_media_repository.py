import logging.config
import re
from datetime import datetime
import requests
from bs4 import BeautifulSoup
from dateutil import parser
from retry import retry
from transformers import PegasusTokenizer, PegasusForConditionalGeneration

from drltrader.media import Media, TickerMediaRepository

logging.config.fileConfig('log.ini', disable_existing_loggers=False)
logger = logging.getLogger(__name__)


class NewsMediaRepository(TickerMediaRepository):
    def __init__(self):
        self._init_summarization_model()

    def get_column_prefix(self):
        return 'news'

    def find_medias(self, ticker: str, from_date: datetime, to_date: datetime):
        # TODO: Filter by dates
        ticker_urls = self._search_for_ticker_news_urls(ticker)

        summaries = []
        for ticker_url in ticker_urls:
            ticker_media = self._get_media_content(ticker_url)
            ticker_media = self._summarize(ticker_media)

            summaries.append(ticker_media)

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
        include_list = ['yahoo']

        val = []
        for url in hrefs:
            starts_with_https = 'https://' in url
            does_not_contains_excluded_words = not any(exclude_word in url for exclude_word in exclude_list)
            contains_included_words = any(included_word in url for included_word in include_list)
            if starts_with_https and does_not_contains_excluded_words and contains_included_words:
                res = re.findall(r'(https?://\S+)', url)[0].split('&')[0]
                val.append(res)

        logger.debug(f"{len(list(set(val)))} urls found")

        return list(set(val))

    @retry(delay=2, tries=3)
    def _get_media_content(self, url):
        logger.debug(f"Fetching {url}")

        r = requests.get(url)
        soup = BeautifulSoup(r.text, 'html.parser')

        media_datetime = parser.parse(soup.find_all('time')[0]['datetime'])

        paragraphs = soup.find_all('p')
        text = [paragraph.text for paragraph in paragraphs]
        # TODO: Hardcoded value
        words = ' '.join(text).split(' ')[:400]
        media_content = ' '.join(words)

        return Media(datetime=media_datetime,
                     id=url,
                     content=media_content)

    def _summarize(self, media):
        logger.debug(f"Summarizing {media.id}")

        # TODO: Hardcoded Values
        input_ids = self.tokenizer.encode(media.content, return_tensors='pt', truncation=True, max_length=512)
        output = self.model.generate(input_ids, max_length=55, num_beams=5, early_stopping=True)
        media_summary = self.tokenizer.decode(output[0], skip_special_tokens=True)
        media.summary = media_summary

        return media



