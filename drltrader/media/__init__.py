from datetime import datetime


class Media:
    def __init__(self,
                 datetime: datetime = None,
                 id: str = None,
                 content: str = None,
                 summary: str = None,
                 sentiment: str = None):
        self.datetime = datetime
        self.id = id
        self.content = content
        self.summary = summary
        self.sentiment = sentiment


class TickerMediaRepository:
    def get_column_prefix(self):
        pass

    def find_medias(self, ticker: str, from_date: datetime, to_date: datetime) -> list:
        pass
