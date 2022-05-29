import unittest
import pandas as pd
from datetime import datetime
from datetime import timedelta

from drltrader.data import Scenario
from drltrader.data.static_data_repository import StaticDataRepository
from drltrader.data.cached_data_repository import CachedDataRepository


class SentimentDataRepositoryTestCase(unittest.TestCase):
    def __init__(self, method_name: str):
        super(SentimentDataRepositoryTestCase, self).__init__(method_name)

        self._source_data_repository = self._initialize_data_repository()
        self._cached_data_repository = CachedDataRepository(source_data_repository=self._source_data_repository)
        self._testing_scenario = Scenario(symbols=['FOO'],
                                          start_date=datetime.now() - timedelta(days=30),
                                          end_date=datetime.now())

    def test_retrieve_cached_data(self):
        # Arrange
        fixed_time_scenario = Scenario(symbols=['FOO'],
                                       start_date=datetime.fromtimestamp(1500000000),
                                       end_date=datetime.fromtimestamp(1600000000))

        # Act
        dataframe_per_symbol = self._cached_data_repository.retrieve_datas(fixed_time_scenario)

        # Assert
        self.assertIsNotNone(dataframe_per_symbol)

    def test_retrieve_uncached_data(self):
        # Arrange
        fixed_time_scenario = Scenario(symbols=['FOO'],
                                       start_date=datetime.fromtimestamp(1500000000),
                                       end_date=datetime.now())

        # Act
        dataframe_per_symbol = self._cached_data_repository.retrieve_datas(fixed_time_scenario)

        # Assert
        self.assertIsNotNone(dataframe_per_symbol)

    def _initialize_data_repository(self):
        df = pd.DataFrame()
        df['time'] = pd.date_range('08/12/2021',
                                   periods=5,
                                   freq='10S')
        df['values_a'] = range(0, 5)
        df.set_index('time', inplace=True)

        return StaticDataRepository(dataframe_per_symbol={'FOO': df})


if __name__ == '__main__':
    unittest.main()
