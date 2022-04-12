import unittest
from datetime import datetime
from datetime import timedelta
import pandas as pd

from drltrader.data import Scenario
from drltrader.data.composite_data_repository import CompositeDataRepository
from drltrader.data.static_data_repository import StaticDataRepository


class SentimentDataRepositoryTestCase(unittest.TestCase):
    def test_retrieve_datas(self):
        # Arrange
        testing_scenario = Scenario(symbols=['FOO'],
                                    start_date=datetime.now() - timedelta(days=30),
                                    end_date=datetime.now())
        data_repository: CompositeDataRepository = CompositeDataRepository(data_repositories=[
            self._initialize_data_repository_A(),
            self._initialize_data_repository_B()
        ])

        # Act
        dataframe_per_symbol = data_repository.retrieve_datas(testing_scenario)

        # Assert
        self.assertIsNotNone(dataframe_per_symbol)

    def _initialize_data_repository_A(self):
        df = pd.DataFrame()
        df['time'] = pd.date_range('08/12/2021',
                                   periods=5,
                                   freq='10S')
        df['values_a'] = range(0, 5)
        df.set_index('time', inplace=True)

        return StaticDataRepository(dataframe_per_symbol={'FOO': df})

    def _initialize_data_repository_B(self):
        df = pd.DataFrame()
        df['time'] = pd.date_range('08/12/2021',
                                   periods=15,
                                   freq='3S')
        df['values_b'] = range(0, -15, -1)
        df.set_index('time', inplace=True)

        return StaticDataRepository(dataframe_per_symbol={'FOO': df})


if __name__ == '__main__':
    unittest.main()