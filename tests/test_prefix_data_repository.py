import unittest
from datetime import datetime
from datetime import timedelta

import pandas as pd

from drltrader.data import DataRepository
from drltrader.data import Scenario
from drltrader.data.prefix_data_repository import PrefixDataRepository
from drltrader.data.static_data_repository import StaticDataRepository


class PrefixDataRepositoryTestCase(unittest.TestCase):

    def test_resample_retrieve_data(self):
        # Arrange
        testing_scenario = Scenario(symbols=['FOO'],
                                    start_date=datetime.now() - timedelta(days=360),
                                    end_date=datetime.now() - timedelta(days=320),
                                    interval='5m')
        data_repository: DataRepository = self._initialize_data_repository()

        data_repository: DataRepository = PrefixDataRepository(data_repository, column_prefix='5m')

        # Act
        dataframe = data_repository.retrieve_datas(testing_scenario)

        # Assert
        self.assertIsNotNone(dataframe)

    def _initialize_data_repository(self):
        df = pd.DataFrame()
        df['time'] = pd.date_range('08/12/2021',
                                   periods=15,
                                   freq='3S')
        df['values_b'] = range(0, -15, -1)
        df.set_index('time', inplace=True)

        return StaticDataRepository(dataframe_per_symbol={'FOO': df})


if __name__ == '__main__':
    unittest.main()