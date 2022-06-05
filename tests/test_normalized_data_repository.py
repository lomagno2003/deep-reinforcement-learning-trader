import unittest

from datetime import datetime
from datetime import timedelta
import pandas as pd
import numpy as np

from drltrader.data import DataRepository
from drltrader.data import Scenario
from drltrader.data.normalized_data_repository import NormalizedDataRepository
from drltrader.data.static_data_repository import StaticDataRepository


class NormalizedDataRepositoryTestCase(unittest.TestCase):

    def test_resample_retrieve_data(self):
        # Arrange
        testing_scenario = Scenario(symbols=['FOO'],
                                    start_date=datetime.now() - timedelta(days=360),
                                    end_date=datetime.now() - timedelta(days=320),
                                    interval='5m')
        data_repository: DataRepository = self._initialize_data_repository()

        data_repository: DataRepository = NormalizedDataRepository(data_repository, excluded_columns=['column_d'])

        # Act
        dataframe_per_symbol = data_repository.retrieve_datas(testing_scenario)

        # Assert
        foo_dataset = dataframe_per_symbol['FOO']

        self.assertIsNotNone(foo_dataset)
        self.assertTrue(np.logical_and(
            np.array(list(foo_dataset['column_a'])) >= -1.0,
            np.array(list(foo_dataset['column_a'])) <= 1.0
        ).all())
        self.assertTrue(np.logical_and(
            np.array(list(foo_dataset['column_b'])) >= -1.0,
            np.array(list(foo_dataset['column_b'])) <= 1.0
        ).all())
        self.assertTrue(np.logical_and(
            np.array(list(foo_dataset['column_c'])) >= -1.0,
            np.array(list(foo_dataset['column_c'])) <= 1.0
        ).all())
        self.assertTrue(np.logical_or(
            np.array(list(foo_dataset['column_d'])) < -1.0,
            np.array(list(foo_dataset['column_d'])) > 1.0
        ).any())

    def _initialize_data_repository(self):
        df = pd.DataFrame()
        df['time'] = pd.date_range('08/12/2021',
                                   periods=30,
                                   freq='3S')
        df['column_a'] = range(15, -15, -1)
        df['column_b'] = range(0, -30, -1)
        df['column_c'] = range(30, 0, -1)
        df['column_d'] = range(30, -30, -2)
        df.set_index('time', inplace=True)

        return StaticDataRepository(dataframe_per_symbol={'FOO': df})


if __name__ == '__main__':
    unittest.main()