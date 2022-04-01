import unittest
from datetime import datetime
from datetime import timedelta
import pandas as pd

from drltrader.data.data_provider import DataProvider
from drltrader.data.data_provider import Scenario


class DataProviderTestCase(unittest.TestCase):
    def test_retrieve_data(self):
        # Arrange
        testing_scenario = Scenario(symbol='TSLA',
                                    start_date=datetime.now() - timedelta(days=30),
                                    end_date=datetime.now())

        data_provider: DataProvider = DataProvider()

        # Act
        dataframe = data_provider.retrieve_data(testing_scenario)

        # Assert
        self.assertIsNotNone(dataframe)
        self.assertIsNotNone(data_provider.indicator_column_names)

        pd.set_option('display.max_columns', None)
        print(dataframe.head())


if __name__ == '__main__':
    unittest.main()