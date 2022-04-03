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

    def test_retrieve_data_without_end_date(self):
        # Arrange
        data_provider: DataProvider = DataProvider()

        no_end_date_scenario = Scenario(symbol='TSLA',
                                        interval='1d',
                                        start_date=datetime.now() - timedelta(days=30))
        now_scenario = Scenario(symbol='TSLA',
                                interval='1d',
                                start_date=datetime.now() - timedelta(days=30),
                                end_date=datetime.now())

        # Act
        no_end_date_dataframe = data_provider.retrieve_data(no_end_date_scenario)
        now_dataframe = data_provider.retrieve_data(now_scenario)

        # Assert
        self.assertIsNotNone(no_end_date_dataframe)
        self.assertIsNotNone(now_dataframe)


if __name__ == '__main__':
    unittest.main()