import unittest
from datetime import datetime
from datetime import timedelta

from drltrader.data import Scenario
from drltrader.data import DataRepository
from drltrader.data.ohlcv_data_repository import AlpacaOHLCVDataRepository
from drltrader.data.resampled_data_repository import ResampleDataRepository


class ResampleDataRepositoryTestCase(unittest.TestCase):

    def test_resample_retrieve_data_60m(self):
        # Arrange
        testing_scenario = Scenario(symbols=['SHOP', 'TSLA'],
                                    start_date=datetime.now() - timedelta(days=360),
                                    end_date=datetime.now() - timedelta(days=320),
                                    interval='5m')

        data_repository: DataRepository = ResampleDataRepository(AlpacaOHLCVDataRepository(), resampled_interval='60m')

        # Act
        dataframe = data_repository.retrieve_datas(testing_scenario)

        # Assert
        self.assertIsNotNone(dataframe)

    def test_resample_retrieve_data_1d(self):
        # Arrange
        testing_scenario = Scenario(symbols=['SHOP', 'TSLA'],
                                    start_date=datetime.now() - timedelta(days=360),
                                    end_date=datetime.now() - timedelta(days=320),
                                    interval='5m')

        data_repository: DataRepository = ResampleDataRepository(AlpacaOHLCVDataRepository(), resampled_interval='1d')

        # Act
        dataframe = data_repository.retrieve_datas(testing_scenario)

        # Assert
        self.assertIsNotNone(dataframe)


if __name__ == '__main__':
    unittest.main()