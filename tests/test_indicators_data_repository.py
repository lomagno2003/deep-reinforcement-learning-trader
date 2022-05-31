import unittest
from datetime import datetime
from datetime import timedelta

from drltrader.data import Scenario
from drltrader.data import DataRepository
from drltrader.data.ohlcv_data_repository import AlpacaOHLCVDataRepository
from drltrader.data.indicators_data_repository import IndicatorsDataRepository


class IndicatorsDataRepositoryTestCase(unittest.TestCase):

    def test_indicators_retrieve_data(self):
        # Arrange
        testing_scenario = Scenario(symbols=['SHOP', 'TSLA'],
                                    start_date=datetime.now() - timedelta(days=360),
                                    end_date=datetime.now() - timedelta(days=320),
                                    interval='30m')

        data_repository: DataRepository = IndicatorsDataRepository(AlpacaOHLCVDataRepository())

        # Act
        dataframe = data_repository.retrieve_datas(testing_scenario)

        # Assert
        self.assertIsNotNone(dataframe)


if __name__ == '__main__':
    unittest.main()