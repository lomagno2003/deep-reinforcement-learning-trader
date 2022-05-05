import unittest
from datetime import datetime
from datetime import timedelta

from drltrader.data import Scenario
from drltrader.data.ohlcv_data_repository import OHLCVDataRepository
from drltrader.data.ohlcv_data_repository import YahooOHLCVDataRepository
from drltrader.data.ohlcv_data_repository import AlpacaOHLCVDataRepository


class OHLCVDataRepositoryTestCase(unittest.TestCase):
    def test_yahoo_retrieve_data(self):
        # Arrange
        testing_scenario = Scenario(symbol='USD',
                                    start_date=datetime.now() - timedelta(days=30),
                                    end_date=datetime.now(),
                                    interval='15m')

        data_repository: OHLCVDataRepository = YahooOHLCVDataRepository()

        # Act
        dataframe = data_repository.retrieve_data(testing_scenario)

        # Assert
        self.assertIsNotNone(dataframe)

    def test_yahoo_retrieve_datas(self):
        # Arrange
        testing_scenario = Scenario(symbols=['TSLA', 'MELI'],
                                    start_date=datetime.now() - timedelta(days=30),
                                    end_date=datetime.now())

        data_repository: OHLCVDataRepository = YahooOHLCVDataRepository()

        # Act
        dataframe_per_symbol = data_repository.retrieve_datas(testing_scenario)

        # Assert
        self.assertIsNotNone(dataframe_per_symbol)

    def test_yahoo_retrieve_data_without_end_date(self):
        # Arrange
        data_repository: OHLCVDataRepository = YahooOHLCVDataRepository()

        no_end_date_scenario = Scenario(symbol='TSLA',
                                        interval='1d',
                                        start_date=datetime.now() - timedelta(days=30))
        now_scenario = Scenario(symbol='TSLA',
                                interval='1d',
                                start_date=datetime.now() - timedelta(days=30),
                                end_date=datetime.now())

        # Act
        no_end_date_dataframe = data_repository.retrieve_data(no_end_date_scenario)
        now_dataframe = data_repository.retrieve_data(now_scenario)

        # Assert
        self.assertIsNotNone(no_end_date_dataframe)
        self.assertIsNotNone(now_dataframe)

    def test_alpaca_retrieve_data(self):
        # Arrange
        testing_scenario = Scenario(symbol='SHOP',
                                    start_date=datetime.now() - timedelta(days=360),
                                    end_date=datetime.now() - timedelta(days=320),
                                    interval='30m')

        data_repository: OHLCVDataRepository = AlpacaOHLCVDataRepository()

        # Act
        dataframe = data_repository.retrieve_data(testing_scenario)

        # Assert
        self.assertIsNotNone(dataframe)

    def test_compare_alpaca_with_yahoo(self):
        # Arrange
        testing_scenario = Scenario(symbol='SHOP',
                                    start_date=datetime.now() - timedelta(days=30),
                                    end_date=datetime.now() - timedelta(days=1),
                                    interval='30m')

        alpaca_data_repository: AlpacaOHLCVDataRepository = AlpacaOHLCVDataRepository()
        yahoo_data_repository: OHLCVDataRepository = YahooOHLCVDataRepository()

        # Act
        alpaca_dataframe = alpaca_data_repository.retrieve_data(testing_scenario)
        yahoo_dataframe = yahoo_data_repository.retrieve_data(testing_scenario)

        # Assert
        self.assertIsNotNone(alpaca_dataframe)
        self.assertIsNotNone(yahoo_dataframe)


if __name__ == '__main__':
    unittest.main()