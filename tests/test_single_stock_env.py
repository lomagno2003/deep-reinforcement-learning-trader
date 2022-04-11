import unittest
from datetime import datetime
from datetime import timedelta

from drltrader.data import Scenario
from drltrader.envs.single_stock_env import SingleStockEnv
from drltrader.data.ohlcv_data_repository import OHLCVDataRepository


class SingleStockEnvTestCase(unittest.TestCase):
    def test_init(self):
        # Arrange
        testing_scenario = Scenario(symbol='TSLA',
                                    start_date=datetime.now() - timedelta(days=30),
                                    end_date=datetime.now())
        symbol_dataframe = OHLCVDataRepository().retrieve_data(testing_scenario)

        # Act
        environment = SingleStockEnv(df=symbol_dataframe,
                                     window_size=12,
                                     frame_bound=(12, len(symbol_dataframe.index) - 1),
                                     prices_feature_name='Close',
                                     signal_feature_names=['RSI_6'])

        # Assert
        self.assertIsNotNone(environment)


if __name__ == '__main__':
    unittest.main()
