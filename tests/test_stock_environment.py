import unittest
from datetime import datetime
from datetime import timedelta
from drltrader.envs.stock_environment import SingleStocksEnv
from drltrader.data.data_provider import DataProvider


class StockEnvironmentTestCase(unittest.TestCase):
    def test_init(self):
        symbol_dataframe = DataProvider().retrieve_data(symbol='TSLA',
                                                        start_date=datetime.now(),
                                                        end_date=datetime.now() - timedelta(days=90),
                                                        interval='1d')

        environment = SingleStocksEnv(df=symbol_dataframe,
                                      window_size=12,
                                      frame_bound=(12, len(symbol_dataframe.index) - 1))

        self.assertIsNotNone(environment)


if __name__ == '__main__':
    unittest.main()
