import unittest
import random
from datetime import datetime
from datetime import timedelta

from drltrader.data import Scenario
from drltrader.data.scenarios import Scenarios
from drltrader.envs.portfolio_stocks_env import PortfolioStocksEnv
from drltrader.data.ohlcv_data_repository import OHLCVDataRepository
from drltrader.data.ohlcv_data_repository import AlpacaOHLCVDataRepository
from drltrader.data.indicators_data_repository import IndicatorsDataRepository
from drltrader.data.cached_data_repository import CachedDataRepository


class PortfolioStocksEnvTestCase(unittest.TestCase):
    def test_init(self):
        # Arrange
        dataframe_per_symbol = self._build_testing_dataframe_per_symbol()
        initial_portfolio = {'TSLA': 1.0}

        # Act
        environment = PortfolioStocksEnv(window_size=8,
                                         dataframe_per_symbol=dataframe_per_symbol,
                                         initial_portfolio=initial_portfolio)

        # Assert
        self.assertIsNotNone(environment)

    def test_current_portfolio_value(self):
        # Arrange
        environment: PortfolioStocksEnv = self._build_testing_environment()

        # Act
        current_portfolio_value = environment.portfolio_value()

        # Assert
        self.assertIsNotNone(current_portfolio_value)

    def test_current_profit(self):
        # Arrange
        environment: PortfolioStocksEnv = self._build_testing_environment()

        # Act
        current_profit = environment.profit()

        # Assert
        self.assertIsNotNone(current_profit)

    def test_transfer_allocations(self):
        # Arrange
        environment: PortfolioStocksEnv = self._build_testing_environment()

        # Act
        environment._transfer_allocations('TSLA', 'long', 'AAPL', 'short')
        current_profit = environment.profit()

        # Assert
        self.assertIsNotNone(current_profit)

    def test_step(self):
        # Arrange
        environment: PortfolioStocksEnv = self._build_testing_environment()

        # Act
        environment.reset()

        while True:
            action = random.choice(list(range(0, 3)))
            obs, rewards, done, info = environment.step(action)
            if done:
                break

        # Assert
        current_profit = environment.profit()
        self.assertNotEqual(1.0, current_profit)

    def test_append_data(self):
        # Arrange
        data_repository: OHLCVDataRepository = IndicatorsDataRepository(AlpacaOHLCVDataRepository())
        first_scenario = Scenario(symbols=['TSLA', 'MSFT', 'AAPL'],
                                  start_date=datetime.now() - timedelta(days=60),
                                  end_date=datetime.now() - timedelta(days=30),
                                  interval='15m')
        second_scenario = Scenario(symbols=['TSLA', 'MSFT', 'AAPL'],
                                   start_date=datetime.now() - timedelta(days=45),
                                   end_date=datetime.now() - timedelta(days=15),
                                   interval='15m')
        first_dataframe_per_symbol = data_repository.retrieve_datas(first_scenario)
        second_dataframe_per_symbol = data_repository.retrieve_datas(second_scenario)

        initial_portfolio = {'TSLA': 1.0}

        environment = PortfolioStocksEnv(window_size=8,
                                         dataframe_per_symbol=first_dataframe_per_symbol,
                                         initial_portfolio=initial_portfolio)

        # Act
        while True:
            action = random.choice(list(range(0, 3)))
            obs, rewards, done, info = environment.step(action)
            if done:
                break

        self.assertTrue(done)
        environment.append_data(second_dataframe_per_symbol)

        # Assert
        obs, rewards, done, info = environment.step(action)
        self.assertFalse(done)

    def _build_testing_dataframe_per_symbol(self):
        scenario = Scenarios.last_market_week(['TSLA', 'AAPL'], '5m')
        data_repository = CachedDataRepository(IndicatorsDataRepository(AlpacaOHLCVDataRepository()))
        dataframe_per_symbol = data_repository.retrieve_datas(scenario)

        return dataframe_per_symbol

    def _build_testing_environment(self) -> PortfolioStocksEnv:
        dataframe_per_symbol = self._build_testing_dataframe_per_symbol()
        initial_portfolio = {'TSLA': 1.0}

        environment = PortfolioStocksEnv(window_size=8,
                                         dataframe_per_symbol=dataframe_per_symbol,
                                         initial_portfolio=initial_portfolio)

        return environment


if __name__ == '__main__':
    unittest.main()
