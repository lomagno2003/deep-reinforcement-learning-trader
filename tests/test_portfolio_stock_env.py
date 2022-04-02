import unittest
import random
from datetime import datetime
from datetime import timedelta

from drltrader.data.scenario import Scenario
from drltrader.envs.portfolio_stocks_env import PortfolioStocksEnv
from drltrader.data.data_provider import DataProvider


class PortfolioStocksEnvTestCase(unittest.TestCase):
    def test_init(self):
        # Arrange
        dataframe_per_symbol = self._build_testing_dataframe_per_symbol()
        initial_portfolio_allocation = {'TSLA': 1.0}

        # Act
        environment = PortfolioStocksEnv(window_size=8,
                                         frame_bound=(8, 100),
                                         dataframe_per_symbol=dataframe_per_symbol,
                                         initial_portfolio_allocation=initial_portfolio_allocation)

        # Assert
        self.assertIsNotNone(environment)

    def test_current_portfolio_value(self):
        # Arrange
        environment: PortfolioStocksEnv = self._build_testing_environment()

        # Act
        current_portfolio_value = environment.current_portfolio_value()

        # Assert
        self.assertIsNotNone(current_portfolio_value)

    def test_current_profit(self):
        # Arrange
        environment: PortfolioStocksEnv = self._build_testing_environment()

        # Act
        current_profit = environment.current_profit()

        # Assert
        self.assertIsNotNone(current_profit)

    def test_transfer_allocations(self):
        # Arrange
        environment: PortfolioStocksEnv = self._build_testing_environment()

        # Act
        environment._transfer_allocations('TSLA', 'AAPL', environment._current_tick)
        current_profit = environment.current_profit()

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
        current_profit = environment.current_profit()
        self.assertNotEqual(1.0, current_profit)

    def _build_testing_dataframe_per_symbol(self):
        tsla_scenario = Scenario(symbol='TSLA',
                                 start_date=datetime.now() - timedelta(days=30),
                                 end_date=datetime.now())
        msft_scenario = Scenario(symbol='MSFT',
                                 start_date=datetime.now() - timedelta(days=30),
                                 end_date=datetime.now())
        aapl_scenario = Scenario(symbol='AAPL',
                                 start_date=datetime.now() - timedelta(days=30),
                                 end_date=datetime.now())

        tsla_dataframe = DataProvider().retrieve_data(tsla_scenario)
        msft_dataframe = DataProvider().retrieve_data(msft_scenario)
        aapl_dataframe = DataProvider().retrieve_data(aapl_scenario)

        dataframe_per_symbol = {
            'TSLA': tsla_dataframe,
            'MSFT': msft_dataframe,
            'AAPL': aapl_dataframe
        }

        return dataframe_per_symbol

    def _build_testing_environment(self) -> PortfolioStocksEnv:
        dataframe_per_symbol = self._build_testing_dataframe_per_symbol()
        initial_portfolio_allocation = {'TSLA': 1.0}

        environment = PortfolioStocksEnv(window_size=8,
                                         frame_bound=(8, 100),
                                         dataframe_per_symbol=dataframe_per_symbol,
                                         initial_portfolio_allocation=initial_portfolio_allocation)

        return environment


if __name__ == '__main__':
    unittest.main()
