import unittest
import random
from datetime import datetime
from datetime import timedelta

from drltrader.data.scenario import Scenario
from drltrader.envs.portfolio_stocks_env import PortfolioStocksEnv
from drltrader.data.ohlcv_data_repository import OHLCVDataRepository


class PortfolioStocksEnvTestCase(unittest.TestCase):
    def test_init(self):
        # Arrange
        dataframe_per_symbol = self._build_testing_dataframe_per_symbol()
        initial_portfolio_allocation = {'TSLA': 1.0}

        # Act
        environment = PortfolioStocksEnv(window_size=8,
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

    def test_append_data(self):
        # Arrange
        data_repository: OHLCVDataRepository = OHLCVDataRepository()
        first_scenario = Scenario(symbols=['TSLA', 'MSFT', 'AAPL'],
                                  start_date=datetime.now() - timedelta(days=60),
                                  end_date=datetime.now() - timedelta(days=30))
        second_scenario = Scenario(symbols=['TSLA', 'MSFT', 'AAPL'],
                                   start_date=datetime.now() - timedelta(days=45),
                                   end_date=datetime.now() - timedelta(days=15))
        first_dataframe_per_symbol = data_repository.retrieve_datas(first_scenario)
        second_dataframe_per_symbol = data_repository.retrieve_datas(second_scenario)

        initial_portfolio_allocation = {'TSLA': 1.0}

        environment = PortfolioStocksEnv(window_size=8,
                                         dataframe_per_symbol=first_dataframe_per_symbol,
                                         initial_portfolio_allocation=initial_portfolio_allocation)

        # Act
        while True:
            action = random.choice(list(range(0, 3)))
            obs, rewards, done, info = environment.step(action)
            if done:
                break

        self.assertIsNone(obs)
        environment.append_data(second_dataframe_per_symbol)

        # Assert
        obs, rewards, done, info = environment.step(action)
        self.assertIsNotNone(obs)

    def _build_testing_dataframe_per_symbol(self):
        scenario = Scenario(symbols=['TSLA', 'MSFT', 'AAPL'],
                            start_date=datetime.now() - timedelta(days=30),
                            end_date=datetime.now())

        dataframe_per_symbol = OHLCVDataRepository().retrieve_datas(scenario)

        return dataframe_per_symbol

    def _build_testing_environment(self) -> PortfolioStocksEnv:
        dataframe_per_symbol = self._build_testing_dataframe_per_symbol()
        initial_portfolio_allocation = {'TSLA': 1.0}

        environment = PortfolioStocksEnv(window_size=8,
                                         dataframe_per_symbol=dataframe_per_symbol,
                                         initial_portfolio_allocation=initial_portfolio_allocation)

        return environment


if __name__ == '__main__':
    unittest.main()
