import unittest
from datetime import datetime, timedelta

from gym.spaces import Box
from stable_baselines3 import A2C
import torch as th

from drltrader.data import Scenario
from drltrader.data.cached_data_repository import CachedDataRepository
from drltrader.data.indicators_data_repository import IndicatorsDataRepository
from drltrader.data.ohlcv_data_repository import AlpacaOHLCVDataRepository
from drltrader.envs.portfolio_features_extractor import PortfolioFeaturesExtractor
from drltrader.envs.portfolio_stocks_env import PortfolioStocksEnv


class PortfolioFeaturesExtractorTestCase(unittest.TestCase):
    def test_forward(self):
        # Arrange
        self._initialize_testing_data()

        # Act
        result = self._features_extractor.forward(th.as_tensor([self._testing_observation]).float())

        # Assert
        self.assertIsNotNone(result)

    def test_a2c_policy(self):
        # Arrange
        self._initialize_testing_data()
        self._initialize_a2c_policy()

        # Act
        self._a2c_model.learn(100)

    def _initialize_testing_data(self):
        scenario = Scenario(symbols=['TSLA', 'MSFT', 'AAPL'],
                            start_date=datetime.fromtimestamp(1647777600),
                            end_date=datetime.fromtimestamp(1653048000),
                            interval='1d')
        data_repository = CachedDataRepository(IndicatorsDataRepository(AlpacaOHLCVDataRepository()))
        dataframe_per_symbol = data_repository.retrieve_datas(scenario)
        initial_portfolio_allocation = {'TSLA': 1.0}

        self._env = PortfolioStocksEnv(window_size=16,
                                       dataframe_per_symbol=dataframe_per_symbol,
                                       initial_portfolio_allocation=initial_portfolio_allocation)

        self._testing_observation = self._env.reset()
        self._features_extractor = PortfolioFeaturesExtractor(observation_space=self._env.observation_space)

    def _initialize_a2c_policy(self):
        policy_kwargs = dict(
            features_extractor_class=PortfolioFeaturesExtractor,
            features_extractor_kwargs=dict(),
        )

        self._a2c_model = A2C('MlpPolicy', self._env, verbose=0, policy_kwargs=policy_kwargs)


if __name__ == '__main__':
    unittest.main()
