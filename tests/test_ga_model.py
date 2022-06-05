import unittest
import pandas as pd
import numpy as np

from drltrader.data.static_data_repository import StaticDataRepository
from drltrader.data.scenarios import Scenarios
from drltrader.brain.ga_model import GAModel, GAPolicy
from drltrader.envs.portfolio_stocks_env import PortfolioStocksEnv
from drltrader.envs.portfolio_features_extractor import PortfolioFeaturesExtractor


class GAModelTestCase(unittest.TestCase):
    def __init__(self, name):
        super(GAModelTestCase, self).__init__(name)

        self.training_scenario_multi_stock = Scenarios.last_market_weeks(start_week=3,
                                                                         end_week=1)
        self.testing_scenario_multi_stock = Scenarios.last_market_week()

    def test_init(self):
        # Act
        self._init_model()

        # Assert
        self.assertIsNotNone(self._model)

    def test_predict(self):
        # Arrange
        self._init_model()
        obs, _, _, _ = self._env.get_step_outputs()

        # Act
        prediction = self._model.predict(observation=obs)

        # Assert
        self.assertIsNotNone(prediction)

    def test_learn(self):
        # Arrange
        self._init_model()

        # Act
        self._model.learn(total_timesteps=1000)

    def _init_model(self):
        self._init_env()
        self._model = GAModel(env=self._env,
                              policy_kwargs=self._policy_kwargs())

    def _policy_kwargs(self):
        return dict(
            features_extractor_class=PortfolioFeaturesExtractor,
            features_extractor_kwargs=dict(
                f_cnn1_kernel_count=10,
                f_cnn1_kernel_size=10,
                f_cnn2_kernel_count=10,
                f_cnn2_kernel_size=10,
                f_linear1_size=10,
                f_linear2_size=10)
        )

    def _init_env(self) -> PortfolioStocksEnv:
        df = pd.DataFrame()
        df['time'] = pd.date_range('08/12/2021',
                                   periods=60,
                                   freq='3S')
        df['Close'] = np.concatenate([range(0, 30, 1), range(30, 0, -1)])
        df['RSI_4'] = np.concatenate([range(0, 30, 1), range(30, 0, -1)])
        df['RSI_16'] = np.concatenate([range(0, 30, 1), range(30, 0, -1)])
        df.set_index('time', inplace=True)

        data_repository = StaticDataRepository(dataframe_per_symbol={'FOO': df})

        dataframe_per_symbol = data_repository.retrieve_datas(None)
        initial_portfolio = {'FOO': 1.0}

        self._env = PortfolioStocksEnv(window_size=8,
                                       dataframe_per_symbol=dataframe_per_symbol,
                                       initial_portfolio=initial_portfolio)


if __name__ == '__main__':
    unittest.main()
