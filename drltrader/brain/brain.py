import numpy as np
import json

import tensorflow as tf
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines import A2C
from matplotlib import pyplot as plt
from stable_baselines.common.callbacks import EvalCallback
from stable_baselines.common.callbacks import BaseCallback

from drltrader.data.data_provider import DataProvider
from drltrader.data.scenario import Scenario
from drltrader.envs.single_stock_env import SingleStockEnv
from drltrader.envs.portfolio_stocks_env import PortfolioStocksEnv


# FIXME: This callback doesn't work with PorfolioStocksEnv
class CustomCallback(BaseCallback):
    def _on_step(self):
        if not hasattr(self, 'infos'):
            self.infos = []

        self.infos.append(self.locals['info'])

        initial_price = self.model.env.envs[0].prices[self.model.env.envs[0].frame_bound[0]]
        final_price = self.model.env.envs[0].prices[self.model.env.envs[0].frame_bound[1] - 1]
        benchmark_profit = 1.0 + (final_price - initial_price) / initial_price

        print(f"Benchmark: {benchmark_profit}")
        print(f"Profits: {list(map(lambda x: x['total_profit'], self.infos))}")
        print(f"Rewards: {list(map(lambda x: x['total_reward'], self.infos))}")


class BrainConfiguration:
    def __init__(self,
                 first_layer_size: int = 256,
                 second_layer_size: int = 256,
                 window_size: int = 3,
                 prices_feature_name: str = 'Low',
                 signal_feature_names: list = ['Low', 'Volume'],
                 use_normalized_observations: bool = True):
        self.first_layer_size = first_layer_size
        self.second_layer_size = second_layer_size
        self.window_size = window_size
        self.prices_feature_name = prices_feature_name
        self.use_normalized_observations = use_normalized_observations
        self.signal_feature_names = signal_feature_names

    def __str__(self):
        return json.dumps(self.__dict__)


class Brain:
    def __init__(self,
                 data_provider: DataProvider = DataProvider(),
                 brain_configuration: BrainConfiguration = BrainConfiguration()):
        # Store Configurations
        self._data_provider = data_provider
        self._brain_configuration = brain_configuration

        # Initialize Runtime Variables
        self._model = None
        self._using_multi_symbol_scenarios = None

    def learn(self,
              training_scenario: Scenario,
              testing_scenario: Scenario = None,
              total_timesteps: int = 1000):
        training_environment = self._build_environment(training_scenario)

        if self._model is None:
            self._init_model(training_environment)
        else:
            self._model.set_env(training_environment)

        eval_callback = None
        if testing_scenario is not None:
            testing_environment = self._build_environment(testing_scenario)

            eval_callback = EvalCallback(testing_environment,
                                         best_model_save_path='./logs/',
                                         log_path='./logs/',
                                         eval_freq=500,
                                         deterministic=True,
                                         render=False,
                                         callback_on_new_best=CustomCallback(),
                                         verbose=0)

        self._model.learn(total_timesteps=total_timesteps, callback=eval_callback)

    def _init_model(self, env):
        policy_kwargs = dict(act_fun=tf.nn.tanh,
                             net_arch=['lstm',
                                       self._brain_configuration.first_layer_size,
                                       self._brain_configuration.second_layer_size])

        self._model = A2C('MlpLstmPolicy', env, verbose=0, policy_kwargs=policy_kwargs)

    def test(self,
             testing_scenario: Scenario,
             render=True):
        testing_environment = self._build_environment(scenario=testing_scenario)

        obs = testing_environment.reset()

        # FIXME: This needs to be done because the VecEnvs auto-calls the reset on done==true
        if self._brain_configuration.use_normalized_observations:
            internal_environment = testing_environment.venv.envs[0]
        else:
            internal_environment = testing_environment.envs[0]

        internal_environment.disable_reset()

        while True:
            obs = obs[np.newaxis, ...]
            action, _states = self._model.predict(obs[0])
            obs, rewards, done, info = testing_environment.step(action)
            if done[0]:
                break

        if render:
            plt.figure(figsize=(15, 6))
            plt.cla()
            internal_environment.render_all()
            plt.show()

        return info[0]

    def _build_environment(self, scenario: Scenario):
        env = None
        if scenario.symbols is not None:
            env = self._build_portfolio_stock_scenario(scenario)
        else:
            env = self._build_single_stock_scenario(scenario)

        if self._brain_configuration.use_normalized_observations:
            return VecNormalize(DummyVecEnv([lambda: env]))
        else:
            return DummyVecEnv([lambda: env])

    def _build_single_stock_scenario(self, scenario: Scenario):
        symbol_dataframe = self._data_provider.retrieve_data(scenario)
        env = SingleStockEnv(df=symbol_dataframe,
                             window_size=self._brain_configuration.window_size,
                             frame_bound=(self._brain_configuration.window_size, len(symbol_dataframe.index) - 1),
                             prices_feature_name=self._brain_configuration.prices_feature_name,
                             signal_feature_names=self._brain_configuration.signal_feature_names)

        return env

    def _build_portfolio_stock_scenario(self, scenario: Scenario):
        dataframe_per_symbol = self._data_provider.retrieve_datas(scenario)
        first_symbol = list(dataframe_per_symbol.keys())[0]

        start_tick = self._brain_configuration.window_size
        end_tick = len(dataframe_per_symbol[first_symbol].index) - 1
        initial_portfolio_allocation = {first_symbol: 1.0} # FIXME: This is not configurable

        env = PortfolioStocksEnv(initial_portfolio_allocation=initial_portfolio_allocation,
                                 dataframe_per_symbol=dataframe_per_symbol,
                                 window_size=self._brain_configuration.window_size,
                                 frame_bound=(start_tick, end_tick),
                                 prices_feature_name=self._brain_configuration.prices_feature_name,
                                 signal_feature_names=self._brain_configuration.signal_feature_names)

        return env
