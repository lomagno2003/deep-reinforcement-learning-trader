import numpy as np

import tensorflow as tf
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import A2C
from matplotlib import pyplot as plt
from stable_baselines.common.callbacks import EvalCallback
from stable_baselines.common.callbacks import BaseCallback

from drltrader.data.data_provider import DataProvider
from drltrader.data.scenario import Scenario
from drltrader.envs.stock_environment import SingleStocksEnv


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
                 window_size: int = 12,
                 prices_feature_name: str = 'Low',
                 signal_feature_names: list = ['Low', 'Volume', 'SMA', 'RSI', 'OBV']):
        self.first_layer_size = first_layer_size
        self.second_layer_size = second_layer_size
        self.window_size = window_size
        self.prices_feature_name = prices_feature_name
        self.signal_feature_names = signal_feature_names


class Brain:
    def __init__(self,
                 data_provider: DataProvider = DataProvider(),
                 brain_configuration: BrainConfiguration = BrainConfiguration()):
        self._data_provider = data_provider
        self._model = None
        self._brain_configuration = brain_configuration

    def learn(self,
              training_scenario: Scenario,
              testing_scenario: Scenario = None,
              total_timesteps: int = 25000):
        # TODO: Clean this
        training_environment = self._build_environment(training_scenario)
        testing_environment = self._build_environment(testing_scenario) \
            if testing_scenario is not None else training_environment

        env = DummyVecEnv([lambda: training_environment])

        if self._model is None:
            self._init_model(env)
        else:
            self._model.set_env(env)

        eval_callback = EvalCallback(testing_environment,
                                     best_model_save_path='./logs/',
                                     log_path='./logs/',
                                     eval_freq=500,
                                     deterministic=True,
                                     render=False,
                                     callback_on_new_best=CustomCallback(),
                                     verbose=1)

        self._model.learn(total_timesteps=total_timesteps, callback=eval_callback)

    def _init_model(self, env):
        policy_kwargs = dict(act_fun=tf.nn.tanh,
                             net_arch=['lstm',
                                       self._brain_configuration.first_layer_size,
                                       self._brain_configuration.second_layer_size])

        self._model = A2C('MlpLstmPolicy', env, verbose=1, policy_kwargs=policy_kwargs)

    def test(self,
             testing_scenario: Scenario):
        env = self._build_environment(scenario=testing_scenario)

        obs = env.reset()
        while True:
            obs = obs[np.newaxis, ...]
            action, _states = self._model.predict(obs)
            obs, rewards, done, info = env.step(action)
            if done:
                print("info", info)
                break

        plt.figure(figsize=(15, 6))
        plt.cla()
        env.render_all()
        plt.show()

        return info

    def _build_environment(self, scenario: Scenario):
        symbol_dataframe = self._data_provider.retrieve_data(scenario)
        env = SingleStocksEnv(df=symbol_dataframe,
                              window_size=self._brain_configuration.window_size,
                              frame_bound=(self._brain_configuration.window_size, len(symbol_dataframe.index) - 1),
                              prices_feature_name=self._brain_configuration.prices_feature_name,
                              signal_features_names=self._brain_configuration.signal_feature_names)

        return env
