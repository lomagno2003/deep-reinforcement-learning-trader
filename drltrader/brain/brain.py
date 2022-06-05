import numpy as np
import json
import logging
import logging.config
import time

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3 import A2C

from drltrader.data import DataRepository
from drltrader.data.ohlcv_data_repository import AlpacaOHLCVDataRepository
from drltrader.data.indicators_data_repository import IndicatorsDataRepository
from drltrader.data import Scenario
from drltrader.envs.portfolio_stocks_env import PortfolioStocksEnv
from drltrader.envs.portfolio_features_extractor import PortfolioFeaturesExtractor
from drltrader.observers import Observer

logging.config.fileConfig('log.ini', disable_existing_loggers=False)
logger = logging.getLogger(__name__)


class BrainConfiguration:
    def __init__(self,
                 f_cnn1_kernel_count: int = 32,
                 f_cnn1_kernel_size: int = 8,
                 f_pool1_size: int = 2,
                 f_pool1_stride: int = 8,
                 f_cnn2_kernel_count: int = 64,
                 f_cnn2_kernel_size: int = 4,
                 f_pool2_size: int = 2,
                 f_pool2_stride: int = 8,
                 f_linear1_size: int = 64,
                 f_linear2_size: int = 64,
                 f_pi_net_arch: list = [64, 64],
                 f_vf_net_arch: list = [64, 64],
                 window_size: int = 12,
                 prices_feature_name: str = 'Low',
                 signal_feature_names: list = ['Low', 'Volume'],
                 use_normalized_observations: bool = False,
                 interval: str = '5m',
                 symbols: list = ['SPY']):
        self.f_cnn1_kernel_count = f_cnn1_kernel_count
        self.f_cnn1_kernel_size = f_cnn1_kernel_size
        self.f_pool1_size = f_pool1_size
        self.f_pool1_stride = f_pool1_stride
        self.f_cnn2_kernel_count = f_cnn2_kernel_count
        self.f_cnn2_kernel_size = f_cnn2_kernel_size
        self.f_pool2_size = f_pool2_size
        self.f_pool2_stride = f_pool2_stride
        self.f_linear1_size = f_linear1_size
        self.f_linear2_size = f_linear2_size

        self.f_pi_net_arch = f_pi_net_arch
        self.f_vf_net_arch = f_vf_net_arch

        self.window_size = window_size
        self.prices_feature_name = prices_feature_name
        self.use_normalized_observations = use_normalized_observations
        self.signal_feature_names = signal_feature_names
        self.interval = interval
        self.symbols = symbols

    def __str__(self):
        return json.dumps(self.__dict__)


class Brain:
    def __init__(self,
                 data_repository: DataRepository = IndicatorsDataRepository(AlpacaOHLCVDataRepository()),
                 brain_configuration: BrainConfiguration = BrainConfiguration()):
        # Store Configurations
        self._data_repository = data_repository
        self._brain_configuration = brain_configuration

        # Initialize Runtime Variables
        self._observing = False
        self._model = None
        self._using_multi_symbol_scenarios = None

    def learn(self,
              training_scenario: Scenario,
              total_timesteps: int = 1000):
        training_environment = self._build_environment(training_scenario)

        if self._model is None:
            self._init_model(training_environment)
        else:
            self._model.set_env(training_environment)

        self._model.learn(total_timesteps=total_timesteps)

    def evaluate(self,
                 testing_scenario: Scenario,
                 render: bool = False,
                 observer: Observer = None):
        _, _, info = self._analyze_scenario(testing_scenario, render=render, observer=observer)
        return info

    def start_observing(self, scenario: Scenario, observer: Observer = None):
        scenario = self._build_scenario(scenario)
        # TODO: This is done only for PortfolioStocksEnv
        # TODO: Validate that scenario is without end_date
        internal_environment, environment, info = self._analyze_scenario(scenario, render=False)

        internal_environment.observe(observer)

        self._observing = True
        while self._observing:
            print("Running cycle...")
            logger.info("Running cycle...")
            new_dataframe_per_symbol = self._data_repository.retrieve_datas(scenario)
            internal_environment.append_data(dataframe_per_symbol=new_dataframe_per_symbol)

            obs, rewards, done, info = internal_environment.get_step_outputs()

            while not done:
                obs = obs[np.newaxis, ...]
                action, _states = self._model.predict(obs)
                obs_arr, rewards_arr, done_arr, info_arr = environment.step(action)

                obs = obs_arr[0]
                rewards = rewards_arr[0]
                done = done_arr[0]
                info = info_arr[0]

                if done:
                    break

            logger.info("Cycle finished, sleeping")
            time.sleep(120)

        return info

    def stop_observing(self):
        self._observing = False

    def _init_model(self, env):
        policy_kwargs = dict(
            features_extractor_class=PortfolioFeaturesExtractor,
            features_extractor_kwargs=dict(
                f_cnn1_kernel_count=self._brain_configuration.f_cnn1_kernel_count,
                f_cnn1_kernel_size=self._brain_configuration.f_cnn1_kernel_size,
                f_cnn2_kernel_count=self._brain_configuration.f_cnn2_kernel_count,
                f_cnn2_kernel_size=self._brain_configuration.f_cnn2_kernel_size,
                f_linear1_size=self._brain_configuration.f_linear1_size,
                f_linear2_size=self._brain_configuration.f_linear2_size),
            net_arch=[dict(
                pi=self._brain_configuration.f_pi_net_arch,
                vf=self._brain_configuration.f_vf_net_arch)]
        )

        self._model = A2C('MlpPolicy',
                          env,
                          n_steps=30,
                          verbose=0,
                          policy_kwargs=policy_kwargs)

        # policy_kwargs = dict(net_arch=[dict(pi=[512, 512], vf=[512, 512])])
        # self._model = A2C('MlpPolicy', env, verbose=0, policy_kwargs=policy_kwargs)

    def _analyze_scenario(self,
                          scenario: Scenario,
                          render: bool = True,
                          observer: Observer = None):
        environment = self._build_environment(scenario=scenario)

        obs = environment.reset()

        # FIXME: This needs to be done because the VecEnvs auto-calls the reset on done==true
        if self._brain_configuration.use_normalized_observations:
            internal_environment = environment.venv.envs[0]
        else:
            internal_environment = environment.envs[0]

        internal_environment.disable_reset()
        internal_environment.observe(observer)

        while True:
            obs = obs[np.newaxis, ...]
            action, _states = self._model.predict(obs[0])
            obs, rewards, done, info = environment.step(action)
            if done[0]:
                break

        if render:
            internal_environment.render_all()

        return internal_environment, environment, info[0]

    def _build_scenario(self, scenario: Scenario):
        if scenario.interval is not None:
            logger.warning(f"Scenario already has an interval, overriding it with {self._brain_configuration.interval}")
        scenario = scenario.copy_with_interval(self._brain_configuration.interval)

        if scenario.symbols is not None:
            logger.warning(f"Scenario already has symbols, overriding it with {self._brain_configuration.symbols}")
        scenario = scenario.copy_with_symbols(self._brain_configuration.symbols)

        return scenario

    def _build_environment(self, scenario: Scenario):
        scenario = self._build_scenario(scenario)

        env = self._build_portfolio_stock_scenario(scenario)

        if self._brain_configuration.use_normalized_observations:
            return VecNormalize(DummyVecEnv([lambda: env]))
        else:
            return DummyVecEnv([lambda: env])

    def _build_portfolio_stock_scenario(self, scenario: Scenario):
        dataframe_per_symbol = self._data_repository.retrieve_datas(scenario)
        first_symbol = list(dataframe_per_symbol.keys())[0]
        initial_portfolio = {first_symbol: 1.0} # FIXME: This is not configurable

        env = PortfolioStocksEnv(initial_portfolio=initial_portfolio,
                                 dataframe_per_symbol=dataframe_per_symbol,
                                 window_size=self._brain_configuration.window_size,
                                 prices_feature_name=self._brain_configuration.prices_feature_name,
                                 signal_feature_names=self._brain_configuration.signal_feature_names)

        return env
