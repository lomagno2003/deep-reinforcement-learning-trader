import logging.config

import numpy as np

from drltrader.brain.brain import Brain, BrainConfiguration
from drltrader.brain.brain_repository_file import BrainRepositoryFile
from drltrader.data.data_repositories import DataRepositories
from drltrader.data.scenarios import Scenarios
from drltrader.observers.simple_observer import PrintEnvObserver

logging.config.fileConfig('log.ini', disable_existing_loggers=False)
logger = logging.getLogger(__name__)


# Manually architect a brain and re-train it over the same week
class TrainingBenchmarker:
    def __init__(self):
        self._brain_repository = BrainRepositoryFile()
        self._data_repository = DataRepositories.build_normalized_multi_time_interval_data_repository(
            exclude_normalized_columns=['5m_Close']
        )
        self._scenario = Scenarios.last_market_weeks(start_week=1)

    def run(self):
        # Find best brain
        best_brain = self.build_initial_brain_on_datetime()

        # Perform benchmarking
        while True:
            logger.info("Training brain")
            best_brain.learn(training_scenario=self._scenario,
                             total_timesteps=40000,
                             rendering_enabled=False)

    def build_initial_brain_on_datetime(self):
        logger.info(f"Build initial brain")
        best_brain_configuration: BrainConfiguration = BrainConfiguration(
            f_cnn1_kernel_count=64,
            f_cnn1_kernel_size=4,
            f_pool1_size=4,
            f_pool1_stride=2,
            f_cnn2_kernel_count=64,
            f_cnn2_kernel_size=4,
            f_pool2_size=4,
            f_pool2_stride=2,
            f_linear1_size=512,
            f_linear2_size=256,
            f_pi_net_arch=[256, 256],
            f_vf_net_arch=[256, 256],
            window_size=1,
            interval='15m',
            prices_feature_name='5m_Close',
            signal_feature_names=self._data_repository.get_columns_per_symbol(),
            # signal_feature_names=[
            #     '60m_RSI_4'
            # ],
            # symbols=[
            #     'TDOC', 'ETSY', 'MELI', 'SE', 'SQ', 'DIS', 'TSLA', 'AAPL', 'MSFT', 'SHOP', 'FB'
            # ]
            symbols=[
                'TSLA'
            ]
        )
        best_brain = Brain(data_repository=self._data_repository,
                           brain_configuration=best_brain_configuration)

        return best_brain


if __name__ == '__main__':
    training_runner: TrainingBenchmarker = TrainingBenchmarker()

    training_runner.run()
