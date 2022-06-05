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
        testing_profits_str = []
        testing_profits = []

        while True:
            logger.info("Training brain")
            best_brain.learn(self._scenario, total_timesteps=2000)
            # testing_start_date, testing_end_date, testing_profit = self.train_on_datetime(best_brain=best_brain)
            # testing_profits_str.append(f"{testing_start_date}->{testing_end_date}: {testing_profit}")
            # testing_profits.append(testing_profit)
            #
            # logger.info(f"Current list of profits: {testing_profits_str}")
            # logger.info(f"The average profit so far is {np.average(testing_profits)}")

    def build_initial_brain_on_datetime(self):
        logger.info(f"Build initial brain")
        best_brain_configuration: BrainConfiguration = BrainConfiguration(
            f_cnn1_kernel_count=64,
            f_cnn1_kernel_size=32,
            f_pool1_size=4,
            f_pool1_stride=2,
            f_cnn2_kernel_count=32,
            f_cnn2_kernel_size=32,
            f_pool2_size=4,
            f_pool2_stride=2,
            f_linear1_size=512,
            f_linear2_size=256,
            f_pi_net_arch=[64, 64],
            f_vf_net_arch=[64, 64],
            window_size=128,
            interval='5m',
            prices_feature_name='5m_Close',
            # signal_feature_names=self._data_repository.get_columns_per_symbol(),
            signal_feature_names=[
                '60m_RSI_4', '60m_VW_MACD_16_32_12', '60m_MOM_4'
            ],
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
