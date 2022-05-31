import logging
import logging.config
from datetime import datetime, time
from datetime import timedelta
from dateutil.relativedelta import relativedelta, SA, MO
from pytz import timezone
import numpy as np

from drltrader.brain.brain import Brain, BrainConfiguration
from drltrader.brain.brain_repository_file import BrainRepositoryFile
from drltrader.data import Scenario
from drltrader.data.ohlcv_data_repository import AlpacaOHLCVDataRepository
from drltrader.data.cached_data_repository import CachedDataRepository
from drltrader.data.indicators_data_repository import IndicatorsDataRepository
from drltrader.observers.simple_observer import PrintEnvObserver

logging.config.fileConfig('log.ini', disable_existing_loggers=False)
logger = logging.getLogger(__name__)


class TrainingBenchmarker:
    def __init__(self):
        self._brain_repository = BrainRepositoryFile()
        self._data_repository = CachedDataRepository(IndicatorsDataRepository(AlpacaOHLCVDataRepository()))

    def run(self):
        root_datetime = datetime.now()
        testing_weeks = 14

        # Find best brain
        best_brain = self.learn_initial_brain_on_datetime(root_datetime - timedelta(weeks=testing_weeks))

        # Perform benchmarking
        testing_profits_str = []
        testing_profits = []

        for week in range(testing_weeks):
            weeks_offset = testing_weeks - week - 1
            ref_datetime = root_datetime - timedelta(weeks=weeks_offset)
            testing_start_date, testing_end_date, testing_profit = self.train_on_datetime(best_brain=best_brain,
                                                                                          ref_datetime=ref_datetime)
            testing_profits_str.append(f"{testing_start_date}->{testing_end_date}: {testing_profit}")
            testing_profits.append(testing_profit)

            logger.info(f"Current list of profits: {testing_profits_str}")
            logger.info(f"The average profit so far is {np.average(testing_profits)}")

    def learn_initial_brain_on_datetime(self, ref_datetime: datetime):
        self._initiate_initial_scenarios_on_date(ref_datetime=ref_datetime)
        best_brain_configuration: BrainConfiguration = BrainConfiguration(
            f_cnn1_kernel_count=64,
            f_cnn1_kernel_size=8,
            f_cnn2_kernel_count=64,
            f_cnn2_kernel_size=4,
            f_linear1_size=512,
            f_linear2_size=256,
            window_size=16,
            interval='1h',
            # signal_feature_names=[
            #     "SMA_4", "SMA_6", "SMA_16", "RSI_16", "MOM_4", "MOM_18", "ROC_12"
            # ],
            signal_feature_names=self._data_repository.get_columns_per_symbol(),
            symbols=[
                'TDOC', 'ETSY', 'MELI', 'SE', 'SQ', 'DIS', 'TSLA', 'AAPL', 'MSFT', 'SHOP', 'FB'
            ]
        )
        best_brain = Brain(data_repository=self._data_repository,
                           brain_configuration=best_brain_configuration)
        best_brain.learn(self._initial_training_scenarios[0], total_timesteps=10000)

        return best_brain

    def train_on_datetime(self, best_brain: Brain, ref_datetime: datetime):
        # Find best brain configuration
        logger.info("Finding best brain configuration")
        self._initiate_scenarios_on_date(ref_datetime=ref_datetime)

        # Train brain
        logger.info("Training best brain")
        best_brain.learn(self._training_scenarios[0], total_timesteps=10000)

        # Save brain
        logger.info("Saving best brain")
        self._brain_repository.save("best_brain", best_brain, override=True)

        # Test brain
        for testing_scenario in self._testing_scenarios:
            logger.info(f"Testing brain on scenario {testing_scenario}")

            info = best_brain.evaluate(testing_scenario=testing_scenario,
                                       observer=PrintEnvObserver())
            profit = info['total_profit'] if 'total_profit' in info else info['current_profit']

            logger.info(f"The profit on the testing scenario {testing_scenario} is {profit}")

            return testing_scenario.start_date, testing_scenario.end_date, profit

    def _initiate_initial_scenarios_on_date(self, ref_datetime: datetime = datetime.now()):
        est_timezone = timezone('EST')
        market_open_hour = time(hour=9, minute=30)
        market_close_hour = time(hour=16, minute=00)

        training_end_date = datetime.combine(ref_datetime - timedelta(weeks=2) + relativedelta(weekday=SA(-1)) - timedelta(days=1),
                                             market_close_hour,
                                             est_timezone)
        training_start_date = datetime.combine(training_end_date - timedelta(weeks=3) + relativedelta(weekday=MO(-1)),
                                               market_open_hour,
                                               est_timezone)

        validation_end_date = datetime.combine(ref_datetime - timedelta(weeks=1) + relativedelta(weekday=SA(-1)) - timedelta(days=1),
                                               market_close_hour,
                                               est_timezone)
        validation_start_date = datetime.combine(validation_end_date + relativedelta(weekday=MO(-1)),
                                                 market_open_hour,
                                                 est_timezone)

        self._initial_training_scenarios = [Scenario(start_date=training_start_date, end_date=training_end_date)]
        self._initial_validation_scenarios = [Scenario(start_date=validation_start_date, end_date=validation_end_date)]

    def _initiate_scenarios_on_date(self, ref_datetime: datetime = datetime.now()):
        est_timezone = timezone('EST')
        market_open_hour = time(hour=9, minute=30)
        market_close_hour = time(hour=16, minute=00)

        training_end_date = datetime.combine(ref_datetime - timedelta(weeks=2) + relativedelta(weekday=SA(-1)) - timedelta(days=1),
                                             market_close_hour,
                                             est_timezone)
        training_start_date = datetime.combine(training_end_date - timedelta(weeks=3) + relativedelta(weekday=MO(-1)),
                                               market_open_hour,
                                               est_timezone)

        validation_end_date = datetime.combine(ref_datetime - timedelta(weeks=1) + relativedelta(weekday=SA(-1)) - timedelta(days=1),
                                               market_close_hour,
                                               est_timezone)
        validation_start_date = datetime.combine(validation_end_date + relativedelta(weekday=MO(-1)),
                                                 market_open_hour,
                                                 est_timezone)

        testing_end_date = datetime.combine(ref_datetime + relativedelta(weekday=SA(-1)) - timedelta(days=1),
                                            market_close_hour,
                                            est_timezone)
        testing_start_date = datetime.combine(testing_end_date + relativedelta(weekday=MO(-1)),
                                              market_open_hour,
                                              est_timezone)

        self._training_scenarios = [Scenario(start_date=training_start_date, end_date=training_end_date)]
        self._validation_scenarios = [Scenario(start_date=validation_start_date, end_date=validation_end_date)]
        self._testing_scenarios = [Scenario(start_date=testing_start_date, end_date=testing_end_date)]


if __name__ == '__main__':
    training_runner: TrainingBenchmarker = TrainingBenchmarker()

    training_runner.run()
