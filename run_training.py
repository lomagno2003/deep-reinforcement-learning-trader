import logging
import logging.config
from datetime import datetime, time
from datetime import timedelta
from dateutil.relativedelta import relativedelta, SA, MO
from pytz import timezone

from drltrader.brain.brain import Brain, BrainConfiguration
from drltrader.brain.brain_repository_file import BrainRepositoryFile
from drltrader.data import Scenario
from drltrader.data.ohlcv_data_repository import AlpacaOHLCVDataRepository
from drltrader.data.indicators_data_repository import IndicatorsDataRepository
from drltrader.trainer.evolutionary_trainer import EvolutionaryTrainer, TrainingConfiguration

logging.config.fileConfig('log.ini', disable_existing_loggers=False)
logger = logging.getLogger(__name__)


class TrainingRunner:
    def __init__(self):
        self._initiate_scenarios()
        self._initiate_training_configuration()

        self._brain_repository = BrainRepositoryFile()
        self._data_repository = IndicatorsDataRepository(AlpacaOHLCVDataRepository())

    def run(self):
        # Find best brain configuration
        logger.info("Finding best brain configuration")
        trainer: EvolutionaryTrainer = EvolutionaryTrainer()
        best_brain_configuration: BrainConfiguration = trainer.train(self._training_configuration)

        # Train brain
        logger.info("Training best brain")
        best_brain = Brain(data_repository=self._data_repository,
                           brain_configuration=best_brain_configuration)
        best_brain.learn(self._training_scenarios[0], total_timesteps=10000)

        # Save brain
        logger.info("Saving best brain")
        self._brain_repository.save("best_brain", best_brain, override=True)

        # Test brain
        for testing_scenario in self._testing_scenarios:
            logger.info(f"Testing brain on scenario {testing_scenario}")
            testing_scenario.symbols = testing_scenario.copy_with_symbols(best_brain_configuration.symbols)
            info = best_brain.evaluate(testing_scenario=testing_scenario)
            profit = info['total_profit'] if 'total_profit' in info else info['current_profit']
            logger.info(f"The profit on the testing scenario {testing_scenario} is {profit}")

    def _initiate_scenarios(self):
        est_timezone = timezone('EST')
        market_open_hour = time(hour=9, minute=30)
        market_close_hour = time(hour=16, minute=00)

        training_end_date = datetime.combine(datetime.now() - timedelta(weeks=2) + relativedelta(weekday=SA(-1)) - timedelta(days=1),
                                             market_close_hour,
                                             est_timezone)
        training_start_date = datetime.combine(training_end_date - timedelta(weeks=3) + relativedelta(weekday=MO(-1)),
                                               market_open_hour,
                                               est_timezone)

        validation_end_date = datetime.combine(datetime.now() - timedelta(weeks=1) + relativedelta(weekday=SA(-1)) - timedelta(days=1),
                                               market_close_hour,
                                               est_timezone)
        validation_start_date = datetime.combine(validation_end_date + relativedelta(weekday=MO(-1)),
                                                 market_open_hour,
                                                 est_timezone)

        testing_end_date = datetime.combine(datetime.now() + relativedelta(weekday=SA(-1)) - timedelta(days=1),
                                            market_close_hour,
                                            est_timezone)
        testing_start_date = datetime.combine(testing_end_date + relativedelta(weekday=MO(-1)),
                                              market_open_hour,
                                              est_timezone)

        self._training_scenarios = [Scenario(start_date=training_start_date, end_date=training_end_date)]
        self._validation_scenarios = [Scenario(start_date=validation_start_date, end_date=validation_end_date)]
        self._testing_scenarios = [Scenario(start_date=testing_start_date, end_date=testing_end_date)]

    def _initiate_training_configuration(self):
        self._training_configuration = TrainingConfiguration(training_scenarios=self._training_scenarios,
                                                             validation_scenarios=self._validation_scenarios,
                                                             generations=0,
                                                             start_population=10,
                                                             stop_population=10,
                                                             step_population=-2,
                                                             start_timesteps=5000,
                                                             stop_timesteps=10000,
                                                             step_timesteps=100,
                                                             solutions_statistics_filename='logs/solutions.csv')


if __name__ == '__main__':
    training_runner: TrainingRunner = TrainingRunner()

    training_runner.run()
