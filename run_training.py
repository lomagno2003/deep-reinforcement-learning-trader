import logging
import logging.config
from datetime import datetime
from datetime import timedelta

from drltrader.brain.brain import Brain, BrainConfiguration
from drltrader.brain.brain_repository_file import BrainRepositoryFile
from drltrader.data import Scenario
from drltrader.trainer.evolutionary_trainer import EvolutionaryTrainer, TrainingConfiguration

logging.config.fileConfig('log.ini', disable_existing_loggers=False)
logger = logging.getLogger(__name__)


class TrainingRunner:
    def __init__(self):
        self._symbols = ['SPY', 'TDOC', 'ETSY', 'MELI', 'SE', 'SQ', 'DIS', 'TSLA', 'AAPL', 'MSFT', 'SHOP']
        self._initiate_scenarios()
        self._initiate_training_configuration()

        self._brain_repository = BrainRepositoryFile()

    def run(self):
        # Find best brain configuration
        logger.info("Finding best brain configuration")
        trainer: EvolutionaryTrainer = EvolutionaryTrainer()
        best_brain_configuration: BrainConfiguration = trainer.train(self._training_configuration)

        # Train brain
        logger.info("Training best brain")
        best_brain = Brain(data_repository=self._data_repository,
                           brain_configuration=best_brain_configuration)
        best_brain.learn(self._training_scenarios[0], total_timesteps=200000)

        # Save brain
        logger.info("Saving best brain")
        self._brain_repository.save("best_brain", best_brain, override=True)

    def _initiate_scenarios(self):
        self._training_scenarios = [Scenario(symbols=self._symbols,
                                             start_date=datetime.now() - timedelta(days=60),
                                             end_date=datetime.now() - timedelta(days=10))]
        self._testing_scenarios = [Scenario(symbols=self._symbols,
                                            start_date=datetime.now() - timedelta(days=10),
                                            end_date=datetime.now())]

    def _initiate_training_configuration(self):
        self._training_configuration = TrainingConfiguration(training_scenarios=self._training_scenarios,
                                                             testing_scenarios=self._testing_scenarios,
                                                             generations=10,
                                                             start_population=50,
                                                             stop_population=10,
                                                             step_population=-5,
                                                             start_timesteps=1000,
                                                             stop_timesteps=50000,
                                                             step_timesteps=1000,
                                                             solutions_statistics_filename='logs/solutions.csv')


if __name__ == '__main__':
    training_runner: TrainingRunner = TrainingRunner()

    training_runner.run()
