import unittest
import warnings
from datetime import datetime
from datetime import timedelta

from drltrader.brain.brain import BrainConfiguration
from drltrader.data import Scenario
from drltrader.trainer.evolutionary_trainer import EvolutionaryTrainer, TrainingConfiguration

warnings.filterwarnings("ignore", category=DeprecationWarning)


class EvolutionaryTrainerTestCase(unittest.TestCase):
    def __init__(self, methodName):
        super().__init__(methodName=methodName)

    def test_short_train_single_stock(self):
        # Arrange
        trainer: EvolutionaryTrainer = EvolutionaryTrainer()

        training_scenarios = [Scenario(symbol='TSLA',
                                       start_date=datetime.now() - timedelta(days=30),
                                       end_date=datetime.now())]
        testing_scenarios = [Scenario(symbol='TSLA',
                                      start_date=datetime.now() - timedelta(days=2),
                                      end_date=datetime.now())]
        training_configuration = TrainingConfiguration(training_scenarios=training_scenarios,
                                                       testing_scenarios=testing_scenarios)

        # Act
        best_brain_configuration: BrainConfiguration = trainer.train(training_configuration)

        # Assert
        self.assertIsNotNone(best_brain_configuration)
        print(best_brain_configuration)

    def test_big_train_single_stock(self):
        # Arrange
        trainer: EvolutionaryTrainer = EvolutionaryTrainer()

        training_scenarios = [Scenario(symbol='TSLA',
                                       start_date=datetime.now() - timedelta(days=30),
                                       end_date=datetime.now()),
                              Scenario(symbol='SHOP',
                                       start_date=datetime.now() - timedelta(days=30),
                                       end_date=datetime.now())
                              ]
        testing_scenarios = [Scenario(symbol='TSLA',
                                      start_date=datetime.now() - timedelta(days=2),
                                      end_date=datetime.now()),
                             Scenario(symbol='SHOP',
                                      start_date=datetime.now() - timedelta(days=2),
                                      end_date=datetime.now())
                             ]
        training_configuration = TrainingConfiguration(training_scenarios=training_scenarios,
                                                       testing_scenarios=testing_scenarios,
                                                       generations=100,
                                                       start_population=20,
                                                       stop_population=5,
                                                       step_population=-1,
                                                       start_timesteps=1000,
                                                       stop_timesteps=20000,
                                                       step_timesteps=500,
                                                       solutions_statistics_filename='logs/solutions.csv')

        # Act
        best_brain_configuration: BrainConfiguration = trainer.train(training_configuration)

        # Assert
        self.assertIsNotNone(best_brain_configuration)
        print(best_brain_configuration)

    def test_short_train_multi_stock(self):
        # Arrange
        trainer: EvolutionaryTrainer = EvolutionaryTrainer()

        symbols = ['TSLA', 'AAPL', 'MSFT', 'SPY', 'SHOP']
        training_scenarios = [Scenario(symbols=symbols,
                                       interval='1d',
                                       start_date=datetime.now() - timedelta(days=360),
                                       end_date=datetime.now())]
        testing_scenarios = training_scenarios
        training_configuration = TrainingConfiguration(training_scenarios=training_scenarios,
                                                       testing_scenarios=testing_scenarios)

        # Act
        best_brain_configuration: BrainConfiguration = trainer.train(training_configuration)

        # Assert
        self.assertIsNotNone(best_brain_configuration)
        print(best_brain_configuration)

    def test_big_train_multi_stock(self):
        # Arrange
        trainer: EvolutionaryTrainer = EvolutionaryTrainer()

        symbols = ['TSLA', 'AAPL', 'MSFT', 'SPY', 'SHOP']
        training_scenarios = [Scenario(symbols=symbols,
                                       interval='1d',
                                       start_date=datetime.now() - timedelta(days=720),
                                       end_date=datetime.now() - timedelta(days=90))]

        testing_scenarios = [Scenario(symbols=symbols,
                                      interval='1d',
                                      start_date=datetime.now() - timedelta(days=90),
                                      end_date=datetime.now())]
        training_configuration = TrainingConfiguration(training_scenarios=training_scenarios,
                                                       testing_scenarios=testing_scenarios,
                                                       generations=100,
                                                       start_population=20,
                                                       stop_population=5,
                                                       step_population=-1,
                                                       start_timesteps=100,
                                                       stop_timesteps=20000,
                                                       step_timesteps=500,
                                                       solutions_statistics_filename='logs/solutions.csv')

        # Act
        best_brain_configuration: BrainConfiguration = trainer.train(training_configuration)

        # Assert
        self.assertIsNotNone(best_brain_configuration)
        print(best_brain_configuration)


if __name__ == '__main__':
    unittest.main()
