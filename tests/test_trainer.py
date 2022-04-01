import unittest
import warnings
from datetime import datetime
from datetime import timedelta

from drltrader.brain.brain import BrainConfiguration
from drltrader.data.data_provider import DataProvider
from drltrader.data.data_provider import Scenario
from drltrader.trainer.evolutionary_trainer import EvolutionaryTrainer, TrainingConfiguration

warnings.filterwarnings("ignore", category=DeprecationWarning)


class EvolutionaryTrainerTestCase(unittest.TestCase):
    def __init__(self, methodName):
        super().__init__(methodName=methodName)

    def test_train(self):
        # Arrange
        data_provider: DataProvider = DataProvider()
        trainer: EvolutionaryTrainer = EvolutionaryTrainer(data_provider=data_provider)

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
                                                       total_timesteps_per_scenario=5000)

        # Act
        best_brain_configuration: BrainConfiguration = trainer.train(training_configuration)

        # Assert
        self.assertIsNotNone(best_brain_configuration)
        print(best_brain_configuration)


if __name__ == '__main__':
    unittest.main()
