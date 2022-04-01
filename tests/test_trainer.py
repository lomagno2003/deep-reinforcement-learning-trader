import unittest
from datetime import datetime
from datetime import timedelta

from drltrader.brain.brain import BrainConfiguration
from drltrader.data.data_provider import DataProvider
from drltrader.data.data_provider import Scenario
from drltrader.trainer.evolutionary_trainer import EvolutionaryTrainer


class EvolutionaryTrainerTestCase(unittest.TestCase):
    def test_train(self):
        # Arrange
        testing_scenario = Scenario(symbol='TSLA',
                                    start_date=datetime.now() - timedelta(days=30),
                                    end_date=datetime.now())

        data_provider: DataProvider = DataProvider()
        # FIXME: This needs to be done to initialize the feature names
        data_provider.retrieve_data(testing_scenario)
        trainer: EvolutionaryTrainer = EvolutionaryTrainer(data_provider=data_provider)

        # Act
        best_brain_configuration: BrainConfiguration = trainer.train()

        # Assert
        self.assertIsNotNone(best_brain_configuration)
        print(best_brain_configuration)


if __name__ == '__main__':
    unittest.main()
