import unittest
from datetime import datetime

from drltrader.brain.brain import Brain
from drltrader.data.scenario import Scenario
from drltrader.brain.brain_repository_file import BrainRepositoryFile


class BrainTestCase(unittest.TestCase):
    def __init__(self, name):
        super(BrainTestCase, self).__init__(name)

        start_day = 13
        end_day = start_day + 4

        self.training_scenario_multi_stock = Scenario(symbols=['TSLA', 'AAPL', 'MSFT'],
                                                      start_date=datetime(year=2022, month=3, day=start_day),
                                                      end_date=datetime(year=2022, month=3, day=end_day))
        self.testing_scenario_multi_stock = self.training_scenario_multi_stock

    def test_save_and_load(self):
        # Arrange
        brain_repository: BrainRepositoryFile = BrainRepositoryFile()
        brain: Brain = Brain()
        brain.learn(training_scenario=self.training_scenario_multi_stock)

        brain_id = "temp/test_save_and_load"

        # Act
        brain_repository.save(brain_id, brain, override=True)
        del brain
        brain = brain_repository.load(brain_id)

        # Assert
        self.assertIsNotNone(brain)
        brain.learn(training_scenario=self.training_scenario_multi_stock)


if __name__ == '__main__':
    unittest.main()
