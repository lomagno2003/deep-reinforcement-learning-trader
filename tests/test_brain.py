import unittest
from datetime import datetime
from datetime import timedelta

from drltrader.brain.brain import Brain
from drltrader.data.scenario import Scenario
from drltrader.envs.env_observer import PrintEnvObserver

class BrainTestCase(unittest.TestCase):
    def __init__(self, name):
        super(BrainTestCase, self).__init__(name)

        start_day = 13
        end_day = start_day + 4
        self.training_scenario_single_stock = Scenario(symbol='TSLA',
                                                       start_date=datetime(year=2022, month=3, day=start_day),
                                                       end_date=datetime(year=2022, month=3, day=end_day))
        self.testing_scenario_single_stock = self.training_scenario_single_stock

        self.training_scenario_multi_stock = Scenario(symbols=['TSLA', 'AAPL', 'MSFT'],
                                                      start_date=datetime(year=2022, month=3, day=start_day),
                                                      end_date=datetime(year=2022, month=3, day=end_day))
        self.testing_scenario_multi_stock = self.training_scenario_multi_stock
        self.observing_scenario_multi_stock = Scenario(symbols=['TSLA', 'AAPL', 'MSFT'],
                                                       start_date=datetime.now() - timedelta(days=2))

    def test_learn_single_stock(self):
        # Arrange
        brain: Brain = Brain()

        # Act/Assert
        brain.learn(training_scenario=self.training_scenario_single_stock,
                    testing_scenario=self.testing_scenario_single_stock)

    def test_learn_evaluate_single_stock(self):
        # Arrange
        brain: Brain = Brain()

        # Act
        brain.learn(training_scenario=self.training_scenario_single_stock,
                    testing_scenario=self.testing_scenario_single_stock)
        results = brain.evaluate(testing_scenario=self.testing_scenario_single_stock)

        # Assert
        self.assertIsNotNone(results)

    def test_learn_learn_single_stock(self):
        # Arrange
        brain: Brain = Brain()

        # Act/Assert
        brain.learn(training_scenario=self.training_scenario_single_stock,
                    testing_scenario=self.testing_scenario_single_stock)
        brain.learn(training_scenario=self.training_scenario_single_stock,
                    testing_scenario=self.testing_scenario_single_stock)

    def test_learn_evaluate_multi_stock(self):
        # Arrange
        brain: Brain = Brain()

        # Act
        brain.learn(training_scenario=self.training_scenario_multi_stock)
        results = brain.evaluate(testing_scenario=self.testing_scenario_multi_stock)

        # Assert
        self.assertIsNotNone(results)

    def test_learn_evaluate_live_multi_stock(self):
        # Arrange
        brain: Brain = Brain(env_observer=PrintEnvObserver())

        # Act
        brain.learn(training_scenario=self.training_scenario_multi_stock)
        brain.evaluate_live(scenario=self.observing_scenario_multi_stock)

        # Assert
        self.assertIsNotNone(None)


if __name__ == '__main__':
    unittest.main()
