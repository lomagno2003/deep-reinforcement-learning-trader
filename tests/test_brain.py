import unittest
from datetime import datetime
from datetime import timedelta

from drltrader.brain.brain import Brain, BrainConfiguration
from drltrader.data import Scenario
from drltrader.data.scenarios import Scenarios
from drltrader.observers import Order
from drltrader.observers.simple_observer import CallbackObserver


class BrainTestCase(unittest.TestCase):
    def __init__(self, name):
        super(BrainTestCase, self).__init__(name)

        self.training_scenario_multi_stock = Scenarios.last_market_weeks(start_week=3,
                                                                         end_week=1)
        self.testing_scenario_multi_stock = Scenarios.last_market_week()
        self.observing_scenario_multi_stock = Scenarios.last_market_week()

    def test_learn_evaluate_multi_stock(self):
        # Arrange
        brain: Brain = Brain(brain_configuration=BrainConfiguration(symbols=['TSLA', 'AAPL', 'MSFT'],
                                                                    interval='5m'))

        # Act
        brain.learn(training_scenario=self.training_scenario_multi_stock)
        results = brain.evaluate(testing_scenario=self.testing_scenario_multi_stock)

        # Assert
        self.assertIsNotNone(results)

    def test_learn_observe_multi_stock(self):
        # Arrange
        brain: Brain = Brain()
        self._processed = False

        def stop_observing(order: Order):
            brain.stop_observing()
            self._processed = True

        # Act
        brain.learn(training_scenario=self.training_scenario_multi_stock)
        brain.start_observing(scenario=self.observing_scenario_multi_stock,
                              observer=CallbackObserver(new_data_callback_function=stop_observing))

        # Assert
        self.assertTrue(self._processed)


if __name__ == '__main__':
    unittest.main()
