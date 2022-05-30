import unittest
from datetime import datetime
from datetime import timedelta

from drltrader.brain.brain import Brain
from drltrader.data import Scenario
from drltrader.observers import Order
from drltrader.observers.simple_observer import CallbackObserver


class BrainTestCase(unittest.TestCase):
    def __init__(self, name):
        super(BrainTestCase, self).__init__(name)

        start_day = 13
        end_day = start_day + 4

        self.training_scenario_multi_stock = Scenario(symbols=['TSLA', 'AAPL', 'MSFT'],
                                                      start_date=datetime(year=2022, month=3, day=start_day),
                                                      end_date=datetime(year=2022, month=3, day=end_day))
        self.testing_scenario_multi_stock = self.training_scenario_multi_stock
        self.observing_scenario_multi_stock = Scenario(symbols=['TSLA', 'AAPL', 'MSFT'],
                                                       start_date=datetime.now() - timedelta(days=10),
                                                       end_date=datetime.now() - timedelta(days=5))

    def test_learn_evaluate_multi_stock(self):
        # Arrange
        brain: Brain = Brain()

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
