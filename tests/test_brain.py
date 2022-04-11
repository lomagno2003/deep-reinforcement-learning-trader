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
        self.training_scenario_single_stock = Scenario(symbol='TSLA',
                                                       start_date=datetime(year=2022, month=3, day=start_day),
                                                       end_date=datetime(year=2022, month=3, day=end_day))
        self.testing_scenario_single_stock = self.training_scenario_single_stock

        self.training_scenario_multi_stock = Scenario(symbols=['TSLA', 'AAPL', 'MSFT'],
                                                      start_date=datetime(year=2022, month=3, day=start_day),
                                                      end_date=datetime(year=2022, month=3, day=end_day))
        self.testing_scenario_multi_stock = self.training_scenario_multi_stock
        self.observing_scenario_multi_stock = Scenario(symbols=['TSLA', 'AAPL', 'MSFT'],
                                                       start_date=datetime.now() - timedelta(days=10),
                                                       end_date=datetime.now() - timedelta(days=5))

    def test_learn_single_stock(self):
        # Arrange
        brain: Brain = Brain()

        # Act/Assert
        brain.learn(training_scenario=self.training_scenario_single_stock)

    def test_learn_evaluate_single_stock(self):
        # Arrange
        brain: Brain = Brain()

        # Act
        brain.learn(training_scenario=self.training_scenario_single_stock)
        results = brain.evaluate(testing_scenario=self.testing_scenario_single_stock)

        # Assert
        self.assertIsNotNone(results)

    def test_learn_learn_single_stock(self):
        # Arrange
        brain: Brain = Brain()

        # Act/Assert
        brain.learn(training_scenario=self.training_scenario_single_stock)
        brain.learn(training_scenario=self.training_scenario_single_stock)

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
                              observer=CallbackObserver(callback_function=stop_observing))

        # Assert
        self.assertTrue(self._processed)

    def test_save_and_load(self):
        # Arrange
        brain: Brain = Brain()
        brain.learn(training_scenario=self.training_scenario_multi_stock)

        brain_path = "temp/test_save_and_load"

        # Act
        brain.save(brain_path, override=True)
        del brain
        brain = Brain.load(brain_path)

        # Assert
        self.assertIsNotNone(brain)
        brain.learn(training_scenario=self.training_scenario_multi_stock)


if __name__ == '__main__':
    unittest.main()
