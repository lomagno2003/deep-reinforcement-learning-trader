import unittest
from datetime import datetime
from datetime import timedelta

from drltrader.brain.brain import Brain
from drltrader.data.scenario import Scenario


class BrainTestCase(unittest.TestCase):
    def __init__(self, name):
        super(BrainTestCase, self).__init__(name)

        start_day = 18
        end_day = start_day + 1
        self.training_scenario = Scenario(symbol='TSLA',
                                          start_date=datetime(year=2022, month=3, day=start_day),
                                          end_date=datetime(year=2022, month=3, day=end_day))
        self.testing_scenario = Scenario(symbol='TSLA',
                                         start_date=datetime(year=2022, month=3, day=start_day),
                                         end_date=datetime(year=2022, month=3, day=end_day))

    def test_learn(self):
        brain: Brain = Brain()

        brain.learn(training_scenario=self.training_scenario,
                    testing_scenario=self.testing_scenario)

    def test_learn_evaluate(self):
        brain: Brain = Brain()

        brain.learn(training_scenario=self.training_scenario,
                    testing_scenario=self.testing_scenario,
                    total_timesteps=20000)

        results = brain.test(testing_scenario=self.testing_scenario)

        self.assertIsNotNone(results)

    def test_learn_learn(self):
        brain: Brain = Brain()

        brain.learn(training_scenario=self.training_scenario,
                    testing_scenario=self.testing_scenario)
        brain.learn(training_scenario=self.training_scenario,
                    testing_scenario=self.testing_scenario)


if __name__ == '__main__':
    unittest.main()
