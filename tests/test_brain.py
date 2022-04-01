import unittest
from datetime import datetime
from datetime import timedelta

from drltrader.brain.brain import Brain
from drltrader.data.scenario import Scenario


class BrainTestCase(unittest.TestCase):
    def __init__(self, name):
        super(BrainTestCase, self).__init__(name)
        self.training_scenario = Scenario(symbol='TSLA',
                                          start_date=datetime.now() - timedelta(days=30),
                                          end_date=datetime.now())
        self.testing_scenario = Scenario(symbol='TSLA',
                                         start_date=datetime.now() - timedelta(days=1),
                                         end_date=datetime.now())

    def test_learn(self):
        brain: Brain = Brain()

        brain.learn(training_scenario=self.training_scenario,
                    testing_scenario=self.testing_scenario)

    def test_learn_evaluate(self):
        brain: Brain = Brain()

        brain.learn(training_scenario=self.training_scenario,
                    testing_scenario=self.testing_scenario,
                    total_timesteps=25000)

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
