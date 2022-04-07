import unittest
import shutil
from pathlib import Path
from datetime import datetime

from drltrader.brain.brain import Brain
from drltrader.data.scenario import Scenario
from drltrader.brain.brain_repository_gcs import BrainRepositoryGoogleCloudStorage


class BrainRepositoryGoogleCloudStorageTestCase(unittest.TestCase):
    def __init__(self, name):
        super(BrainRepositoryGoogleCloudStorageTestCase, self).__init__(name)

        start_day = 13
        end_day = start_day + 4

        self.training_scenario_multi_stock = Scenario(symbols=['TSLA', 'AAPL', 'MSFT'],
                                                      start_date=datetime(year=2022, month=3, day=start_day),
                                                      end_date=datetime(year=2022, month=3, day=end_day))
        self.testing_scenario_multi_stock = self.training_scenario_multi_stock

    def test_save_and_load(self):
        # Arrange
        brain_id = "gcs_test"
        path = f"temp/{brain_id}"
        directory_exists = (Path.cwd() / path).exists()
        if directory_exists:
            shutil.rmtree(path)

        brain_repository: BrainRepositoryGoogleCloudStorage = BrainRepositoryGoogleCloudStorage(
            gcloud_creds_file='gcloud-creds.json')
        brain: Brain = Brain()
        brain.learn(training_scenario=self.training_scenario_multi_stock)

        # Act
        brain_repository.save(brain_id, brain, override=True)
        del brain
        shutil.rmtree(path)
        brain = brain_repository.load(brain_id)

        # Assert
        self.assertIsNotNone(brain)
        brain.learn(training_scenario=self.training_scenario_multi_stock)


if __name__ == '__main__':
    unittest.main()
