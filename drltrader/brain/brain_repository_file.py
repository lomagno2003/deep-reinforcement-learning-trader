import os
from pathlib import Path
import pickle
import shutil
from stable_baselines3 import A2C


from drltrader.brain.brain import Brain, BrainConfiguration
from drltrader.brain.brain_repository import BrainRepository


class BrainRepositoryFile(BrainRepository):
    def __init__(self):
        self._brains_folder = 'temp/'

    def load(self, brain_id: str):
        path = f"{self._brains_folder}brain_id"

        model_path = f"{path}/model"
        brain_configuration_path = f"{path}/brain_configuration.pickle"

        with open(brain_configuration_path, 'rb') as brain_configuration_file:
            brain_configuration: BrainConfiguration = BrainConfiguration(**pickle.load(brain_configuration_file))
            new_brain = Brain(brain_configuration=brain_configuration)
            new_brain._model = A2C.load(model_path)

            return new_brain

    def save(self, brain_id: str, brain: Brain, override: bool = False):
        path = f"{self._brains_folder}brain_id"

        # Check and create directory
        directory_exists = (Path.cwd() / path).exists()
        if directory_exists:
            if not override:
                raise FileExistsError(f"There's already another file/folder with name {path}")
            else:
                shutil.rmtree(path)

        os.mkdir(path)

        # Save brain
        model_path = f"{path}/model"
        brain_configuration_path = f"{path}/brain_configuration.pickle"

        with open(brain_configuration_path, 'wb') as brain_configuration_file:
            brain._model.save(model_path)
            pickle.dump(brain._brain_configuration.__dict__, brain_configuration_file)