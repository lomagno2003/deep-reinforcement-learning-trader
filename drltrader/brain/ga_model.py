import numpy as np
import torch as th
import torch.nn as nn
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type, Union

from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.policies import BasePolicy

import pygad
import pygad.torchga as torchga


class GAModel(BaseAlgorithm):
    INSTANCE = None

    def __init__(self, *args, **kwargs):
        super(GAModel, self).__init__(*args,
                                      policy=GAPolicy,
                                      policy_base=GAPolicy,
                                      learning_rate=None,
                                      **kwargs)
        self._setup_model()

        GAModel.INSTANCE = self

    def _setup_model(self) -> None:
        self.policy = self.policy_class(
            self.observation_space,
            self.action_space,
            **self.policy_kwargs
        )

    def learn(
            self,
            total_timesteps: int = 10,
            callback: MaybeCallback = None,
            log_interval: int = 100,
            tb_log_name: str = "run",
            eval_env: Optional[GymEnv] = None,
            eval_freq: int = -1,
            n_eval_episodes: int = 5,
            eval_log_path: Optional[str] = None,
            reset_num_timesteps: bool = True,
    ) -> "BaseAlgorithm":
        torch_ga = torchga.TorchGA(model=self.policy.model, num_solutions=40)

        ga_instance = pygad.GA(num_generations=total_timesteps,
                               num_parents_mating=20,
                               initial_population=torch_ga.population_weights,
                               fitness_func=GAModel._fitness_func,
                               parent_selection_type='rank',
                               crossover_type='uniform',
                               mutation_type='random',
                               mutation_percent_genes=0.3,
                               keep_parents=5)

        ga_instance.run()

        # Returning the details of the best solution.
        solution, solution_fitness, solution_idx = ga_instance.best_solution()
        print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
        print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))

        # Fetch the parameters of the best solution.
        best_solution_weights = torchga.model_weights_as_dict(model=self.policy.model,
                                                              weights_vector=solution)
        self.policy.model.load_state_dict(best_solution_weights)

        return self

    def predict(self, *args, **kwargs):
        action, state = super().predict(*args, **kwargs)

        return action.argmax(), state

    @staticmethod
    def _fitness_func(solution, sol_idx):
        ga_model = GAModel.INSTANCE
        cumulative_rewards = 0.0

        model_weights_dict = torchga.model_weights_as_dict(model=ga_model.policy.model,
                                                           weights_vector=solution)
        ga_model.policy.model.load_state_dict(model_weights_dict)

        obs = ga_model.env.reset()

        while True:
            obs = obs[np.newaxis, ...]
            action, _ = ga_model.predict(obs[0])
            obs, rewards, done, info = ga_model.env.step([action])

            cumulative_rewards += rewards[0]

            if done[0]:
                break

        return cumulative_rewards


class GAPolicy(BasePolicy):
    def __init__(self, *args, **kwargs):
        super(GAPolicy, self).__init__(*args, **kwargs)

        self.features_extractor = self.features_extractor_class(self.observation_space,
                                                                **self.features_extractor_kwargs)

        self.model = nn.Sequential(self.features_extractor,
                                   nn.Linear(self.features_extractor.features_dim, self.action_space.n))

    def forward(self, *args, **kwargs):
        pass

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        observation = observation.float()
        prediction_vector = self.model(observation)
        return prediction_vector
