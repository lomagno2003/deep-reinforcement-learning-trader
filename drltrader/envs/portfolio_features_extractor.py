import gym
import torch as th
import torch.nn as nn

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class PortfolioFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self,
                 observation_space: gym.spaces.Box,
                 f_cnn1_kernel_count: int = 32,
                 f_cnn1_kernel_size: int = 8,
                 f_cnn2_kernel_count: int = 64,
                 f_cnn2_kernel_size: int = 4,
                 f_linear1_size: int = 64,
                 f_linear2_size: int = 64):
        super(PortfolioFeaturesExtractor, self).__init__(observation_space, f_linear2_size)

        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv1d(n_input_channels, f_cnn1_kernel_count, kernel_size=f_cnn1_kernel_size, stride=1, padding=f_cnn1_kernel_size),
            nn.ReLU(),
            nn.Conv1d(f_cnn1_kernel_count, f_cnn2_kernel_count, kernel_size=f_cnn2_kernel_size, stride=2, padding=f_cnn2_kernel_size),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, f_linear1_size),
            nn.ReLU(),
            nn.Linear(f_linear1_size, f_linear2_size),
            nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))
