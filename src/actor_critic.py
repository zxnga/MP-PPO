from typing import Dict, List, Optional, Tuple, Type

import numpy as np
from gymnasium import spaces
import torch
import torch.nn as nn

from stable_baselines3.common.type_aliases import PyTorchObs
from stable_baselines3.common.preprocessing import preprocess_obs

from stable_baselines3.common.policies import ContinuousCritic
from stable_baselines3.sac.policies import Actor, LOG_STD_MAX, LOG_STD_MIN
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class IndependantStateActor(Actor):
    action_space: spaces.Box

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        *args,
        **kwargs,
    ):
        super().__init__(
            observation_space,
            action_space,
            *args,
            **kwargs,
        )

    def get_action_dist_params(self, obs: PyTorchObs, past_independant_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Get the parameters for the action distribution.

        :param obs:
        :return:
            Mean, standard deviation and optional keyword arguments.
        """
        features = self.extract_features(obs, past_independant_state, self.features_extractor)
        latent_pi = self.latent_pi(features)
        mean_actions = self.mu(latent_pi)

        if self.use_sde:
            return mean_actions, self.log_std, dict(latent_sde=latent_pi)
        # Unstructured exploration (Original implementation)
        log_std = self.log_std(latent_pi)  # type: ignore[operator]
        # Original Implementation to cap the standard deviation
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean_actions, log_std, {}

    def forward(self, obs: PyTorchObs, past_independant_state: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        mean_actions, log_std, kwargs = self.get_action_dist_params(obs, past_independant_state)
        # Note: the action is squashed
        return self.action_dist.actions_from_params(mean_actions, log_std, deterministic=deterministic, **kwargs)

    def action_log_prob(self, obs: PyTorchObs, past_independant_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean_actions, log_std, kwargs = self.get_action_dist_params(obs, past_independant_state)
        # return action and associated log prob
        return self.action_dist.log_prob_from_params(mean_actions, log_std, **kwargs)

    def _predict(self, observation: PyTorchObs, past_independant_state: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        return self(observation, past_independant_state, deterministic)

    def extract_features(self, obs: PyTorchObs, past_independant_state: torch.Tensor, features_extractor: BaseFeaturesExtractor) -> torch.Tensor:
        """
            Preprocess the observation if needed and extract features.

            :param obs: Observation
            :param features_extractor: The features extractor to use.
            :return: The extracted features
        """
        preprocessed_obs = preprocess_obs(obs, self.observation_space, normalize_images=self.normalize_images)
        return features_extractor(preprocessed_obs, past_independant_state)

class IndependantStateCritic(ContinuousCritic):

    features_extractor: BaseFeaturesExtractor

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        net_arch: List[int],
        features_extractor: BaseFeaturesExtractor,
        features_dim: int,
        activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
        n_critics: int = 2,
        share_features_extractor: bool = True,
    ):
        super().__init__(
            observation_space,
            action_space,
            net_arch,
            features_extractor,
            features_dim,
            activation_fn,
            normalize_images,
            n_critics,
            share_features_extractor,
        )

    def extract_features(self, obs: PyTorchObs, past_independant_state: torch.Tensor, features_extractor: BaseFeaturesExtractor) -> torch.Tensor:
        """
            Preprocess the observation if needed and extract features.

            :param obs: Observation
            :param features_extractor: The features extractor to use.
            :return: The extracted features
        """
        preprocessed_obs = preprocess_obs(obs, self.observation_space, normalize_images=self.normalize_images)
        return features_extractor(preprocessed_obs, past_independant_state)

    def forward(self, obs: torch.Tensor, actions: torch.Tensor, past_independant_state) -> Tuple[torch.Tensor, ...]:
        # Learn the features extractor using the policy loss only
        # when the features_extractor is shared with the actor
        with torch.set_grad_enabled(not self.share_features_extractor):
            features = self.extract_features(obs, past_independant_state, self.features_extractor)
        qvalue_input = torch.cat([features, actions], dim=1)
        return tuple(q_net(qvalue_input) for q_net in self.q_networks)

    def q1_forward(self, obs: torch.Tensor, past_independant_state: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Only predict the Q-value using the first network.
        This allows to reduce computation when all the estimates are not needed
        (e.g. when updating the policy in TD3).
        """
        with torch.no_grad():
            features = self.extract_features(obs, past_independant_state, self.features_extractor)
        return self.q_networks[0](torch.cat([features, actions], dim=1))