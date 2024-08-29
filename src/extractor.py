from typing import Any, Dict, List, Tuple, Type, Union

import numpy as np
from gymnasium import spaces
import torch

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from models import TransformerWrapper2, ConcatLayer



class TSExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: spaces.Space,
        trained_ts: TransformerWrapper2,
        episode_length: int, 
        n_envs: int,
        pos_independant_state: int
    ) -> None:
        super().__init__(observation_space, features_dim=observation_space.shape[0]+trained_ts.prediction_length)
        
        self.transformer = trained_ts
        self.concat_layer = ConcatLayer(dim=1) # for training purposes cause obs is of shape (n_envs, n_obs)
        # Freeze transformer weights
        self.freeze_ts_weights()
        assert n_envs == 1, "To use Transformer n_envs has to equal to 1"

    def freeze_ts_weights(self):
        for param in self.transformer.model.parameters():
            param.requires_grad = False
    
    def get_independant_state(self, observations: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
            return observations[0][self.pos_independant_state]

    def forward(self, observations: torch.Tensor, past_independant_states: Union[List[float],np.ndarray], mode='eval') -> torch.Tensor:       
        # Note: At this stage the current independant_state is already inside the past_independant_state vector
        # Note: When traning past_independant_states is np.ndarray of dtype object
        infer_df = self.transformer.create_infer_dataset(
            self.get_static_features(past_independant_states),
            past_independant_states)
        predictions = self.transformer.get_prediction(infer_df, mode)
        return self.concat_layer(observations, predictions)

    def get_static_features(self, past_independant_states: Union[List[float],np.ndarray]) -> List[List[Any]]:
        # Note: this assumes the static features are the same for all n_envs

        # During training after sample past_independant_states is np.ndarray of dtype object
        if isinstance(past_independant_states, np.ndarray):
            nb_samples = past_independant_states.shape[0] #batch_size
            # Assumes same static feature for every sample disregarding n_envs of origin
            return [self.transformer.static_features[0]] * nb_samples
        # during collection of samples, of size n_envs so normally okay
        elif isinstance(past_independant_states, list):
            # TODO: we only want to check this once
            assert len(self.transformer.static_features) == len(past_independant_states), \
                "length of static features needs to be the same length as n_envs"
            return self.transformer.static_features
        else:
            # TODO
            raise ValueError