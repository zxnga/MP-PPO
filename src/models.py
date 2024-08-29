from typing import NamedTuple, Union, Optional, Generator, List, Any

import numpy as np
import torch as th
import torch.nn as nn
from datasets import Dataset
import pandas as pd
import datetime
from functools import partial

from gymnasium import spaces

from ts_utils import transform_start_field, create_test_dataloader
from transformers import TimeSeriesTransformerForPrediction      


class ConcatLayer(nn.Module):
    def __init__(self, dim=2):
        super().__init__()
        self.dim = dim

    def forward(self, x, y):
        res = th.cat((x, y), dim=self.dim)
        return res
        # return res.view(-1, self.concat_output_size)


class TransformerWrapper2:
    def __init__(self, trained_model: TimeSeriesTransformerForPrediction, freq: str, static_features: List[List[Any]]):
        self.config = trained_model.config
        self.freq = freq
        self.model = trained_model #nn.module
        self.prediction_length = self.config.prediction_length
        self.device = trained_model.device
        #features known ahead that are fixed and can be passed to model during inference ex. building type/climate zone
        self.static_features = static_features #List of List. 1 list for each n_envs
        
        # If multiple n_envs during training, need to specify static features for each one of them
        assert isinstance(self.static_features, list) and all(isinstance(elem, list) for elem in self.static_features), \
            f"static_features needs to be a List of Lists to accomodate possible n_envs."

        if self.config.num_static_categorical_features > 0:
            assert all(all(element is not None for element in sublist) for sublist in self.static_features), \
                f"static_features must not be None when num_static_categorical_features > 0 ({self.config.num_static_categorical_features})"
            assert all(len(sublist) == self.config.num_static_categorical_features for sublist in self.static_features), \
                f"Length of static_features must be equal to num_static_categorical_features ({self.config.num_static_categorical_features})"
        elif self.config.num_static_categorical_features == 0:
            assert all(len(sublist) == 1 and sublist[0] is None for sublist in self.static_features), \
                f"static_features must be None when num_static_categorical_features == 0 ({self.config.num_static_categorical_features})"

        print(f'Transformer model correctly wrapped. Model set on device: {self.model.device}')

    def get_prediction(self, infer_df: pd.DataFrame, mode: str = 'eval') -> th.Tensor:
        """
            Generate predictions from the transformer model.

            Args:
                infer_df: DataFrame containing the input data for inference.
                mode: Mode for the model, either 'eval' or 'train'. Default is 'eval'.

            Returns:
                A tensor containing the predicted values.
        """
        test_data = Dataset.from_pandas(infer_df, preserve_index=True)
        test_data.set_transform(partial(transform_start_field, freq=self.freq))

        test_dataloader = create_test_dataloader(
            config=self.config,
            freq=self.freq,
            data=test_data,
            batch_size=128 #1
        )

        self.model.to(self.device)
        self.model.eval() if mode == 'eval' else self.model.train()

        forecasts = []
        for batch in test_dataloader:
            outputs = self.model.generate(
                static_categorical_features=batch["static_categorical_features"].to(self.device)
                if self.config.num_static_categorical_features > 0
                else None,
                static_real_features=batch["static_real_features"].to(self.device)
                if self.config.num_static_real_features > 0
                else None,
                past_time_features=batch["past_time_features"].to(self.device),
                past_values=batch["past_values"].to(self.device),
                future_time_features=batch["future_time_features"].to(self.device),
                past_observed_mask=batch["past_observed_mask"].to(self.device),
            )
            forecasts.append(outputs.sequences)
        forecasts = th.vstack(tuple(forecasts))
        pred = th.median(forecasts, dim=1).values
        
        return pred # dimension for training purposes shape of obs is (n_envs, n_obs)

    @staticmethod
    def create_infer_dataset(static_features: List[List[Any]], past_values: Union[List[np.ndarray], np.ndarray]) -> pd.DataFrame:
        """
            Create a DataFrame for inference.

            Args:
                static_features: List of static features.
                past_values: Past values of independant state. is a list of arrays (one for each n_envs) that will
                                correspond to a line in the infer_df

            Returns:
                A DataFrame formatted for inference.
        """
        # Note: At this stage the arrays of past_values are of dtype object and not float
        infer_df = pd.DataFrame(columns=['target', 'start', 'feat_static_cat', 'feat_dynamic_real', 'item_id'])
        # TODO: need to adapt if n_envs > 1
        if past_values.shape[0] > 1:
            past_values = np.squeeze(past_values)

        for static_feature_i, past_values_i in zip(static_features, past_values):
            if len(static_feature_i) == 1 and static_feature_i[0] is None:
                static_feature_i = None
            #TODO: when sampling buffer in train, indepndant states are list of 256 of list of 1 array each
            # check modification ? maybe dimension is due to possible multiple n_envs
            # if past_values_i.shape == (1,):
            #     print('create_infer_dataset condi in')
            #     past_values_i = past_values_i[0]
            idx = len(infer_df)
            infer_df.loc[idx] = [
                past_values_i,
                datetime.datetime(2017, 1, 1, 0, 0),
                static_feature_i,
                None,
                'T'+str(idx+1)]
        return infer_df