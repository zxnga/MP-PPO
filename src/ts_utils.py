from typing import (
    Any,
    ClassVar, 
    Dict, 
    Optional, 
    Tuple, 
    Type, 
    TypeVar, 
    Union,
    Callable,
    List,
    NamedTuple,
    Generator,
    Iterable)

import torch
import pandas as pd
import numpy as np
from datasets import Dataset

from transformers import (
    TimeSeriesTransformerConfig,
    TimeSeriesTransformerForPrediction,
    PretrainedConfig)
from gluonts.dataset.field_names import FieldName
from gluonts.time_feature import (
    get_lags_for_frequency, 
    time_features_from_frequency_str,
    TimeFeature,
    get_seasonality)

from gluonts.transform import (
    AddAgeFeature,
    AddObservedValuesIndicator,
    AddTimeFeatures,
    AsNumpyArray,
    Chain,
    ExpectedNumInstanceSampler,
    InstanceSplitter,
    RemoveFields,
    SelectFields,
    SetField,
    TestSplitSampler,
    Transformation,
    ValidationSplitSampler,
    VstackFeatures,
    RenameFields)

from gluonts.transform.sampler import InstanceSampler
from gluonts.itertools import Cached, Cyclic
from gluonts.dataset.loader import as_stacked_batches

from functools import partial, lru_cache

from accelerate import Accelerator
from torch.optim import AdamW



@lru_cache(10_000)
def convert_to_pandas_period(date, freq):
    return pd.Period(date, freq)

def transform_start_field(batch, freq):
    batch["start"] = [convert_to_pandas_period(date, freq) for date in batch["start"]]
    return batch


def create_transformation(freq: str, config: PretrainedConfig) -> Transformation:
    remove_field_names = []
    if config.num_static_real_features == 0:
        remove_field_names.append(FieldName.FEAT_STATIC_REAL)
    if config.num_dynamic_real_features == 0:
        remove_field_names.append(FieldName.FEAT_DYNAMIC_REAL)
    if config.num_static_categorical_features == 0:
        remove_field_names.append(FieldName.FEAT_STATIC_CAT)

    # a bit like torchvision.transforms.Compose
    return Chain(
        # step 1: remove static/dynamic fields if not specified
        [RemoveFields(field_names=remove_field_names)]
        # step 2: convert the data to NumPy (potentially not needed)
        + (
            [
                AsNumpyArray(
                    field=FieldName.FEAT_STATIC_CAT,
                    expected_ndim=1,
                    dtype=int,
                )
            ]
            if config.num_static_categorical_features > 0
            else []
        )
        + (
            [
                AsNumpyArray(
                    field=FieldName.FEAT_STATIC_REAL,
                    expected_ndim=1,
                )
            ]
            if config.num_static_real_features > 0
            else []
        )
        + [
            AsNumpyArray(
                field=FieldName.TARGET,
                # we expect an extra dim for the multivariate case:
                expected_ndim=1 if config.input_size == 1 else 2,
            ),
            # step 3: handle the NaN's by filling in the target with zero
            # and return the mask (which is in the observed values)
            # true for observed values, false for nan's
            # the decoder uses this mask (no loss is incurred for unobserved values)
            # see loss_weights inside the xxxForPrediction model
            AddObservedValuesIndicator(
                target_field=FieldName.TARGET,
                output_field=FieldName.OBSERVED_VALUES,
            ),
            # step 4: add temporal features based on freq of the dataset
            # month of year in the case when freq="M"
            # these serve as positional encodings
            AddTimeFeatures(
                start_field=FieldName.START,
                target_field=FieldName.TARGET,
                output_field=FieldName.FEAT_TIME,
                time_features=time_features_from_frequency_str(freq),
                pred_length=config.prediction_length,
            ),
            # step 5: add another temporal feature (just a single number)
            # tells the model where in its life the value of the time series is,
            # sort of a running counter
            AddAgeFeature(
                target_field=FieldName.TARGET,
                output_field=FieldName.FEAT_AGE,
                pred_length=config.prediction_length,
                log_scale=True,
            ),
            # step 6: vertically stack all the temporal features into the key FEAT_TIME
            VstackFeatures(
                output_field=FieldName.FEAT_TIME,
                input_fields=[FieldName.FEAT_TIME, FieldName.FEAT_AGE]
                + (
                    [FieldName.FEAT_DYNAMIC_REAL]
                    if config.num_dynamic_real_features > 0
                    else []
                ),
            ),
            # step 7: rename to match HuggingFace names
            RenameFields(
                mapping={
                    FieldName.FEAT_STATIC_CAT: "static_categorical_features",
                    FieldName.FEAT_STATIC_REAL: "static_real_features",
                    FieldName.FEAT_TIME: "time_features",
                    FieldName.TARGET: "values",
                    FieldName.OBSERVED_VALUES: "observed_mask",
                }
            ),
        ]
    )

def create_instance_splitter(
    config: PretrainedConfig,
    mode: str,
    train_sampler: Optional[InstanceSampler] = None,
    validation_sampler: Optional[InstanceSampler] = None,
    past_length = None,
) -> Transformation:
    assert mode in ["train", "validation", "test", "infer"]

    instance_sampler = {
        "train": train_sampler
        or ExpectedNumInstanceSampler(
            num_instances=1.0, min_future=config.prediction_length
        ),
        "validation": validation_sampler
        or ValidationSplitSampler(min_future=config.prediction_length),
        "test": TestSplitSampler(),
        "infer": TestSplitSampler(),
    }[mode]

    if past_length is None:
        past_length = config.context_length if mode=='infer' else config.context_length + max(config.lags_sequence)

    return InstanceSplitter(
        target_field="values",
        is_pad_field=FieldName.IS_PAD,
        start_field=FieldName.START,
        forecast_start_field=FieldName.FORECAST_START,
        instance_sampler=instance_sampler,
        past_length=past_length,
        future_length=config.prediction_length,
        time_series_fields=["time_features", "observed_mask"],
    )

def create_test_dataloader(
    config: PretrainedConfig,
    freq,
    data,
    batch_size: int,
    mode = 'test',
    **kwargs,
):
    # print(mode)
    PREDICTION_INPUT_NAMES = [
        "past_time_features",
        "past_values",
        "past_observed_mask",
        "future_time_features",
    ]
    if config.num_static_categorical_features > 0:
        PREDICTION_INPUT_NAMES.append("static_categorical_features")

    if config.num_static_real_features > 0:
        PREDICTION_INPUT_NAMES.append("static_real_features")

    transformation = create_transformation(freq, config)
    transformed_data = transformation.apply(data, is_train=False)

    # we create a Test Instance splitter which will sample the very last
    # context window seen during training only for the encoder.
    instance_sampler = create_instance_splitter(config, mode)

    # we apply the transformations in test mode
    testing_instances = instance_sampler.apply(transformed_data, is_train=False)

    return as_stacked_batches(
        testing_instances,
        batch_size=batch_size,
        output_type=torch.tensor,
        field_names=PREDICTION_INPUT_NAMES,
    )

def create_train_dataloader(
    config: PretrainedConfig,
    freq,
    data,
    batch_size: int,
    num_batches_per_epoch: int,
    shuffle_buffer_length: Optional[int] = None,
    cache_data: bool = True,
    **kwargs,
) -> Iterable:
    PREDICTION_INPUT_NAMES = [
        "past_time_features",
        "past_values",
        "past_observed_mask",
        "future_time_features",
    ]
    if config.num_static_categorical_features > 0:
        PREDICTION_INPUT_NAMES.append("static_categorical_features")

    if config.num_static_real_features > 0:
        PREDICTION_INPUT_NAMES.append("static_real_features")

    TRAINING_INPUT_NAMES = PREDICTION_INPUT_NAMES + [
        "future_values",
        "future_observed_mask",
    ]

    #print('config', config)

    transformation = create_transformation(freq, config)
    # print('transformation', transformation)
    transformed_data = transformation.apply(data, is_train=True)
    # print('transformed_data', transformed_data)
    
    if cache_data:
        transformed_data = Cached(transformed_data)

    # we initialize a Training instance
    instance_splitter = create_instance_splitter(config, "train")

    # the instance splitter will sample a window of
    # context length + lags + prediction length (from the 366 possible transformed time series)
    # randomly from within the target time series and return an iterator.
    stream = Cyclic(transformed_data).stream()
    training_instances = instance_splitter.apply(stream)
    
    return as_stacked_batches(
        training_instances,
        batch_size=batch_size,
        shuffle_buffer_length=shuffle_buffer_length,
        field_names=TRAINING_INPUT_NAMES,
        output_type=torch.tensor,
        num_batches_per_epoch=num_batches_per_epoch,
    )

def create_ts_dataset(df, freq, preserve_index):
    data = Dataset.from_pandas(df, preserve_index=False)
    data.set_transform(partial(transform_start_field, freq=freq))
    return data

def calculate_position_cardinalities(df, column_name):
    # Find the maximum length of the lists in the column
    max_length = max(df[column_name].apply(len))

    # Initialize a list to store the sets of unique elements for each position
    unique_elements = [set() for _ in range(max_length)]

    # Iterate through each list in the DataFrame
    for lst in df[column_name]:
        for idx, elem in enumerate(lst):
            unique_elements[idx].add(elem)

    # Calculate the cardinality for each position
    cardinalities = [len(elements) for elements in unique_elements]

    return cardinalities

def setup_training(freq, train_df, test_df=None, prediction_length=24, limit_lags=None,
                        embedding_dimension=2, encoder_layers=4, decoder_layers=4,
                        d_model=32):
    test_data = None
    train_data = create_ts_dataset(train_df, freq, False)
    if test_df is not None:
        test_data = create_ts_dataset(test_df, freq, True)
    lags_sequence = get_lags_for_frequency(freq)
    if limit_lags is not None:
        lags_sequence = [i for i in lags_sequence if i<=limit_lags]
    time_features = time_features_from_frequency_str(freq)

    if train_df['feat_static_cat'][0] is None:
        num_static_categorical_features = 0
        cardinality = None
        embedding_dimension = None
    else:
        num_static_categorical_features=len(train_df['feat_static_cat'][0])
        cardinality = calculate_position_cardinalities(train_df, 'feat_static_cat')
        embedding_dimension = [embedding_dimension]*num_static_categorical_features

    config = TimeSeriesTransformerConfig(
        prediction_length=prediction_length,
        # context length:
        context_length=prediction_length * 2,
        # lags coming from helper given the freq:
        lags_sequence=lags_sequence,
        # we'll add 2 time features ("month of year" and "age", see further):
        num_time_features=len(time_features) + 1,
        # we have a single static categorical feature, namely time series ID:
        num_static_categorical_features=num_static_categorical_features,
        # week of the year, building type, climate zone
        cardinality=cardinality, #[52,3,4], #must be > 1
        # the model will learn an embedding of size 2 for each of the possible values of each static feature
        embedding_dimension=embedding_dimension,
        
        # transformer params:
        encoder_layers=encoder_layers,
        decoder_layers=decoder_layers,
        d_model=d_model,
    )
    return train_data, test_data, config

def train_transformer(train_data, config, freq, epochs=200, batch_size=32, num_batches_per_epoch=16):
    train_dataloader = create_train_dataloader(
            config=config, freq=freq, 
            data=train_data, batch_size=batch_size,
            num_batches_per_epoch=num_batches_per_epoch,)

    transformer = TimeSeriesTransformerForPrediction(config)
    accelerator = Accelerator()
    device = accelerator.device

    transformer.to(device)
    optimizer = AdamW(transformer.parameters(), lr=6e-4, betas=(0.9, 0.95), weight_decay=1e-1)

    transformer, optimizer, train_dataloader = accelerator.prepare(
        transformer,
        optimizer,
        train_dataloader,
    )
    transformer.train()
    list_loss = []
    for epoch in range(epochs):
        total_loss = 0
        for idx, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            outputs = transformer(
                static_categorical_features=batch["static_categorical_features"].to(device)
                if config.num_static_categorical_features > 0
                else None,
                static_real_features=batch["static_real_features"].to(device)
                if config.num_static_real_features > 0
                else None,
                past_time_features=batch["past_time_features"].to(device),
                past_values=batch["past_values"].to(device),
                future_time_features=batch["future_time_features"].to(device),
                future_values=batch["future_values"].to(device),
                past_observed_mask=batch["past_observed_mask"].to(device),
                future_observed_mask=batch["future_observed_mask"].to(device),
            )
            # print(outputs)
            loss = outputs.loss
            total_loss += loss.item()


            # Backpropagation
            accelerator.backward(loss)
            optimizer.step()
        list_loss.append(total_loss)
    
    return transformer, list_loss

def evaluate_transformer(transformer, test_data, config, freq, batch_size=16):
    test_dataloader = create_test_dataloader(
        config=config,
        freq=freq,
        data=test_data,
        batch_size=batch_size
    )

    accelerator = Accelerator()
    device = accelerator.device
    transformer.to(device)

    transformer.eval()
    forecasts = []
    for batch in test_dataloader:
        outputs = transformer.generate(
            static_categorical_features=batch["static_categorical_features"].to(device)
            if config.num_static_categorical_features > 0
            else None,
            static_real_features=batch["static_real_features"].to(device)
            if config.num_static_real_features > 0
            else None,
            past_time_features=batch["past_time_features"].to(device),
            past_values=batch["past_values"].to(device),
            future_time_features=batch["future_time_features"].to(device),
            past_observed_mask=batch["past_observed_mask"].to(device),
        )
        forecasts.append(outputs.sequences.cpu().numpy())
    
    forecasts = np.vstack(forecasts)
    return forecasts
