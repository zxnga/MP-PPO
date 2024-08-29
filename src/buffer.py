from typing import List, Optional, Tuple, Type, NamedTuple

import nump as np
import torch
from gymnasium import spaces

from stable_baselines3.common.buffers import ReplayBuffer

class IndependantStatesReplayBufferSamples(NamedTuple):
    observations: torch.Tensor
    actions: torch.Tensor
    next_observations: torch.Tensor
    dones: torch.Tensor
    rewards: torch.Tensor
    independant_states: np.ndarray
    next_independant_states: np.ndarray

class IndependantStateReplayBuffer(ReplayBuffer):
    """
        Replay buffer used in off-policy algorithms like SAC/TD3.

        :param buffer_size: Max number of element in the buffer
        :param observation_space: Observation space
        :param action_space: Action space
        :param device: PyTorch device
        :param n_envs: Number of parallel environments
        :param optimize_memory_usage: Enable a memory efficient variant
            of the replay buffer which reduces by almost a factor two the memory used,
            at a cost of more complexity.
            See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
            and https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274
            Cannot be used in combination with handle_timeout_termination.
        :param handle_timeout_termination: Handle timeout termination (due to timelimit)
            separately and treat the task as infinite horizon task.
            https://github.com/DLR-RM/stable-baselines3/issues/284
    """

    observations: np.ndarray
    next_observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    timeouts: np.ndarray
    independant_states: np.ndarray #actual values to pass to the transformer as past values
    next_independant_states: np.ndarray #rolling of 1 of independant states, needed to compute next actions
    
    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        episode_timesteps: int,
        lookback_horizon: int,
        circular_independant_states: bool,
        *args,
        **kwargs,
    ):
        super().__init__(buffer_size, observation_space, action_space, *args, **kwargs,)

        self.episode_timesteps = episode_timesteps
        self.lookback_horizon = lookback_horizon
        self.independant_state_pos = 0
        self.independant_states = np.zeros((self.episode_timesteps, self.n_envs), dtype=np.float32)
        self.env_time_steps = np.zeros((self.buffer_size, self.n_envs), dtype=np.int32)
        self.circular_independant_states = circular_independant_states
        if circular_independant_states:
            print(f"circular_independant_states set to True, which assumes circular independant states !")

        assert lookback_horizon > 0, f"lookback_horizon must be > 0, got {lookback_horizon}"


    def add(self, *args, independant_state: Optional[np.ndarray] = None, env_time_step: Optional[int], **kwargs) -> None:
        if not self.independant_state_pos == self.episode_timesteps:
            self.independant_states[self.independant_state_pos] = np.array(independant_state)
            self.independant_state_pos += 1
        self.env_time_steps[self.pos] = np.array(env_time_step)
        super().add(*args, **kwargs)

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> IndependantStatesReplayBufferSamples:
        # Sample randomly the env idx
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

        independant_states = self.sample_independant_states(batch_inds, env_indices)
        # TODO: next one: assume circular episode
        next_independant_states = self.sample_independant_states(batch_inds+1%self.episode_timesteps, env_indices)      

        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(self.observations[(batch_inds + 1) % self.buffer_size, env_indices, :], env)
        else:
            next_obs = self._normalize_obs(self.next_observations[batch_inds, env_indices, :], env)

        data = (
            self._normalize_obs(self.observations[batch_inds, env_indices, :], env),
            self.actions[batch_inds, env_indices, :],
            next_obs,
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            (self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(-1, 1),
            self._normalize_reward(self.rewards[batch_inds, env_indices].reshape(-1, 1), env),
            # Change dtype of arrays to add a dimension for training purposes as arrays have different sizes
            np.array(independant_states, dtype=object).reshape(-1,1),
            np.array(next_independant_states, dtype=object).reshape(-1, 1),
        )
        #don't convert independent states to tensor due to dimension mismatch
        tensor_data = tuple(map(self.to_torch, data[:5]))
        final_data = tensor_data + data[5:]
        # Note: if n_envs > 1 need to specify which env corresponds to which independant state samples
        # Cause static fetaures might not be the same for each env (TODO), check TSExtractor.get_static_features()
        return IndependantStatesReplayBufferSamples(*final_data)

    def sample_independant_states(self, batch_inds, env_indices):
        results = []
        for indexA, indexB in zip(batch_inds, env_indices):
            prev_values = self.get_independant_state(indexA, indexB)
            results.append(prev_values)
        return results

    def get_independant_state(self, pos_buffer: int, pos_n_envs: int):
        """
            Get the self.lookback_horizon previous independant states from the buffer.
            Note: If circular_independant_states is True independant states are considered circular.
            :param pos_buffer: First dimension of self.independant_states, which state we wish to retrieve.
            :param pos_n_envs: Second dimension of self.independant_states, corresponds to the env from which we wish
                            to retrieve the independant state (in case of multiple envs and different independant states).
        """
        if not self.circular_independant_states:
            # Get the reset coordinates for the environment
            reset_coordinates = self.get_reset_coordinates()
            
            # Find the most recent reset point for the given environment (pos_n_envs) that is before or at pos_buffer
            recent_reset_points = [pos for pos, env in reset_coordinates if env == pos_n_envs and pos <= pos_buffer]
            last_reset_point = max(recent_reset_points) if recent_reset_points else 0
            
            # Ensure pos_buffer does not go before the last reset point
            start_index = max(last_reset_point, pos_buffer - self.lookback_horizon)
        else:
            # Assumes circular independant_states
            start_index = max(0, pos_buffer - self.lookback_horizon)
        
        # Avoid empty arrays when sampling when restting the env (so only 1 independant state at pos_buffer)
        if start_index == pos_buffer:
            if pos_buffer == self.independant_states.shape[0]:
                start_index -= 1
            else:
                pos_buffer += 1
        ind_states_start = self.env_time_steps[start_index, pos_n_envs]
        ind_states_end = self.env_time_steps[pos_buffer, pos_n_envs]
    
        return self.independant_states[ind_states_start:ind_states_end, pos_n_envs]


    def get_reset_coordinates(self):
        positions = np.where(self.env_time_steps == 0)
        filtered_indices = positions[0] <= self.pos
        positions = (positions[0][filtered_indices], positions[1][filtered_indices])
        return list(zip(*positions))
