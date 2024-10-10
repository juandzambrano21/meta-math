# models/world_model.py

import numpy as np
from typing import Tuple, Dict, Any

class ProbabilisticWorldModel:
    def __init__(
        self,
        transition_model,
        observation_model,
        goal_position: np.ndarray,
        goal_threshold: float = 0.5
    ):
        """
        Initializes the ProbabilisticWorldModel.

        :param transition_model: A function that defines state transitions.
        :param observation_model: A function that defines observations based on the state.
        :param goal_position: A numpy array representing the goal position in the state space.
        :param goal_threshold: Distance threshold to consider the goal as reached.
        """
        self.transition_model = transition_model
        self.observation_model = observation_model
        self.goal_position = goal_position
        self.goal_threshold = goal_threshold
        self.process_noise_cov = getattr(transition_model, 'process_noise_cov', np.eye(len(goal_position)) * 0.05)
        self.observation_noise_cov = getattr(observation_model, 'observation_noise_cov', np.eye(len(goal_position)) * 0.1)

    def update_belief_state(self, state: Dict[str, Any], observation: np.ndarray) -> Dict[str, Any]:
        """
        Updates the belief state based on the observation.

        :param state: Current belief state as a dictionary with 'mean' and 'cov'.
        :param observation: Observation received as a numpy array.
        :return: Updated belief state.
        """
        # Example implementation using a simple Kalman Filter update
        H, R = self.observation_model(observation, state['mean'])
        S = np.dot(H, np.dot(state['cov'], H.T)) + R
        K = np.dot(state['cov'], np.dot(H.T, np.linalg.inv(S)))
        y = observation - np.dot(H, state['mean'])
        new_mean = state['mean'] + np.dot(K, y)
        I = np.eye(len(state['mean']))
        new_cov = np.dot(I - np.dot(K, H), state['cov'])
        return {'mean': new_mean, 'cov': new_cov}

    def sample_observation(self, belief_state: Dict[str, Any]) -> np.ndarray:
        """
        Samples an observation based on the current belief state.

        :param belief_state: Current belief state as a dictionary with 'mean' and 'cov'.
        :return: Sampled observation as a numpy array.
        """
        mean = belief_state['mean']
        cov = self.observation_noise_cov
        return np.random.multivariate_normal(mean, cov)

    def transition_model_func(self, state: np.ndarray, action: Tuple[str, ...]) -> np.ndarray:
        """
        Applies the transition model to the current state and action.

        :param state: Current state as a numpy array.
        :param action: Action taken as a tuple.
        :return: New state as a numpy array.
        """
        return self.transition_model(state, action)
