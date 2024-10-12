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
        Updates the belief state based on the observation using the Kalman Filter.

        :param state: Current belief state as a dictionary with 'mean' and 'cov'.
        :param observation: Observation received as a numpy array.
        :return: Updated belief state.
        """
        H, R = self.observation_model(observation, state['mean'])

        # Innovation or measurement residual
        y = observation - H @ state['mean']
        S = H @ state['cov'] @ H.T + R

        try:
            # Kalman Gain
            K = state['cov'] @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            print(f"LinAlgError: Cannot invert S matrix with covariance:\n{S}")
            K = np.zeros((state['cov'].shape[0], H.shape[0]))

        # Updated state estimate
        new_mean = state['mean'] + K @ y
        # Updated estimate covariance
        I = np.eye(len(state['mean']))
        new_cov = (I - K @ H) @ state['cov']

        return {'mean': new_mean, 'cov': new_cov}

    def sample_observation(self, belief_state: Dict[str, Any]) -> np.ndarray:
        mean = belief_state['mean']
        cov = self.observation_noise_cov
        print(f"mean: {mean}, type: {type(mean)}, shape: {np.shape(mean)}")
        print(f"cov: {cov}, type: {type(cov)}, shape: {np.shape(cov)}")
        return np.random.multivariate_normal(mean, cov)

    def transition_model_func(self, state: np.ndarray, action: Tuple[str, ...], current_goal: str) -> Tuple[np.ndarray, int]:

        """
        Applies the transition model to the current state and action.

        :param state: Current state as a numpy array.
        :param action: Action taken as a tuple.
        :return: New state as a numpy array.
        """
        return self.transition_model(state, action, current_goal)
