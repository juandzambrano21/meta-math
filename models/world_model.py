# models/world_model.py

import numpy as np
from scipy.stats import multivariate_normal

class ProbabilisticWorldModel:
    def __init__(self, transition_model, observation_model):
        """
        Initializes the ProbabilisticWorldModel.

        :param transition_model: Function(state, action) -> new_state with noise.
        :param observation_model: Object with methods to compute observation probabilities.
        """
        self.transition_model = transition_model
        self.observation_model = observation_model
        self.goal_position = None  # To be set externally if needed
        self.goal_threshold = 1.0  # Default threshold for goal achievement

    def predict_next_state(self, belief_state, action):
        """
        Predicts the next belief state based on the current belief state and action.

        :param belief_state: Current belief state as a dictionary with 'mean' and 'cov'.
        :param action: Action taken by the agent as a numpy array.
        :return: Predicted next belief state as a dictionary with 'mean' and 'cov'.
        """
        mean = self.transition_model(belief_state['mean'], action)
        cov = belief_state['cov'] + self.transition_model.noise_cov
        return {'mean': mean, 'cov': cov}

    def update_belief_state(self, belief_state, observation):
        """
        Updates the belief state based on the received observation.

        :param belief_state: Current belief state as a dictionary with 'mean' and 'cov'.
        :param observation: Observation received by the agent as a numpy array.
        :return: Updated belief state as a dictionary with 'mean' and 'cov'.
        """
        # Kalman Filter Update
        H = self.observation_model.H
        R = self.observation_model.R
        S = H @ belief_state['cov'] @ H.T + R
        K = belief_state['cov'] @ H.T @ np.linalg.inv(S)
        y = observation - H @ belief_state['mean']
        mean = belief_state['mean'] + K @ y
        cov = (np.eye(len(belief_state['mean'])) - K @ H) @ belief_state['cov']
        return {'mean': mean, 'cov': cov}

    def sample_state(self, belief_state):
        """
        Samples a state based on the belief state.

        :param belief_state: Current belief state as a dictionary with 'mean' and 'cov'.
        :return: Sampled state as a numpy array.
        """
        return np.random.multivariate_normal(belief_state['mean'], belief_state['cov'])
