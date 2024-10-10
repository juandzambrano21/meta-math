import numpy as np
from scipy.stats import multivariate_normal

class EnvironmentHypothesis:
    def __init__(self, name, transition_model, observation_model, is_dynamic=False):
        """
        Initializes an Environment Hypothesis.

        :param name: Identifier for the hypothesis.
        :param transition_model: Function(state, action) -> new_state with noise.
        :param observation_model: Object with methods to compute observation probabilities.
        :param is_dynamic: Boolean indicating if the hypothesis can evolve.
        """
        self.name = name
        self.transition_model = transition_model
        self.observation_model = observation_model
        self.belief_state = None  # Represented as mean and covariance for Gaussian
        self.is_dynamic = is_dynamic

    def initialize_belief_state(self, initial_state, initial_covariance=None):
        """
        Initializes the belief state with the initial state.

        :param initial_state: The starting state of the environment as a numpy array.
        :param initial_covariance: Covariance matrix for the initial belief.
        """
        self.belief_state = {
            'mean': np.array(initial_state),
            'cov': np.array(initial_covariance) if initial_covariance is not None else np.eye(len(initial_state))
        }

    def predict_next_state(self, action):
        """
        Predicts the next belief state based on the action taken.

        :param action: The action taken by the agent as a numpy array.
        """
        mean = self.transition_model(self.belief_state['mean'], action)
        cov = self.belief_state['cov'] + self.transition_model.noise_cov
        self.belief_state = {'mean': mean, 'cov': cov}

    def update_belief_state(self, observation):
        """
        Updates the belief state based on the observation received.

        :param observation: The observation received as a numpy array.
        """
        # Kalman Filter Update
        H = self.observation_model.H
        R = self.observation_model.R
        S = H @ self.belief_state['cov'] @ H.T + R
        K = self.belief_state['cov'] @ H.T @ np.linalg.inv(S)
        y = observation - H @ self.belief_state['mean']
        mean = self.belief_state['mean'] + K @ y
        cov = (np.eye(len(self.belief_state['mean'])) - K @ H) @ self.belief_state['cov']
        self.belief_state = {'mean': mean, 'cov': cov}

    def compute_observation_likelihood(self, observation):
        """
        Computes the likelihood P(observation | state) using the observation model.

        :param observation: The observation received as a numpy array.
        :return: Likelihood value.
        """
        mean = self.belief_state['mean']
        cov = self.belief_state['cov']
        return self.observation_model.prob(observation, mean, cov)
    
    def compute_likelihood(self, action: np.ndarray, observation: np.ndarray) -> float:
        """
        Computes the likelihood P(observation | action, hypothesis).

        :param action: The action taken by the agent as a numpy array.
        :param observation: The observation received by the agent as a numpy array.
        :return: The likelihood value as a float.
        """
        # Predict the next state based on the current belief state and action
        predicted_mean = self.transition_model(self.belief_state['mean'], action)
        predicted_cov = self.belief_state['cov'] + self.transition_model.noise_cov

        # Compute the likelihood of the observation given the predicted state
        likelihood = self.observation_model.prob(observation, predicted_mean, predicted_cov)
        return likelihood