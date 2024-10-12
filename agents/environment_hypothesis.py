import numpy as np
from typing import Callable, Tuple, Optional, Dict
from scipy.stats import multivariate_normal


class EnvironmentHypothesis:
    def __init__(
        self,
        name: str,
        transition_model: Callable[[np.ndarray, Tuple[str, ...]], np.ndarray],
        observation_model: Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]],
        process_noise_cov: Optional[np.ndarray] = None,
        observation_noise_cov: Optional[np.ndarray] = None,
        is_dynamic: bool = False
    ):
        """
        Initializes an Environment Hypothesis.

        :param name: Identifier for the hypothesis.
        :param transition_model: Function(state, action) -> new_state.
        :param observation_model: Function(observation, new_state) -> (H, R).
        :param process_noise_cov: Process noise covariance matrix.
        :param observation_noise_cov: Observation noise covariance matrix.
        :param is_dynamic: Indicates if the hypothesis can evolve over time.
        """
        self.name = name
        self.transition_model = transition_model
        self.observation_model = observation_model
        self.process_noise_cov = process_noise_cov if process_noise_cov is not None else np.eye(2) * 0.05
        self.observation_noise_cov = observation_noise_cov if observation_noise_cov is not None else np.eye(2) * 0.1
        self.is_dynamic = is_dynamic
        self.belief_state: Optional[Dict[str, np.ndarray]] = None  # {'mean': np.ndarray, 'cov': np.ndarray}

    def initialize_belief_state(self, initial_state: np.ndarray, initial_covariance: Optional[np.ndarray] = None):
        """
        Initializes the belief state with the initial state.

        :param initial_state: The starting state of the environment as a numpy array.
        :param initial_covariance: Covariance matrix for the initial belief.
        """
        if initial_covariance is None:
            initial_covariance = np.eye(len(initial_state)) * 0.1  # Small initial uncertainty
        self.belief_state = {
            'mean': initial_state,
            'cov': initial_covariance
        }

    def predict_next_state(self, action: Tuple[str, ...]):
        """
        Predicts the next belief state based on the action taken.

        :param action: The action taken by the agent as a tuple.
        """
        if self.belief_state is None:
            raise ValueError("Belief state not initialized.")

        # Predict the next state using the transition model
        new_mean = self.transition_model(self.belief_state['mean'], action)
        new_cov = self.belief_state['cov'] + self.process_noise_cov

        self.belief_state = {
            'mean': new_mean,
            'cov': new_cov
        }

    def update_belief_state(self, observation: np.ndarray):
        """
        Updates the belief state based on the observation received using the Kalman Filter.

        :param observation: The observation received as a numpy array.
        """
        if self.belief_state is None:
            raise ValueError("Belief state not initialized.")

        # Kalman Filter Update
        H, R = self.observation_model(observation, self.belief_state['mean'])

        # Innovation or measurement residual
        y = observation - H @ self.belief_state['mean']
        S = H @ self.belief_state['cov'] @ H.T + R

        try:
            # Kalman Gain
            K = self.belief_state['cov'] @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            print(f"LinAlgError: Cannot invert S matrix with covariance:\n{S}")
            K = np.zeros((self.belief_state['cov'].shape[0], H.shape[0]))

        # Updated state estimate
        new_mean = self.belief_state['mean'] + K @ y
        # Updated estimate covariance
        new_cov = (np.eye(len(self.belief_state['mean'])) - K @ H) @ self.belief_state['cov']

        self.belief_state = {
            'mean': new_mean,
            'cov': new_cov
        }

    def compute_likelihood(self, observation: np.ndarray) -> float:
        """
        Computes the likelihood P(observation | state) using the observation model.

        :param observation: The observation received as a numpy array.
        :return: Likelihood value.
        """
        if self.belief_state is None:
            raise ValueError("Belief state not initialized.")

        H, R = self.observation_model(observation, self.belief_state['mean'])

        S = H @ self.belief_state['cov'] @ H.T + R
        y = observation - H @ self.belief_state['mean']
        try:
            likelihood = multivariate_normal.pdf(y, mean=np.zeros(len(y)), cov=S)
        except np.linalg.LinAlgError:
            print(f"LinAlgError: Cannot compute PDF with covariance:\n{S}")
            likelihood = 1e-10  # Assign a small likelihood in case of numerical issues

        return likelihood
