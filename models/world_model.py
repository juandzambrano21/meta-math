# models/world_model.py

import numpy as np
from typing import Tuple, Dict, Any, Callable, Optional
from scipy.stats import multivariate_normal


class ProbabilisticWorldModel:
    def __init__(
        self,
        observation_model: Callable[[Optional[np.ndarray], np.ndarray], Tuple[np.ndarray, np.ndarray]],
        transition_model_func: Callable[[Dict[str, Any], Tuple[str, ...], str], Tuple[Dict[str, Any], int]],
        process_noise_cov: Optional[np.ndarray] = None,
        goal_position: Optional[np.ndarray] = None,
        goal_threshold: float = 0.5,
        hypotheses: Optional[Dict[str, Any]] = None
    ):
        """
        Initializes the ProbabilisticWorldModel with observation and transition models.
    
        :param observation_model: A callable that takes in the observation and current mean,
                                  and returns the observation matrix H and observation noise covariance R.
        :param transition_model_func: A callable that takes in the current proof state, action, and goal,
                                     and returns the predicted next proof state and tokens used.
        :param process_noise_cov: Process noise covariance matrix. Defaults to identity if not provided.
        :param goal_position: The target position the agent aims to reach.
        :param goal_threshold: The distance threshold to consider the goal as reached.
        :param hypotheses: A dictionary of environment hypotheses.
        """
        self.observation_model = observation_model
        self.transition_model_func = transition_model_func
        self.process_noise_cov = process_noise_cov if process_noise_cov is not None else np.eye(2) * 0.05
        self.goal_position = goal_position if goal_position is not None else np.array([10.0, 10.0])  # Example default
        self.goal_threshold = goal_threshold
        self.observation_noise_cov: Optional[np.ndarray] = None
        self.hypotheses = hypotheses if hypotheses is not None else {}

    def update_belief_state(self, state: Dict[str, Any], observation: np.ndarray) -> Dict[str, Any]:
        """
        Updates the belief state based on the observation using the Kalman Filter.
    
        :param state: Current belief state as a dictionary with 'mean', 'cov', and 'tactics_applied'.
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

        # Update tactics_applied if necessary
        new_tactics_applied = state.get('tactics_applied', []).copy()

        return {'mean': new_mean, 'cov': new_cov, 'tactics_applied': new_tactics_applied}

    def compute_likelihood(self, state: Dict[str, Any], observation: np.ndarray) -> float:
        """
        Computes the likelihood P(observation | state) using the observation model.
    
        :param state: Current belief state as a dictionary with 'mean', 'cov', and 'tactics_applied'.
        :param observation: Observation received as a numpy array.
        :return: Likelihood value.
        """
        H, R = self.observation_model(observation, state['mean'])

        S = H @ state['cov'] @ H.T + R
        y = observation - H @ state['mean']
        try:
            likelihood = multivariate_normal.pdf(y, mean=np.zeros(len(y)), cov=S)
        except np.linalg.LinAlgError:
            print(f"LinAlgError: Cannot compute PDF with covariance:\n{S}")
            likelihood = 1e-10  # Assign a small likelihood in case of numerical issues

        return likelihood

    def sample_observation(self, proof_state: Dict[str, Any]) -> np.ndarray:
        """
        Samples an observation based on the current proof state.
    
        :param proof_state: Current proof state as a dictionary with 'mean', 'cov', and 'tactics_applied'.
        :return: Sampled observation as a numpy array.
        """
        # Assuming the proof_state contains necessary information to determine H and R
        H, R = self.observation_model(None, proof_state['mean'])  # 'None' is used as a placeholder for observation input
        S = H @ proof_state['cov'] @ H.T + R
        try:
            observation = np.random.multivariate_normal(mean=np.zeros(H.shape[0]), cov=S)
        except np.linalg.LinAlgError:
            print(f"LinAlgError: Cannot sample observation with covariance:\n{S}")
            observation = np.zeros(H.shape[0])
        return observation

    def transition_model_func(self, proof_state: Dict[str, Any], action: Tuple[str, ...], current_goal: str) -> Tuple[Dict[str, Any], int]:
        new_proof_state = proof_state.copy()
        
        # Initialize 'tactics_applied' if it doesn't exist
        if 'tactics_applied' not in new_proof_state:
            new_proof_state['tactics_applied'] = []
        
        # Handle different action types appropriately
        if action[0] == 'ApplyTactic' and len(action) > 1:
            tactic = action[1]
            new_proof_state['tactics_applied'].append(tactic)
            # Modify the 'mean' based on the tactic
            if tactic == 'intro':
                new_proof_state['mean'] += np.array([1.0, 0.0])
            elif tactic == 'split':
                new_proof_state['mean'] += np.array([0.0, 1.0])
            # Add more tactics as needed
        elif action[0] == 'ProofStrategy' and len(action) > 1:
            strategy = action[1]
            if strategy == 'SimplifyGoal':
                # Example: Reduce complexity
                new_proof_state['mean'] *= 0.9
            elif strategy == 'TryLemma':
                # Example: Attempt to apply a lemma
                new_proof_state['mean'] += np.array([0.5, 0.5])
        elif action[0] == 'QueryOntology':
            # Example: Modify state based on ontology query
            if len(action) > 2:
                related_type = action[2]
                # Modify 'mean' based on related_type
                new_proof_state['mean'] += np.array([0.2, 0.2])
        else:
            print(f"Unhandled action type or insufficient action parameters: {action}")
        
        # Calculate tokens used based on action complexity
        tokens_used = self._calculate_tokens_used(action)
        
        return new_proof_state, tokens_used

    def _calculate_tokens_used(self, action: Tuple[str, ...]) -> int:
        """
        Calculates the number of tokens used based on the action.

        :param action: Action taken as a tuple.
        :return: Number of tokens used.
        """
        # Placeholder implementation; replace with actual token calculation logic
        # For example, actions with more parameters might consume more tokens
        return len(action) * 5  # Example: 5 tokens per action element

