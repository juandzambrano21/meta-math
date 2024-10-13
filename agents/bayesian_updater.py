# agents/bayesian_updater.py

from typing import Dict, Tuple, Optional, Any
import numpy as np
from agents.environment_hypothesis import EnvironmentHypothesis
from scipy.stats import multivariate_normal


class BayesianUpdater:
    def __init__(
        self,
        hypotheses: Dict[str, EnvironmentHypothesis],
        prior_distribution: Optional[Dict[str, float]] = None,
        smoothing: float = 1e-6
    ):
        """
        Initializes the BayesianUpdater with hypotheses and their prior probabilities.

        :param hypotheses: A dictionary mapping hypothesis names to EnvironmentHypothesis instances.
        :param prior_distribution: A dictionary mapping hypothesis names to their prior probabilities.
                                   If None, uniform priors are assumed.
        :param smoothing: A small constant to prevent zero probabilities.
        """
        self.hypotheses = hypotheses
        self.smoothing = smoothing

        if prior_distribution is None:
            num_hypotheses = len(hypotheses)
            uniform_prob = 1.0 / num_hypotheses
            self.prior_distribution = {name: uniform_prob for name in hypotheses}
        else:
            if set(prior_distribution.keys()) != set(hypotheses.keys()):
                raise ValueError("Prior distribution keys must match hypothesis names.")
            self.prior_distribution = prior_distribution.copy()

        self.posterior_distribution: Dict[str, float] = self.prior_distribution.copy()

    def update(
        self,
        action: Tuple[str, ...],
        observation: np.ndarray,
        current_goal: str
    ):
        """
        Updates the posterior distribution based on the action taken and observation received.

        :param action: The action taken by the agent.
        :param observation: The observation received as a numpy array.
        :param current_goal: The current proof goal as a string.
        """
        likelihoods: Dict[str, float] = {}
        total_likelihood = 0.0

        # Calculate likelihood for each hypothesis
        for hypo_name, hypo in self.hypotheses.items():
            # Predict the next state based on the action
            hypo.predict_next_state(action, current_goal)

            # Compute the likelihood of the observation given the new state
            likelihood = hypo.compute_likelihood(observation)
            likelihoods[hypo_name] = likelihood

            # Accumulate total likelihood weighted by prior
            total_likelihood += likelihood * self.prior_distribution[hypo_name]

        # Update posterior probabilities using Bayes' theorem with smoothing
        for hypo_name in self.posterior_distribution:
            prior = self.prior_distribution[hypo_name]
            likelihood = likelihoods.get(hypo_name, 0.0)
            if total_likelihood > 0:
                posterior = (likelihood * prior) / total_likelihood
            else:
                # Apply smoothing to avoid zero probabilities
                posterior = prior

            # Incorporate smoothing
            posterior += self.smoothing
            self.posterior_distribution[hypo_name] = posterior

        # Normalize posterior distribution
        self.normalize_posteriors()

    def normalize_posteriors(self):
        """
        Normalizes the posterior probabilities so that they sum to 1.
        """
        total = sum(self.posterior_distribution.values())
        if total > 0:
            for hypo_name in self.posterior_distribution:
                self.posterior_distribution[hypo_name] /= total
        else:
            # Reassign uniform probabilities if total is zero
            num_hypotheses = len(self.posterior_distribution)
            for hypo_name in self.posterior_distribution:
                self.posterior_distribution[hypo_name] = 1.0 / num_hypotheses

    def get_posterior(self) -> Dict[str, float]:
        """
        Returns the current posterior distribution.

        :return: A dictionary mapping hypothesis names to their posterior probabilities.
        """
        return self.posterior_distribution.copy()

    def aggregate_belief_state(self) -> Optional[Dict[str, Any]]:
        """
        Aggregates the belief states from all hypotheses weighted by their posterior probabilities.

        :return: A dictionary with combined 'mean' and 'cov' or None if aggregation fails.
        """
        combined_mean = None
        combined_cov = None

        for hypo_name, prob in self.posterior_distribution.items():
            hypo = self.hypotheses[hypo_name]
            if hypo.belief_state is None:
                continue
            if combined_mean is None:
                combined_mean = prob * hypo.belief_state['mean']
                combined_cov = prob * hypo.belief_state['cov']
            else:
                combined_mean += prob * hypo.belief_state['mean']
                combined_cov += prob * hypo.belief_state['cov']

        if combined_mean is not None and combined_cov is not None:
            return {'mean': combined_mean, 'cov': combined_cov}
        else:
            print("Aggregation failed: No valid belief states found.")
            return None

    def reset(self):
        """
        Resets the posterior distribution to the prior distribution.
        """
        self.posterior_distribution = self.prior_distribution.copy()
