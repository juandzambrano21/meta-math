# agents/bayesian_updater.py

from typing import Dict, Tuple
import numpy as np
from agents.environment_hypothesis import EnvironmentHypothesis


class BayesianUpdater:
    def __init__(self, prior_distribution: Dict[str, float]):
        """
        Initializes the BayesianUpdater with prior probabilities.

        :param prior_distribution: A dictionary mapping hypothesis names to their prior probabilities.
        """
        self.prior_distribution = prior_distribution.copy()
        self.posterior_distribution: Dict[str, float] = prior_distribution.copy()

    def update(
        self,
        action: Tuple[str, ...],
        observation: np.ndarray,
        hypotheses: Dict[str, EnvironmentHypothesis]
    ):
        """
        Updates the posterior distribution based on the action taken and observation received.

        :param action: The action taken by the agent as a tuple.
        :param observation: The observation received as a numpy array.
        :param hypotheses: A dictionary mapping hypothesis names to EnvironmentHypothesis instances.
        """
        likelihoods: Dict[str, float] = {}
        total_likelihood = 0.0

        # Calculate likelihood for each hypothesis
        for hypo_name, hypo in hypotheses.items():
            # Predict the next state based on the action
            hypo.predict_next_state(action)

            # Compute the likelihood of the observation given the new state
            likelihood = hypo.compute_likelihood(observation)
            likelihoods[hypo_name] = likelihood

            # Accumulate total likelihood weighted by prior
            total_likelihood += likelihood * self.prior_distribution[hypo_name]

        # Update posterior probabilities using Bayes' theorem
        for hypo_name in self.posterior_distribution:
            prior = self.prior_distribution[hypo_name]
            likelihood = likelihoods[hypo_name]
            if total_likelihood > 0:
                posterior = (likelihood * prior) / total_likelihood
            else:
                # Avoid division by zero; retain prior if total likelihood is zero
                posterior = prior
            self.posterior_distribution[hypo_name] = posterior

    def get_posterior(self) -> Dict[str, float]:
        """
        Returns the current posterior distribution.

        :return: A dictionary mapping hypothesis names to their posterior probabilities.
        """
        return self.posterior_distribution.copy()

    def reset(self):
        """
        Resets the posterior distribution to the prior distribution.
        """
        self.posterior_distribution = self.prior_distribution.copy()
