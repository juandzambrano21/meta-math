# agents/bayesian_updater.py

import numpy as np
from typing import Dict
from agents.prior_distribution import PriorDistribution
from models.world_model import ProbabilisticWorldModel
from agents.environment_hypothesis import EnvironmentHypothesis

class BayesianUpdater:
    def __init__(self, prior_distribution: PriorDistribution, world_model: ProbabilisticWorldModel):
        """
        Initializes the BayesianUpdater.

        :param prior_distribution: An instance of PriorDistribution.
        :param world_model: An instance of ProbabilisticWorldModel.
        """
        self.prior_distribution = prior_distribution
        self.world_model = world_model
        self.posterior: Dict[EnvironmentHypothesis, float] = self.prior_distribution.get_prior()

    def update(self, action: np.ndarray, observation: np.ndarray) -> None:
        """
        Updates the posterior distribution based on the action taken and observation received.

        :param action: The action taken by the agent as a numpy array.
        :param observation: The observation received by the agent as a numpy array.
        """
        likelihoods: Dict[EnvironmentHypothesis, float] = {}
        total_likelihood = 0.0

        # Compute likelihood for each hypothesis
        for hypothesis, prior_prob in self.posterior.items():
            likelihood = hypothesis.compute_likelihood(action, observation)
            likelihoods[hypothesis] = likelihood
            total_likelihood += likelihood * prior_prob

        # Update posterior probabilities
        new_posterior: Dict[EnvironmentHypothesis, float] = {}
        for hypothesis in self.posterior:
            prior_prob = self.posterior[hypothesis]
            likelihood = likelihoods[hypothesis]
            if total_likelihood > 0:
                posterior_prob = (likelihood * prior_prob) / total_likelihood
            else:
                # Avoid division by zero by retaining prior probability
                posterior_prob = prior_prob
            new_posterior[hypothesis] = posterior_prob

        self.posterior = new_posterior

    def get_posterior(self) -> Dict[EnvironmentHypothesis, float]:
        """
        Returns the current posterior distribution.

        :return: A dictionary mapping EnvironmentHypothesis instances to their posterior probabilities.
        """
        return self.posterior.copy()

    def reset(self) -> None:
        """
        Resets the posterior to the prior distribution.
        """
        self.posterior = self.prior_distribution.get_prior()
