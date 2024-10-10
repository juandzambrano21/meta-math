import numpy as np

class PriorDistribution:
    def __init__(self, hypotheses, smoothing=1e-6):
        """
        Initializes the Prior Distribution over Environment Hypotheses.

        :param hypotheses: List of EnvironmentHypothesis instances.
        :param smoothing: Smoothing factor to avoid zero probabilities.
        """
        self.hypotheses = hypotheses
        self.smoothing = smoothing
        self.prior = self.initialize_prior()

    def initialize_prior(self):
        # Assign uniform priors initially
        num_hypotheses = len(self.hypotheses)
        prior = {hypo: 1.0 / num_hypotheses for hypo in self.hypotheses}
        return prior

    def update_prior(self, likelihoods):
        """
        Updates the prior based on the likelihoods.

        :param likelihoods: Dictionary {EnvironmentHypothesis: likelihood}.
        """
        total = sum(likelihoods.values())
        if total > 0:
            for hypo in self.hypotheses:
                self.prior[hypo] = (likelihoods[hypo] + self.smoothing) / (total + self.smoothing * len(self.hypotheses))
        else:
            # Apply smoothing to avoid zero probabilities
            num_hypotheses = len(self.hypotheses)
            self.prior = {hypo: self.smoothing / (self.smoothing * num_hypotheses) for hypo in self.hypotheses}

    def get_prior(self):
        return self.prior.copy()