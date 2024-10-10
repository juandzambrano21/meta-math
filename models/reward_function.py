import math

class TokenAwareRewardFunction:
    def __init__(self, base_reward_weight=1.0, token_penalty_weight=0.05, exploration_reward=10.0, reasoning_quality_weight=0.1, token_budget=1000):
        """
        Initializes the TokenAwareRewardFunction.

        :param base_reward_weight: Weight for the base reward.
        :param token_penalty_weight: Weight for the token penalty.
        :param exploration_reward: Reward for exploration actions.
        :param reasoning_quality_weight: Weight for reasoning quality.
        :param token_budget: Total token budget.
        """
        self.base_reward_weight = base_reward_weight
        self.token_penalty_weight = token_penalty_weight
        self.exploration_reward = exploration_reward
        self.reasoning_quality_weight = reasoning_quality_weight
        self.token_budget = token_budget

    def compute_reward(self, base_reward, tokens_used, exploration=False, reasoning_quality=1.0):
        """
        Computes the total reward by combining the base reward, token penalty, exploration, and reasoning quality.

        :param base_reward: The primary reward from the environment.
        :param tokens_used: The number of tokens used in reasoning.
        :param exploration: Boolean indicating if the action is exploratory.
        :param reasoning_quality: A scalar representing the quality of reasoning.
        :return: The combined reward.
        """
        # Non-linear penalty: exponential decay
        token_penalty = self.token_penalty_weight * math.exp(tokens_used / self.token_budget)
        exploration_bonus = self.exploration_reward if exploration else 0.0
        reasoning_bonus = self.reasoning_quality_weight * reasoning_quality
        total_reward = self.base_reward_weight * base_reward - token_penalty + exploration_bonus + reasoning_bonus
        return total_reward