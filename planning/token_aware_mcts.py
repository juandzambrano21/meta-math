# planning/token_aware_mcts.py

import math
import numpy as np
import random
from collections import deque
from typing import List, Tuple, Dict
from models.world_model import ProbabilisticWorldModel
from models.reward_function import TokenAwareRewardFunction
from utils.tokenizer import Tokenizer

class TokenAwareMCTSNode:
    def __init__(self, belief_state: dict, token_budget: int, parent: 'TokenAwareMCTSNode' = None, action: Tuple[float, float] = None, token_cost: int = 0):
        """
        Initializes a MCTS node.

        :param belief_state: Current belief state as a dictionary with 'mean' and 'cov'.
        :param token_budget: Remaining token budget.
        :param parent: Parent TokenAwareMCTSNode.
        :param action: Action taken to reach this node.
        :param token_cost: Cumulative token cost to reach this node.
        """
        self.belief_state = belief_state
        self.token_budget = token_budget
        self.parent = parent
        self.action = action
        self.token_cost = token_cost
        self.children: List['TokenAwareMCTSNode'] = []
        self.visits = 0
        self.total_value = 0.0
        self.untried_actions: List[Tuple[float, float]] = None
        self.world_model = None  # To be assigned externally

    def is_fully_expanded(self, actions: List[Tuple[float, float]]) -> bool:
        return len(self.children) == len(actions)

    def is_terminal(self) -> bool:
        return self.token_budget <= 0 or self._is_goal_reached()

    def _is_goal_reached(self) -> bool:
        if self.world_model:
            distance = np.linalg.norm(self.belief_state['mean'] - self.world_model.goal_position)
            return distance <= self.world_model.goal_threshold
        return False

class TokenAwareMCTS:
    def __init__(self, world_model: 'ProbabilisticWorldModel', reward_function: 'TokenAwareRewardFunction', actions: List[Tuple[float, float]], tokenizer: 'Tokenizer', max_depth: int = 10, exploration_constant: float = 1.4):
        """
        Initializes the Token-Aware MCTS.

        :param world_model: An instance of ProbabilisticWorldModel.
        :param reward_function: An instance of TokenAwareRewardFunction.
        :param actions: List of possible actions as tuples.
        :param tokenizer: An instance of Tokenizer.
        :param max_depth: Maximum depth for simulations.
        :param exploration_constant: Exploration parameter for UCB.
        """
        self.world_model = world_model
        self.reward_function = reward_function
        self.actions = actions
        self.tokenizer = tokenizer
        self.max_depth = max_depth
        self.exploration_constant = exploration_constant

    def search(self, root_belief_state: dict, token_budget: int, num_simulations: int = 1000) -> Tuple[float, float]:
        """
        Performs MCTS search to select the best action.

        :param root_belief_state: The initial belief state as a dictionary with 'mean' and 'cov'.
        :param token_budget: Remaining token budget.
        :param num_simulations: Number of simulations to run.
        :return: The best action determined by MCTS.
        """
        root_node = TokenAwareMCTSNode(root_belief_state, token_budget)
        root_node.world_model = self.world_model  # Assign world_model to root_node for goal checking

        for _ in range(num_simulations):
            node = self._tree_policy(root_node)
            reward = self._simulate(node)
            self._backup(node, reward)

        best_action = self._best_action(root_node)
        return best_action

    def _tree_policy(self, node: TokenAwareMCTSNode) -> TokenAwareMCTSNode:
        """
        Selects a node to expand based on the tree policy.

        :param node: Current MCTS node.
        :return: Node selected for expansion.
        """
        while not node.is_terminal() and not self._is_done(node):
            if not node.is_fully_expanded(self.actions):
                return self._expand(node)
            else:
                node = self._best_child(node)
        return node

    def _expand(self, node: TokenAwareMCTSNode) -> TokenAwareMCTSNode:
        """
        Expands a node by adding a new child node.

        :param node: The node to expand.
        :return: The newly created child node.
        """
        if node.untried_actions is None:
            node.untried_actions = self.actions.copy()
            random.shuffle(node.untried_actions)

        if not node.untried_actions:
            return node

        action = node.untried_actions.pop()
        action_token_cost = self.tokenizer.count_tokens(str(action))

        if node.token_budget < action_token_cost:
            return node  # Cannot expand due to token budget

        # Predict next belief state
        next_belief_state = self.world_model.predict_next_state(node.belief_state, action)

        # Simulate observation based on the world model
        observation = self.world_model.sample_state(next_belief_state)

        # Update belief state with the observation
        updated_belief_state = self.world_model.update_belief_state(next_belief_state, observation)

        # Determine if the action is exploratory
        exploration = self._is_exploratory(node.belief_state, action, observation)

        # Assess reasoning quality with both action and observation
        reasoning_quality = self._assess_reasoning_quality(action, observation)

        # Compute reward
        distance_to_goal = np.linalg.norm(observation - self.world_model.goal_position)
        reward = self.reward_function.compute_reward(
            base_reward=-distance_to_goal,
            tokens_used=action_token_cost,
            exploration=exploration,
            reasoning_quality=reasoning_quality
        )

        # Incorporate token efficiency into reward
        token_efficiency = self._compute_token_efficiency(reward, action_token_cost)
        reward += token_efficiency

        # Create child node
        child_node = TokenAwareMCTSNode(
            belief_state=updated_belief_state,
            token_budget=node.token_budget - action_token_cost,
            parent=node,
            action=action,
            token_cost=node.token_cost + action_token_cost
        )
        child_node.world_model = self.world_model  # Assign world_model for goal checking
        node.children.append(child_node)

        return child_node

    def _is_exploratory(self, current_belief: dict, action: Tuple[float, float], observation: np.ndarray) -> bool:
        """
        Determines if an action is exploratory based on the change in belief.

        :param current_belief: Current belief state before action.
        :param action: Action taken as a tuple.
        :param observation: Observation received as a numpy array.
        :return: Boolean indicating if the action is exploratory.
        """
        # Heuristic: if the observation significantly changes the belief
        expected_observation = current_belief['mean'] + np.array(action)
        distance = np.linalg.norm(observation - expected_observation)
        return distance > 0.5  # Threshold for exploration

    def _assess_reasoning_quality(self, action: Tuple[float, float], observation: np.ndarray) -> float:
        """
        Assesses the quality of reasoning based on the action and observation.

        :param action: Action taken as a tuple.
        :param observation: Observation received as a numpy array.
        :return: Scalar representing reasoning quality.
        """
        # Example: Higher quality if the action moves closer to the goal
        distance_before = np.linalg.norm(self.world_model.goal_position - action)
        distance_after = np.linalg.norm(observation - self.world_model.goal_position)
        improvement = distance_before - distance_after
        return max(improvement, 0.0)

    def _compute_token_efficiency(self, reward: float, tokens_used: int) -> float:
        """
        Computes a bonus based on token efficiency.

        :param reward: The current reward.
        :param tokens_used: Number of tokens used.
        :return: Token efficiency bonus.
        """
        # Example: Bonus inversely proportional to tokens used
        efficiency_bonus = 1.0 / (tokens_used + 1)  # +1 to avoid division by zero
        return efficiency_bonus

    def _simulate_observation(self, belief_state: dict) -> np.ndarray:
        """
        Simulates an observation based on the belief state.

        :param belief_state: Current belief state as a dictionary with 'mean' and 'cov'.
        :return: Simulated observation as a numpy array.
        """
        return self.world_model.sample_state(belief_state)

    def _simulate(self, node: TokenAwareMCTSNode) -> float:
        """
        Simulates a reward from the node using a rollout policy.

        :param node: The node from which to start simulation.
        :return: Simulated cumulative reward.
        """
        current_belief = node.belief_state.copy()
        token_budget = node.token_budget
        total_reward = 0.0
        depth = 0

        while not node.is_terminal() and depth < self.max_depth:
            action = self._rollout_policy(current_belief, token_budget)
            if action is None:
                break

            action_token_cost = self.tokenizer.count_tokens(str(action))
            if token_budget < action_token_cost:
                break

            # Predict next belief state
            next_belief = self.world_model.predict_next_state(current_belief, action)

            # Simulate observation
            observation = self.world_model.sample_state(next_belief)

            # Update belief state with the observation
            updated_belief = self.world_model.update_belief_state(next_belief, observation)

            # Determine if the action is exploratory
            exploration = self._is_exploratory(current_belief, action, observation)

            # Assess reasoning quality with both action and observation
            reasoning_quality = self._assess_reasoning_quality(action, observation)

            # Compute reward
            distance_to_goal = np.linalg.norm(observation - self.world_model.goal_position)
            reward = self.reward_function.compute_reward(
                base_reward=-distance_to_goal,
                tokens_used=action_token_cost,
                exploration=exploration,
                reasoning_quality=reasoning_quality
            )

            # Incorporate token efficiency into reward
            token_efficiency = self._compute_token_efficiency(reward, action_token_cost)
            reward += token_efficiency

            total_reward += reward

            # Update for next simulation step
            current_belief = updated_belief
            token_budget -= action_token_cost
            depth += 1

        return total_reward

    def _rollout_policy(self, belief_state: dict, token_budget: int) -> Tuple[float, float]:
        """
        Defines the rollout policy, preferring actions with lower token costs and higher expected rewards.

        :param belief_state: Current belief state as a dictionary with 'mean' and 'cov'.
        :param token_budget: Remaining token budget.
        :return: Selected action for rollout as a tuple.
        """
        # Heuristic: prioritize actions that move closer to the goal
        best_action = None
        min_distance = float('inf')
        for action in self.actions:
            action_token_cost = self.tokenizer.count_tokens(str(action))
            if token_budget < action_token_cost:
                continue
            predicted_state = belief_state['mean'] + np.array(action)
            distance = np.linalg.norm(predicted_state - self.world_model.goal_position)
            if distance < min_distance:
                min_distance = distance
                best_action = action
        return best_action

    def _backup(self, node: TokenAwareMCTSNode, reward: float):
        """
        Backpropagates the reward up the tree.

        :param node: The node to start backpropagation from.
        :param reward: The reward to propagate.
        """
        while node is not None:
            node.visits += 1
            node.total_value += reward
            node = node.parent

    def _best_action(self, root_node: TokenAwareMCTSNode) -> Tuple[float, float]:
        """
        Selects the best action based on the highest average reward.

        :param root_node: The root MCTS node.
        :return: The best action.
        """
        if not root_node.children:
            return random.choice(self.actions)  # Fallback to random action
        # Select child with highest average reward
        best_child = max(
            root_node.children,
            key=lambda c: c.total_value / c.visits if c.visits > 0 else -float('inf')
        )
        return best_child.action

    def _best_child(self, node: TokenAwareMCTSNode) -> TokenAwareMCTSNode:
        """
        Selects the best child node based on the UCB formula, incorporating token cost.

        :param node: The current MCTS node.
        :return: The best child node.
        """
        best_score = float('-inf')
        best_child = None
        for child in node.children:
            if child.visits == 0:
                ucb = float('inf')
            else:
                exploitation = child.total_value / child.visits
                exploration = self.exploration_constant * math.sqrt(math.log(node.visits) / child.visits)
                ucb = exploitation + exploration
            if ucb > best_score:
                best_score = ucb
                best_child = child
        return best_child

    def _is_done(self, node: TokenAwareMCTSNode) -> bool:
        """
        Determines if the node represents a terminal state.

        :param node: The MCTS node.
        :return: Boolean indicating if the node is terminal.
        """
        return node.is_terminal()

    def reset(self):
        """
        Resets the MCTS state. Currently, MCTS is stateless, so no action is needed.
        """
        pass  
