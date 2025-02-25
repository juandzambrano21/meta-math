# planning/token_aware_mcts.py

import math
import numpy as np
import random
from typing import List, Tuple, Dict, Optional, Any

from models.world_model import ProbabilisticWorldModel
from models.reward_function import TokenAwareRewardFunction
from utils.tokenizer import Tokenizer


class TokenAwareMCTSNode:
    def __init__(
        self,
        belief_state: Dict[str, Any],
        token_budget: int,
        parent: Optional['TokenAwareMCTSNode'] = None,
        action: Optional[Tuple[str, ...]] = None,
        token_cost: int = 0,
        current_goal: Optional[str] = None
    ):
        """
        Initializes a MCTS node.

        :param belief_state: Current belief state as a dictionary with 'mean', 'cov', and 'tactics_applied'.
        :param token_budget: Remaining token budget.
        :param parent: Parent TokenAwareMCTSNode.
        :param action: Action taken to reach this node.
        :param token_cost: Cumulative token cost to reach this node.
        :param current_goal: The current proof goal.
        """
        self.belief_state = belief_state
        self.token_budget = token_budget
        self.parent = parent
        self.action = action
        self.token_cost = token_cost
        self.children: List['TokenAwareMCTSNode'] = []
        self.visits = 0
        self.total_value = 0.0
        self.untried_actions: List[Tuple[str, ...]] = []
        self.world_model: Optional[ProbabilisticWorldModel] = None  # To be assigned externally
        self.current_goal = current_goal  # Store current_goal in the node

    def is_fully_expanded(self, actions: List[Tuple[str, ...]]) -> bool:
        return len(self.children) == len(actions)

    def is_terminal(self) -> bool:
        return self.token_budget <= 0 or self._is_goal_reached()

    def _is_goal_reached(self) -> bool:
        if self.world_model:
            distance = np.linalg.norm(self.belief_state['mean'] - self.world_model.goal_position)
            return distance <= self.world_model.goal_threshold
        return False


class TokenAwareMCTS:
    def __init__(
        self,
        world_model: ProbabilisticWorldModel,
        reward_function: TokenAwareRewardFunction,
        actions: List[Tuple[str, ...]],
        tokenizer: Tokenizer,
        max_depth: int = 10,
        exploration_constant: float = 1.4
    ):
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

    def search(self, root_belief_state: Dict[str, Any], token_budget: int, current_goal: str, num_simulations: int = 1000) -> Tuple[Optional[Tuple[str, ...]], int]:
        """
        Performs MCTS search to select the best action.

        :param root_belief_state: The initial belief state as a dictionary with 'mean', 'cov', and 'tactics_applied'.
        :param token_budget: Remaining token budget.
        :param current_goal: The current proof goal.
        :param num_simulations: Number of simulations to run.
        :return: The best action determined by MCTS and the tokens used.
        """
        self.current_token_budget = token_budget

        if 'mean' not in root_belief_state or 'cov' not in root_belief_state or 'tactics_applied' not in root_belief_state:
            print("Invalid root belief state. Cannot perform MCTS search.")
            return None, 0

        root_node = TokenAwareMCTSNode(root_belief_state, token_budget, current_goal=current_goal)
        root_node.world_model = self.world_model

        for sim in range(num_simulations):
            node = self._tree_policy(root_node, current_goal)
            reward = self._simulate(node, current_goal)
            self._backup(node, reward)

            if (sim + 1) % 100 == 0:
                print(f"Completed {sim + 1} simulations...")

        best_action = self._best_action(root_node)
        tokens_used = root_node.token_cost

        self.current_token_budget -= tokens_used

        return best_action, tokens_used

    def _tree_policy(self, node: TokenAwareMCTSNode, current_goal: str) -> TokenAwareMCTSNode:
        """
        Selects a node to expand based on the tree policy.

        :param node: Current MCTS node.
        :param current_goal: The current proof goal.
        :return: Node selected for expansion.
        """
        while not node.is_terminal() and not self._is_done(node):
            if not node.is_fully_expanded(self.actions):
                return self._expand(node, current_goal)
            else:
                node = self._best_child(node)
        return node

    def _expand(self, node: TokenAwareMCTSNode, current_goal: str) -> TokenAwareMCTSNode:
        """
        Expands a node by adding a new child node.

        :param node: The node to expand.
        :param current_goal: The current proof goal.
        :return: The newly created child node.
        """
        if not node.untried_actions:
            node.untried_actions = self.actions.copy()
            random.shuffle(node.untried_actions)

        if not node.untried_actions:
            return node

        action = node.untried_actions.pop()
        action_token_cost = self.tokenizer.count_tokens(str(action))

        if node.token_budget < action_token_cost:
            print(f"Insufficient token budget for action {action}. Skipping expansion.")
            return node  # Cannot expand due to token budget

        # Predict next belief state and tokens used using the transition model
        try:
            next_belief_state, tokens_used = self.world_model.transition_model_func(node.belief_state, action, current_goal)
        except Exception as e:
            print(f"Error in transition model: {e}")
            return node  # Cannot expand due to transition model failure

        if node.token_budget < tokens_used:
            print(f"Insufficient token budget for tokens used ({tokens_used}) in action {action}. Skipping expansion.")
            return node  # Cannot expand due to token budget

        new_token_budget = node.token_budget - tokens_used

        # Sample an observation based on the new belief state
        observation = self.world_model.sample_observation(next_belief_state)

        # Update belief state with the observation using the ProbabilisticWorldModel's method
        updated_belief_state = self.world_model.update_belief_state(next_belief_state, observation)

        # Determine if the action is exploratory
        exploration = self._is_exploratory(node.belief_state, action, observation)

        # Assess reasoning quality with both action and observation
        reasoning_quality = self._assess_reasoning_quality(action, observation)

        # Compute reward
        distance_to_goal = np.linalg.norm(observation - self.world_model.goal_position)
        reward = self.reward_function.compute_reward(
            base_reward=-distance_to_goal,
            tokens_used=action_token_cost + tokens_used,  # Include tokens used in transition
            exploration=exploration,
            reasoning_quality=reasoning_quality
        )

        # Incorporate token efficiency into reward
        token_efficiency = self._compute_token_efficiency(reward, action_token_cost + tokens_used)
        reward += token_efficiency

        # Create child node with the updated belief state
        child_node = TokenAwareMCTSNode(
            belief_state=updated_belief_state,
            token_budget=new_token_budget,
            parent=node,
            action=action,
            token_cost=node.token_cost + tokens_used,
            current_goal=current_goal
        )
        child_node.world_model = self.world_model
        node.children.append(child_node)

        return child_node

    def _is_exploratory(self, current_belief: Dict[str, Any], action: Tuple[str, ...], observation: np.ndarray) -> bool:
        """
        Determines if an action is exploratory based on the change in belief.

        :param current_belief: Current belief state before action.
        :param action: Action taken as a tuple.
        :param observation: Observation received as a numpy array.
        :return: Boolean indicating if the action is exploratory.
        """
        # Heuristic: if the observation significantly changes the belief
        distance = np.linalg.norm(observation - current_belief['mean'])
        return distance > 0.5  # Threshold for exploration

    def _assess_reasoning_quality(self, action: Tuple[str, ...], observation: np.ndarray) -> float:
        """
        Assesses the quality of reasoning based on the action and observation.

        :param action: Action taken as a tuple.
        :param observation: Observation received as a numpy array.
        :return: Scalar representing reasoning quality.
        """
        # Example: Higher quality if the action moves closer to the goal
        distance = np.linalg.norm(observation - self.world_model.goal_position)
        improvement = max(0.0, 1.0 - (distance / (np.linalg.norm(self.world_model.goal_position) + 1e-6)))
        return improvement

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

    def _simulate_observation(self, belief_state: Dict[str, Any]) -> np.ndarray:
        """
        Simulates an observation based on the belief state.

        :param belief_state: Current belief state as a dictionary with 'mean' and 'cov'.
        :return: Simulated observation as a numpy array.
        """
        return self.world_model.sample_observation(belief_state)

    def _simulate(self, node: TokenAwareMCTSNode, current_goal: str) -> float:
        """
        Simulates a rollout from the given node to estimate the potential reward.

        :param node: The node to simulate from.
        :param current_goal: The current proof goal.
        :return: Estimated total reward from the simulation.
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

            # Predict next belief state and tokens used using the transition model
            try:
                next_belief_state, tokens_used = self.world_model.transition_model_func(current_belief, action, current_goal)
            except Exception as e:
                print(f"Error in transition model during simulation: {e}")
                break  # Cannot proceed with simulation

            # Check if token budget allows for the tokens used
            if token_budget < tokens_used:
                break
            token_budget -= tokens_used

            # Sample an observation based on the new belief state
            observation = self.world_model.sample_observation(next_belief_state)

            # Update belief state with the observation using the ProbabilisticWorldModel's method
            updated_belief_state = self.world_model.update_belief_state(next_belief_state, observation)

            # Determine if the action is exploratory
            exploration = self._is_exploratory(current_belief, action, observation)

            # Assess reasoning quality with both action and observation
            reasoning_quality = self._assess_reasoning_quality(action, observation)

            # Compute reward
            distance_to_goal = np.linalg.norm(observation - self.world_model.goal_position)
            reward = self.reward_function.compute_reward(
                base_reward=-distance_to_goal,
                tokens_used=action_token_cost + tokens_used,  # Include tokens used in transition
                exploration=exploration,
                reasoning_quality=reasoning_quality
            )

            # Incorporate token efficiency into reward
            token_efficiency = self._compute_token_efficiency(reward, action_token_cost + tokens_used)
            reward += token_efficiency

            total_reward += reward

            # Update for next simulation step
            current_belief = updated_belief_state
            depth += 1

        return total_reward

    def _rollout_policy(self, belief_state: Dict[str, Any], token_budget: int) -> Optional[Tuple[str, ...]]:
        """
        Defines the rollout policy, preferring actions with lower token costs and higher expected rewards.

        :param belief_state: Current belief state as a dictionary with 'mean', 'cov', and 'tactics_applied'.
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
            action_effect = self._action_effect(action)
            predicted_mean = belief_state['mean'] + np.array([action_effect])
            distance = np.linalg.norm(predicted_mean - self.world_model.goal_position)
            if distance < min_distance:
                min_distance = distance
                best_action = action
        return best_action

    def _action_effect(self, action: Tuple[str, ...]) -> float:
        """
        Determines the numeric effect of an action on the state.

        :param action: Action taken as a tuple.
        :return: Numeric effect of the action.
        """
        # Define a mapping from action to its numeric effect
        if action[0] == 'ApplyTactic':
            tactic = action[1] if len(action) > 1 else ''
            if tactic == 'intro':
                return 1.0
            elif tactic == 'induction':
                return 1.0
            elif tactic == 'apply':
                return 1.0
            elif tactic == 'split':
                return 0.5
            elif tactic == 'destruct':
                return 0.5
            elif tactic == 'contradiction':
                return -1.0
        elif action[0] == 'ProofStrategy':
            strategy = action[1] if len(action) > 1 else ''
            if strategy == 'SimplifyGoal':
                return -0.1
            elif strategy == 'TryLemma':
                return 0.1
        # Default effect
        return 0.0

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

    def _best_action(self, root_node: TokenAwareMCTSNode) -> Optional[Tuple[str, ...]]:
        """
        Selects the best action based on the highest average reward.

        :param root_node: The root MCTS node.
        :return: The best action or None if no action is found.
        """
        if not root_node.children:
            return None  # No actions available

        # Select child with highest average reward
        best_child = max(
            root_node.children,
            key=lambda c: (c.total_value / c.visits) if c.visits > 0 else -float('inf')
        )
        return best_child.action

    def _best_child(self, node: TokenAwareMCTSNode) -> TokenAwareMCTSNode:
        """
        Selects the best child node based on the UCB formula.

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
