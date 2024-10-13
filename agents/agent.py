# agents/agent.py

import torch
import numpy as np
import random
from typing import List, Tuple, Dict, Any, Optional, Callable

from agents.bayesian_updater import BayesianUpdater
from agents.environment_hypothesis import EnvironmentHypothesis
from planning.token_aware_mcts import TokenAwareMCTS
from reasoning.formal_language import parse_reasoning_trace
from reasoning.interpreter import ReasoningInterpreter
from reasoning.coq_engine import CoqEngine
from reasoning.llm_tac_wrapper import LLMTACWrapper
from utils.tokenizer import Tokenizer
from utils.observation_encoder import ObservationEncoder
from knowledge.ontology import Ontology
from environments.ontology_navigation import OntologyNavigationEnv
from models.reward_function import TokenAwareRewardFunction
from models.world_model import ProbabilisticWorldModel


class LLMAIXITACAgent:
    def __init__(
        self,
        hypotheses: List[EnvironmentHypothesis],
        actions: List[Tuple[str, ...]],
        tokenizer: Tokenizer,
        llm_tac_wrapper: LLMTACWrapper,
        world_model: ProbabilisticWorldModel,
        reward_function: TokenAwareRewardFunction,
        environment: OntologyNavigationEnv,
        observation_encoder: ObservationEncoder,
        ontology: Ontology,
        coq_engine: CoqEngine
    ):
        """
        Initializes the LLMAIXITACAgent.

        :param hypotheses: List of EnvironmentHypothesis instances.
        :param actions: List of possible actions as tuples.
        :param tokenizer: An instance of Tokenizer.
        :param llm_tac_wrapper: An instance of LLMTACWrapper.
        :param world_model: An instance of ProbabilisticWorldModel.
        :param reward_function: An instance of TokenAwareRewardFunction.
        :param environment: The environment instance.
        :param observation_encoder: An instance of ObservationEncoder.
        :param ontology: An instance of Ontology.
        :param coq_engine: An instance of CoqEngine.
        """
        self.hypotheses = {hypo.name: hypo for hypo in hypotheses}
        self.actions = actions
        self.tokenizer = tokenizer
        self.llm_tac_wrapper = llm_tac_wrapper
        self.world_model = world_model
        self.reward_function = reward_function
        self.environment = environment
        self.observation_encoder = observation_encoder
        self.ontology = ontology

        # Initialize BayesianUpdater with prior_distribution and hypotheses
        prior_distribution = {h.name: 1.0 / len(hypotheses) for h in hypotheses}
        self.bayesian_updater = BayesianUpdater(hypotheses={hypo.name: hypo for hypo in hypotheses},
            prior_distribution=prior_distribution )

        # Initialize MCTS planner with BayesianUpdater
        self.mcts = TokenAwareMCTS(
            world_model=self.world_model,
            reward_function=self.reward_function,
            actions=self.actions,
            tokenizer=self.tokenizer,
           # bayesian_updater=self.bayesian_updater  # Pass BayesianUpdater if required
        )

        self.history = []
        self.preferred_action = None
        self.interpreter = ReasoningInterpreter(self, coq_engine)
        self.action_mapping = {f'A{i+1}': action for i, action in enumerate(self.actions)}

        self.token_budget = 0

    def set_token_budget(self, budget: int):
        """
        Sets the token budget for the agent.

        :param budget: The total token budget.
        """
        self.reward_function.token_budget = budget
        self.token_budget = budget

    def select_action(self, goal: str) -> Optional[Tuple[str, ...]]:
        """
        Selects the next action based on reasoning and planning.

        :param goal: The goal description.
        :return: The selected action as a tuple or None if no action is possible.
        """
        # Get current observation from the environment
        current_observation = self.environment.get_observation()

        # Encode current observation using ObservationEncoder
        proof_features = self.encode_proof_features(current_observation)
        encoded_observation = self.observation_encoder(
            torch.from_numpy(proof_features).float().unsqueeze(0)
        ).detach().numpy()[0]

        # Generate reasoning trace
        reasoning_trace, tokens_used = self.llm_tac_wrapper.generate_reasoning_trace(
            goal=goal,
            current_state=self.environment.proof_state['goal'],
            current_node=None,  # No current node
            token_budget=self.token_budget
        )
        self.token_budget -= tokens_used

        if reasoning_trace:
            # Parse and interpret reasoning trace
            reasoning_steps = parse_reasoning_trace(reasoning_trace)
            if reasoning_steps:
                step_type = reasoning_steps[0][0]
                if step_type == 'UnprovableStatement':
                    self.interpreter.execute_reasoning_steps(reasoning_steps)
                    print("Encountered an unprovable statement. Terminating the proof attempt.")
                    # Terminate the episode as the proof cannot be completed
                    return None
                else:
                    self.interpreter.execute_reasoning_steps(reasoning_steps)
            else:
                print("Reasoning trace parsing failed. Skipping reasoning steps.")
        else:
            print("No reasoning trace generated. Skipping reasoning steps.")

        # Aggregate belief state from BayesianUpdater
        belief_state = self.aggregate_belief_state()

        if belief_state is None or 'mean' not in belief_state or 'cov' not in belief_state:
            print("Invalid belief state. Selecting a random exploratory action.")
            exploratory_actions = [action for action in self.actions if action[0] == 'ApplyTactic']
            if exploratory_actions:
                action = random.choice(exploratory_actions)
                self.history.append((action, current_observation, None, self.token_budget))
                return action
            else:
                print("No exploratory actions available.")
                return None

        # Extract current_goal from the environment's proof state
        current_goal = self.environment.proof_state.get('goal', '')

        # Use MCTS to select action, passing current_goal
        action = self.mcts.search(belief_state, self.token_budget, current_goal)

        if action is None:
            print("MCTS did not return a valid action. Selecting a random exploratory action.")
            exploratory_actions = [action for action in self.actions if action[0] == 'ApplyTactic']
            if exploratory_actions:
                action = random.choice(exploratory_actions)
                self.history.append((action, current_observation, None, self.token_budget))
                return action
            else:
                print("No exploratory actions available.")
                return None

        # Log action
        self.history.append((action, current_observation, None, self.token_budget))

        return action

    def encode_proof_features(self, proof_state: Dict[str, Any]) -> np.ndarray:
        """
        Encodes proof state into a numerical feature vector.

        :param proof_state: The current proof state as a dictionary.
        :return: A numpy array of features.
        """
        goal = proof_state.get('goal', '')
        tactics_applied = proof_state.get('tactics_applied', [])
        # Example features: length of goal string, number of tactics applied
        features = np.array([len(goal), len(tactics_applied)], dtype=float)
        # Normalize the feature vector
        norm = np.linalg.norm(features)
        if norm > 0:
            features /= norm
        return features

    def aggregate_belief_state(self) -> Optional[Dict[str, Any]]:
        """
        Aggregates the belief states from all hypotheses weighted by their posterior probabilities.

        :return: A dictionary with combined 'mean' and 'cov' or None if aggregation fails.
        """
        posterior = self.bayesian_updater.get_posterior()
        combined_mean = None
        combined_cov = None

        for hypo_name, prob in posterior.items():
            hypo = self.hypotheses.get(hypo_name)
            if hypo is None or hypo.belief_state is None:
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

    def update(
        self,
        action: Tuple[str, ...],
        observation: Dict[str, Any],
        reward: float,
        info: Dict[str, Any]
    ):
        """
        Updates the agent's knowledge based on the action taken and observation received.

        :param action: The action taken by the agent.
        :param observation: The observation received.
        :param reward: The reward received.
        :param info: Additional information from the environment.
        """
        # Extract relevant information from observation
        proof_state = observation.get('proof_state', self.environment.proof_state)

        # Convert proof_state to numpy array for BayesianUpdater
        if proof_state.get('goal') is None:
            # Proof completed; no further observations
            observation_array = np.array([0.0, 0.0])  # Example placeholder
        else:
            # Example: Convert goal string length and number of tactics applied to array
            observation_array = np.array([len(proof_state.get('goal', '')), len(proof_state.get('tactics_applied', []))])

        # Update Bayesian posterior
            self.bayesian_updater.update(action, observation_array, proof_state.get('goal', ''))

        # Update the agent's proof state based on environment info
        self.environment.proof_state = proof_state

        # Update history with reward
        if self.history:
            last_entry = self.history[-1]
            self.history[-1] = (last_entry[0], last_entry[1], reward, self.token_budget)

    def reset(self):
        """
        Resets the agent's state for a new episode.
        """
        self.history = []
        self.bayesian_updater.reset()
        self.mcts.reset()
        self.environment.reset()
        self.token_budget = self.reward_function.token_budget
        self.preferred_action = None
