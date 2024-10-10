# agents/agent.py

import torch
import numpy as np
from agents.bayesian_updater import BayesianUpdater
from agents.prior_distribution import PriorDistribution
from planning.token_aware_mcts import TokenAwareMCTS
from reasoning.formal_language import parse_reasoning_trace
from reasoning.interpreter import ReasoningInterpreter
from reasoning.coq_engine import CoqEngine
from utils.observation_encoder import ObservationEncoder
from environments.continuous_navigation import ContinuousNavigationEnv
from models.reward_function import TokenAwareRewardFunction
from models.world_model import ProbabilisticWorldModel
from reasoning.llm_tac_wrapper import LLMTACWrapper
from utils.tokenizer import Tokenizer
from agents.environment_hypothesis import EnvironmentHypothesis
from typing import List, Tuple
from knowledge.ontology import Ontology

class LLMAIXITACAgent:
    def __init__(
        self,
        hypotheses: List['EnvironmentHypothesis'],
        actions: List[Tuple[float, float]],
        tokenizer: 'Tokenizer',
        llm_tac_wrapper: 'LLMTACWrapper',
        world_model: 'ProbabilisticWorldModel',
        reward_function: 'TokenAwareRewardFunction',
        environment: 'ContinuousNavigationEnv',
        observation_encoder: 'ObservationEncoder',
        ontology: Ontology,
        coq_engine: 'CoqEngine'
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
        self.hypotheses = hypotheses
        self.actions = actions
        self.tokenizer = tokenizer
        self.llm_tac_wrapper = llm_tac_wrapper
        self.world_model = world_model
        self.reward_function = reward_function
        self.environment = environment
        self.observation_encoder = observation_encoder
        self.ontology = ontology  # Assign the ontology to the agent

        self.prior_distribution = PriorDistribution(hypotheses)
        self.bayesian_updater = BayesianUpdater(self.prior_distribution, self.world_model)
        self.mcts = TokenAwareMCTS(world_model, reward_function, actions, tokenizer)
        self.history = []
        self.preferred_action_vector = None
        self.interpreter = ReasoningInterpreter(self, coq_engine)  # Pass coq_engine
        self.action_mapping = {  # Mapping from action identifiers to vectors
            'A1': (1.0, 1.0),
            'A2': (-1.0, 1.0),
            'A3': (1.0, -1.0),
            'A4': (-1.0, -1.0),
            'A5': (0.0, 1.0),
            'A6': (0.0, -1.0),
            'A7': (1.0, 0.0),
            'A8': (-1.0, 0.0),
            'A9': (0.0, 0.0)  # No acceleration
        }

        self.token_budget = 0  # Initialize token budget
        self.current_node_id = 'Type'  # Starting node in the ontology

    def set_token_budget(self, budget: int):
        """
        Sets the token budget for the agent.

        :param budget: The total token budget.
        """
        self.reward_function.token_budget = budget
        self.token_budget = budget

    def select_action(self, goal: str) -> Tuple[float, float]:
        """
        Selects the next action based on reasoning and planning.

        :param goal: The goal description.
        :return: The selected action as a tuple.
        """
        # Get current observation from the environment
        current_observation = self.environment.get_observation()

        # Encode current observation using ObservationEncoder
        # Fixed tensor creation to remove UserWarning
        encoded_observation = self.observation_encoder(
            torch.from_numpy(current_observation).float().unsqueeze(0)
        ).detach().numpy()[0]

        # Generate reasoning trace with correct arguments
        reasoning_trace, tokens_used = self.llm_tac_wrapper.generate_reasoning_trace(
            goal=goal,
            current_state=current_observation,
            current_node=self.ontology.current_node_id,
            token_budget=self.token_budget
        )
        self.token_budget -= tokens_used
        print("=============RESONING TRACE",reasoning_trace)
        # Interpret reasoning trace
        reasoning_steps = parse_reasoning_trace(reasoning_trace)
        self.interpreter.execute_reasoning_steps(reasoning_steps)

        # Aggregate belief state from posterior
        belief_state = self.aggregate_belief_state()

        # Map ontology node to available actions
        available_actions = self.get_actions_based_on_ontology(self.ontology.current_node_id)

        # Use MCTS to select action
        action = self.mcts.search(belief_state, self.token_budget)

        # If the interpreter set a preferred action vector, use it
        if self.preferred_action_vector is not None:
            action = self.preferred_action_vector
            self.preferred_action_vector = None  # Reset for next cycle

        # Log action
        self.history.append((action, current_observation, None, self.token_budget))

        return action

    def create_prompt(self, goal: str, encoded_observation: np.ndarray) -> str:
        """
        Creates a prompt for the LLM based on the history, goal, and ontology state.

        :param goal: The goal description.
        :param encoded_observation: The encoded observation as a numpy array.
        :return: The prompt string.
        """
        prompt = f"Goal: {goal}\nToken Budget: {self.token_budget}\n"
        prompt += "Current Observation Encoding: " + ", ".join(map(str, encoded_observation)) + "\n"
        prompt += f"Current Ontology Node: {self.ontology.current_node_id}\n"
        neighbors = self.ontology.get_neighbors(self.ontology.current_node_id)
        prompt += f"Possible Ontology Moves: {neighbors}\n"
        prompt += "History:\n"
        for action, observation, reward, tb in self.history:
            prompt += f"Action: {action}, Observation: {observation}, Reward: {reward}, Remaining Tokens: {tb}\n"
        prompt += "Provide concise reasoning steps and suggest the next action within the token budget."
        return prompt

    def aggregate_belief_state(self) -> dict:
        """
        Aggregates the belief states from all hypotheses weighted by their posterior probabilities.

        :return: Aggregated belief state as a dictionary with 'mean' and 'cov'.
        """
        aggregated_mean = np.zeros_like(self.hypotheses[0].belief_state['mean'])
        aggregated_cov = np.zeros_like(self.hypotheses[0].belief_state['cov'])
        posterior = self.bayesian_updater.get_posterior()
        for hypo, prob in posterior.items():
            aggregated_mean += prob * hypo.belief_state['mean']
            # Compute the weighted covariance
            mean_diff = hypo.belief_state['mean'] - aggregated_mean
            aggregated_cov += prob * (hypo.belief_state['cov'] + np.outer(mean_diff, mean_diff))
        return {'mean': aggregated_mean, 'cov': aggregated_cov}

    def update(self, action: Tuple[float, float], observation: np.ndarray, reward: float, info: dict):
        """
        Updates the agent's knowledge based on the action taken and observation received.

        :param action: The action taken by the agent as a tuple (ax, ay).
        :param observation: The observation received as a numpy array.
        :param reward: The reward received.
        :param info: Additional information from the environment.
        """
        # Update Bayesian posterior
        action_vector = np.array(action)
        self.bayesian_updater.update(action_vector, observation)

        # Update belief states in each hypothesis
        for hypo in self.hypotheses:
            hypo.predict_next_state(action_vector)
            hypo.update_belief_state(observation)

        # Update the agent's current ontology node based on environment info
        if 'ontology_node' in info and info['ontology_node'] is not None:
            self.ontology.current_node_id = info['ontology_node']

        # Update history with reward
        self.history[-1] = (action, observation, reward, self.token_budget)

    def reset(self):
        """
        Resets the agent's state for a new episode.
        """
        self.history = []
        self.prior_distribution.reset()
        self.bayesian_updater.reset()
        self.mcts.reset()
        self.ontology.reset()
        self.token_budget = self.reward_function.token_budget
        self.current_node_id = 'Type'
        self.preferred_action_vector = None

    def get_actions_based_on_ontology(self, current_node_id: str) -> List[Tuple[float, float]]:
        """
        Maps the current ontology node to a subset of available actions.

        :param current_node_id: The ID of the current ontology node.
        :return: A list of available actions as tuples.
        """
        # Define mappings based on ontology nodes
        ontology_action_map = {
            'Type': [(1.0, 1.0), (1.0, 0.0), (0.0, 1.0)],
            'Term': [(-1.0, 1.0), (-1.0, 0.0), (0.0, 1.0)],
            'Univalence': [(0.0, -1.0), (1.0, -1.0)],
            'Circle': [(1.0, 0.0), (-1.0, 0.0)],
            # TODO: WE NEED TO COMPILE THIS AUTOMATICALLY
        }

        return ontology_action_map.get(current_node_id, self.actions)  # Default to all actions
