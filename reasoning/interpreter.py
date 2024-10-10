# reasoning/interpreter.py

import numpy as np
from reasoning.coq_engine import CoqEngine
from typing import List, Tuple

class ReasoningInterpreter:
    def __init__(self, agent: 'LLMAIXITACAgent', coq_engine: CoqEngine):
        """
        Initializes the ReasoningInterpreter.

        :param agent: The LLMAIXITACAgent instance.
        :param coq_engine: An instance of CoqEngine.
        """
        self.agent = agent
        self.coq_engine = coq_engine  # Corrected from lean_engine to coq_engine

    def execute_reasoning_steps(self, reasoning_steps: List[Tuple]):
        """
        Executes a list of reasoning step tuples.

        :param reasoning_steps: List of reasoning steps.
        """
        for step in reasoning_steps:
            step_type = step[0]
            if step_type == 'PredictAction':
                self.handle_predict_action(step[1])
            elif step_type == 'UpdateState':
                self.handle_update_state(step[1], step[2])
            elif step_type == 'PredictObservation':
                self.handle_predict_observation(step[1], step[2])
            elif step_type == 'EvaluateGoal':
                self.handle_evaluate_goal(step[1])
            elif step_type == 'MoveTo':
                self.handle_move_to(step[1])
            elif step_type == 'Inspect':
                self.handle_inspect(step[1])
            else:
                print(f"Unknown reasoning step type: {step_type}")

    def handle_predict_action(self, action_id: str):
        """
        Handles the PredictAction reasoning step.

        :param action_id: The predicted action identifier.
        """
        # Translate action identifier to actual action vector
        action_vector = self.agent.action_mapping.get(action_id)
        if action_vector is not None:
            self.agent.preferred_action_vector = action_vector
            print(f"Interpreted PredictAction: {action_id} -> {action_vector}")
        else:
            print(f"Unknown action identifier: {action_id}")

    def handle_update_state(self, state_id: str, action_id: str):
        """
        Handles the UpdateState reasoning step.

        :param state_id: The state identifier.
        :param action_id: The action identifier.
        """
        print(f"Interpreted UpdateState: State {state_id}, Action {action_id}")
        # Update the agent's belief state based on the action
        action_vector = self.agent.action_mapping.get(action_id)
        if action_vector is not None:
            for hypo in self.agent.hypotheses:
                hypo.predict_next_state(np.array(action_vector))
            print(f"Updated belief states based on action {action_id}")
        else:
            print(f"Unknown action identifier: {action_id}")

    def handle_predict_observation(self, state_id: str, observation_id: str):
        """
        Handles the PredictObservation reasoning step.

        :param state_id: The state identifier.
        :param observation_id: The observation identifier.
        """
        print(f"Interpreted PredictObservation: From State {state_id} predict Observation {observation_id}")
        # Map observation_id to actual observation vector based on ontology
        observation = self.map_observation(observation_id)
        if observation is not None:
            for hypo in self.agent.hypotheses:
                hypo.update_belief_state(observation)
            print(f"Updated belief states with observation {observation_id}")
        else:
            print(f"Unknown observation identifier: {observation_id}")

    def handle_evaluate_goal(self, state_id: str):
        """
        Handles the EvaluateGoal reasoning step.

        :param state_id: The state identifier.
        """
        print(f"Interpreted EvaluateGoal: Current State {state_id}")
        aggregated_belief_state = self.agent.aggregate_belief_state()
        current_mean = aggregated_belief_state['mean']
        goal_position = self.agent.world_model.goal_position
        distance = np.linalg.norm(current_mean - goal_position)
        if distance <= self.agent.world_model.goal_threshold:
            print("Goal Reached!")
            self.agent.environment.done = True
        else:
            print(f"Current estimated distance to goal: {distance:.2f}")

    def handle_move_to(self, node_id: str):
        """
        Handles the MoveTo reasoning step.

        :param node_id: The ID of the node to move to.
        """
        print(f"Interpreted MoveTo: Moving to node {node_id}")
        if node_id in self.agent.ontology.graph.nodes:
            current_node_id = self.agent.ontology.current_node_id
            if node_id in self.agent.ontology.get_neighbors(current_node_id):
                self.agent.ontology.current_node_id = node_id
                print(f"Moved to node {node_id}")
            else:
                print(f"Cannot move to node {node_id} from {current_node_id}")
        else:
            print(f"Node {node_id} does not exist in the ontology")

    def handle_inspect(self, node_id: str):
        """
        Handles the Inspect reasoning step.

        :param node_id: The ID of the node to inspect.
        """
        print(f"Interpreted Inspect: Inspecting node {node_id}")
        if node_id in self.agent.ontology.graph.nodes:
            node_info = self.agent.ontology.get_node(node_id)
            print(f"Node {node_id} info: {node_info}")
        else:
            print(f"Node {node_id} does not exist in the ontology")

    def map_observation(self, observation_id: str) -> np.ndarray:
        """
        Maps a symbolic observation identifier to an actual observation vector, using the ontology.

        :param observation_id: The observation identifier (e.g., 'O1').
        :return: Observation as a numpy array.
        """
        # Use the ontology to retrieve observation data
        if observation_id in self.agent.ontology.graph.nodes:
            node_data = self.agent.ontology.get_node(observation_id)
            # Extract relevant data to form an observation vector
            # For example, use node attributes like 'Description' length as a feature
            description = node_data.get('Description', '')
            prop_length = len(node_data.get('Proposition', '')) if 'Proposition' in node_data else 0
            type_complexity = len(node_data.get('Type', '')) if 'Type' in node_data else 0
            # Create an observation vector based on extracted features
            observation_vector = np.array([len(description), prop_length, type_complexity], dtype=float)
            # Normalize the observation vector if necessary
            observation_vector = observation_vector / np.linalg.norm(observation_vector) if np.linalg.norm(observation_vector) != 0 else observation_vector
            return observation_vector
        else:
            print(f"Observation ID {observation_id} not found in ontology.")
            return None
