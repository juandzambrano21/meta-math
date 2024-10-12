# environments/ontology_navigation.py

from knowledge.ontology import Ontology
from typing import Tuple, Dict, Any
from reasoning.coq_engine import CoqEngine

class OntologyNavigationEnv:
    def __init__(self, ontology: Ontology, goal_node_id: str, coq_engine: CoqEngine):
        """
        Initializes the Ontology Navigation Environment.
        """
        self.ontology = ontology
        self.goal_node_id = goal_node_id
        self.done = False
        self.proof_state = self.initialize_proof_state()
        self.coq_engine = coq_engine  # Assign CoqEngine to the environment

    def reset(self):
        """
        Resets the environment to the initial state.
        """
        self.done = False
        self.proof_state = self.initialize_proof_state()
        return self.get_observation()

    def step(self, action: Tuple[str, ...]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        Executes an action in the environment.
        """
        if self.done:
            raise Exception("Environment has been terminated. Please reset.")

        action_type = action[0]

        if action_type == 'ApplyTactic':
            tactic = action[1]
            success = self.apply_tactic(tactic)
            reward = 10.0 if success else -1.0
        elif action_type == 'QueryOntology':
            query_type = action[1]
            query_param = action[2]
            result = self.query_ontology(query_type, query_param)
            reward = -0.5
        elif action_type == 'ProofStrategy':
            strategy = action[1]
            success = self.apply_proof_strategy(strategy)
            reward = 5.0 if success else -2.0
        else:
            reward = -10.0  # Penalty for unknown actions

        observation = self.get_observation()
        info = {'proof_state': self.proof_state}

        if self.is_proof_complete():
            self.done = True
            reward = 100.0  # Reward for completing the proof

        return observation, reward, self.done, info

    def initialize_proof_state(self) -> Dict[str, Any]:
        goal_statement = self.ontology.graph.nodes[self.goal_node_id]['Proposition']
        print(f"Initial Goal: {goal_statement}")  # Debug statement
        return {'goal': goal_statement, 'tactics_applied': []}


    def apply_tactic(self, tactic: str) -> bool:
        coq_code = self.generate_coq_code_for_tactic(tactic)
        print("Generated Coq Code:\n", coq_code)  # Debug statement

        if not coq_code.strip():
            print("Generated Coq code is empty. Skipping execution.")
            return False

        results, metadata = self.coq_engine.forward(coq_code)
        coq_output = results[0].value['output'] if results else ''

        if metadata.get('status') == 'success':
            self.proof_state['tactics_applied'].append(tactic)
            self.update_proof_state_after_success(tactic)
            return True
        else:
            print(f"Coq execution failed: {metadata.get('message', '')}")
            return False


    def generate_coq_code_for_tactic(self, tactic: str) -> str:
        """
        Generates Coq code for applying the given tactic to the current goal.
        """
        coq_code = f"""
        Lemma proof_{len(self.proof_state['tactics_applied'])}: {self.proof_state['goal']}.
        Proof.
        {tactic}.
        Qed.
        """
        return coq_code
    
    def update_proof_state_after_success(self, tactic: str):
        """
        Updates the proof state after a successful tactic application.
        """
        self.proof_state['goal'] = None  # Proof completed


    def query_ontology(self, query_type: str, query_param: str) -> Any:
        """
        Performs a query on the ontology.
        """
        if query_type == 'FindRelatedType':
            related_nodes = self.ontology.get_neighbors(query_param)
            return related_nodes
        else:
            return []

    def apply_proof_strategy(self, strategy: str) -> bool:
        """
        Applies a proof strategy.
        """
        if strategy == 'SimplifyGoal':
            if self.proof_state['goal']:
                self.proof_state['goal'] = 'Simplified ' + self.proof_state['goal']
                return True
            else:
                return False
        else:
            return False

    def is_proof_complete(self) -> bool:
        """
        Checks if the proof is complete.
        """
        return self.proof_state['goal'] is None


    def get_observation(self) -> Dict[str, Any]:
        """
        Returns the current proof state.
        """
        return self.proof_state

    def render(self):
        """
        Prints the current proof state.
        """
        print(f"Current Proof State: {self.proof_state}")
