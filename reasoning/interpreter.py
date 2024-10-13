# reasoning/interpreter.py

import numpy as np
from reasoning.coq_engine import CoqEngine
from typing import List, Tuple, Optional
from reasoning.formal_language import parse_reasoning_trace
import re

class ReasoningInterpreter:
    def __init__(self, agent: 'LLMAIXITACAgent', coq_engine: CoqEngine):
        """
        Initializes the ReasoningInterpreter.
        """
        self.agent = agent
        self.coq_engine = coq_engine

    def execute_reasoning_steps(self, reasoning_steps: List[Tuple]):
        """
        Executes a list of reasoning step tuples.
        """
        for step in reasoning_steps:
            step_type = step[0]
            if step_type == 'PredictAction':
                action = step[1]
                if action[0] == 'ApplyTactic':
                    self.handle_apply_tactic(action[1])
                elif action[0] == 'QueryOntology':
                    self.handle_query_ontology(action[1], action[2])
                elif action[0] == 'ProofStrategy':
                    self.handle_proof_strategy(action[1])
                else:
                    print(f"Unknown action: {action}")
            elif step_type == 'UpdateState':
                self.handle_update_state(step[1], step[2])
            elif step_type == 'PredictObservation':
                self.handle_predict_observation(step[1], step[2])
            elif step_type == 'EvaluateGoal':
                self.handle_evaluate_goal(step[1])
            elif step_type == 'UnprovableStatement':
                self.handle_unprovable_statement()
                # Stop processing further steps
                break
            else:
                print(f"Unknown reasoning step type: {step_type}")

    def handle_apply_tactic(self, tactic: str):
        """
        Handles the ApplyTactic reasoning step.
        """
        print(f"Applying tactic: {tactic}")

        coq_code, tokens_used = self.agent.llm_tac_wrapper.generate_coq_code_for_tactic(
            tactic,
            self.agent.environment.proof_state['goal']
        )
        coq_code = re.sub(r'```|coq', '', coq_code)

        print("Generated Coq Code:\n", coq_code)  
        self.agent.token_budget -= tokens_used

        if not coq_code.strip():
            print("Generated Coq code is empty. Skipping execution.")
            return

        results, metadata = self.coq_engine.forward(coq_code)

        if metadata.get('status') == 'success':
            self.agent.environment.proof_state['tactics_applied'].append(tactic)
            self.agent.environment.update_proof_state_after_success(tactic, coq_output=metadata.get('status'))
            print(f"Tactic '{tactic}' applied successfully.")
        else:
            print(f"Coq execution failed: {metadata.get('message', '')}")
            print(f"Tactic '{tactic}' failed.")


    def handle_query_ontology(self, query_type: str, query_param: str):
        """
        Handles the QueryOntology reasoning step.
        """
        print(f"Querying ontology: {query_type} with parameter '{query_param}'")
        result = self.agent.environment.query_ontology(query_type, query_param)
        print(f"Query result: {result}")

    def handle_proof_strategy(self, strategy: str):
        """
        Handles the ProofStrategy reasoning step.
        """
        print(f"Applying proof strategy: {strategy}")
        success = self.agent.environment.apply_proof_strategy(strategy)
        if success:
            print(f"Strategy '{strategy}' applied successfully.")
        else:
            print(f"Strategy '{strategy}' failed.")

    def handle_update_state(self, state_id: str, action_id: str):
        """
        Handles the UpdateState reasoning step.
        """
        print(f"Interpreted UpdateState: State {state_id}, Action {action_id}")
        # Update the agent's belief state based on the action
        pass

    def handle_predict_observation(self, state_id: str, observation_id: str):
        """
        Handles the PredictObservation reasoning step.
        """
        print(f"Interpreted PredictObservation: From State {state_id} predict Observation {observation_id}")
        pass

    def handle_evaluate_goal(self, state_id: str):
        """
        Handles the EvaluateGoal reasoning step.
        """
        print(f"Interpreted EvaluateGoal: Current State {state_id}")
        if self.agent.environment.is_proof_complete():
            print("Goal Reached! Proof is complete.")
            self.agent.environment.done = True
        else:
            print("Proof is not yet complete.")

    def handle_unprovable_statement(self):
        """
        Handles the scenario where the goal is identified as an axiom or unprovable statement.
        """
        print("Detected an axiom or unprovable statement. No reasoning steps will be generated.")
        self.agent.environment.done = True
