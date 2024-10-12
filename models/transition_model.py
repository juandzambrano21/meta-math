# models/transition_model.py

import numpy as np
from typing import Tuple
from reasoning.coq_engine import CoqEngine
from reasoning.llm_tac_wrapper import LLMTACWrapper
from utils.tokenizer import Tokenizer

class TransitionModel:
    def __init__(
        self,
        state_dim: int,
        coq_engine: CoqEngine,
        tokenizer: Tokenizer,
        llm_tac_wrapper: LLMTACWrapper
    ):
        """
        Initializes the TransitionModel with CoqEngine and LLMTACWrapper.

        :param state_dim: Dimension of the state vector.
        :param coq_engine: Instance of CoqEngine to execute Coq code.
        :param tokenizer: Instance of Tokenizer to count tokens.
        :param llm_tac_wrapper: Instance of LLMTACWrapper to generate Coq code using an LLM.
        """
        self.state_dim = state_dim
        self.coq_engine = coq_engine
        self.tokenizer = tokenizer
        self.llm_tac_wrapper = llm_tac_wrapper
        self.process_noise_cov = np.eye(self.state_dim) * 0.05  # Adjust as needed

    def __call__(self, state: np.ndarray, action: Tuple[str, ...], current_goal: str) -> Tuple[np.ndarray, int]:
        """
        Predicts the next state based on the current state and action by generating and executing Coq code using the LLM.

        :param state: Current state as a numpy array.
        :param action: Action taken as a tuple (e.g., ('ApplyTactic', 'intro')).
        :param current_goal: The current proof goal as a string.
        :return: Tuple of predicted next state and tokens used.
        """
        print(f"TransitionModel called with state: {state}, type: {type(state)}, shape: {state.shape}")
        if not isinstance(state, np.ndarray):
            raise TypeError("State must be a numpy array.")

        if state.shape[0] != self.state_dim:
            raise ValueError(f"State dimension must be {self.state_dim}.")

        # Generate Coq code using LLMTACWrapper
        coq_code, tokens_used = self.generate_coq_code(action, current_goal)

        if not coq_code:
            # If no Coq code was generated, return the state unchanged
            print("No Coq code generated. Action may not be applicable.")
            return state.copy(), tokens_used

        # Execute Coq code using CoqEngine
        results, metadata = self.coq_engine.forward(coq_code)

        # Process the results to update the state
        if metadata.get('status') == 'success':
            # Update state based on successful proof step
            updated_state = self.update_state(state, success=True)
        else:
            # Proof step failed; handle accordingly
            print(f"Coq execution failed: {metadata.get('message', '')}")
            updated_state = self.update_state(state, success=False)

        # Incorporate process noise
        noise = np.random.multivariate_normal(mean=np.zeros(self.state_dim), cov=self.process_noise_cov)
        updated_state += noise

        return updated_state, tokens_used

    def generate_coq_code(self, action: Tuple[str, ...], current_goal: str) -> Tuple[str, int]:
        action_type = action[0]
        action_param = action[1] if len(action) > 1 else None

        if action_type == 'ApplyTactic':
            tactic = action_param
            prompt = f"""You are a Coq expert assistant.

                        Write a Coq proof applying the tactic '{tactic}' to prove the following goal:

                        {current_goal}

                        Provide only the Coq code, including the lemma statement and proof, without any additional explanation or comments.
                        Do not write anything other than the Coq code.
                        """
        elif action_type == 'ProofStrategy':
            strategy = action_param
            prompt = f"""You are a Coq expert assistant.

                        Apply the proof strategy '{strategy}' to prove the following goal:

                        {current_goal}

                        Provide only the Coq code, including the lemma statement and proof, without any additional explanation or comments.
                        Do not write anything other than the Coq code.
                        """
        else:
            # For 'QueryOntology' or other actions, Coq code generation is not applicable
            return "", 0

        # Generate Coq code using LLMTACWrapper
        coq_code, tokens_used = self.llm_tac_wrapper.generate_coq_code_from_prompt(prompt)

        return coq_code, tokens_used


    def update_state(self, state: np.ndarray, success: bool) -> np.ndarray:
        """
        Updates the state vector based on the success or failure of the proof step.

        :param state: Current state as a numpy array.
        :param success: Boolean indicating if the proof step was successful.
        :return: Updated state as a numpy array.
        """
        updated_state = state.copy()
        if success:
            # Increase state values to represent progress
            updated_state += np.random.uniform(0.1, 0.5, size=self.state_dim)
        else:
            # Decrease state values to represent setback
            updated_state -= np.random.uniform(0.1, 0.3, size=self.state_dim)
        return updated_state
