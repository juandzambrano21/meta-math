# models/transition_model.py

import numpy as np
from typing import Tuple, Dict, Any
from reasoning.coq_engine import CoqEngine
from reasoning.llm_tac_wrapper import LLMTACWrapper
from utils.tokenizer import Tokenizer
import logging
import re
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    

    def __call__(
        self,
        proof_state: Dict[str, Any],
        action: Tuple[str, ...],
        current_goal: str
    ) -> Tuple[Dict[str, Any], int]:
        """
        Predicts the next state based on the current proof state and action by generating and executing Coq code using the LLM.

        :param proof_state: Current proof state as a dictionary.
        :param action: Action taken as a tuple (e.g., ('ApplyTactic', 'intro')).
        :param current_goal: The current proof goal as a string.
        :return: Tuple of predicted next proof state and tokens used.
        """
        print(f"TransitionModel called with proof_state: {proof_state}, type: {type(proof_state)}")
        if not isinstance(proof_state, dict):
            raise TypeError("proof_state must be a dictionary.")

        # Generate Coq code using LLMTACWrapper
        coq_code, tokens_used = self.generate_coq_code(action, current_goal)

        if not coq_code:
            # If no Coq code was generated, return the proof_state unchanged
            print("No Coq code generated. Action may not be applicable.")
            return proof_state.copy(), tokens_used

        # Execute Coq code using CoqEngine
        results, metadata = self.coq_engine.forward(coq_code)

        # Process the results to update the proof_state
        if metadata.get('status') == 'success':
            # Update proof_state based on successful proof step
            updated_proof_state = self.update_proof_state(proof_state, success=True)
        else:
            # Proof step failed; handle accordingly
            print(f"Coq execution failed: {metadata.get('message', '')}")
            updated_proof_state = self.update_proof_state(proof_state, success=False)

        # Incorporate process noise
        noise = np.random.multivariate_normal(mean=np.zeros(self.state_dim), cov=self.process_noise_cov)
        updated_proof_state['mean'] += noise

        return updated_proof_state, tokens_used

    def transition_model_func(
        self,
        proof_state: Dict[str, Any],
        action: Tuple[str, ...],
        current_goal: str
    ) -> Tuple[Dict[str, Any], int]:
        """
        Predicts the next belief state based on the current belief state and action by generating and executing Coq code.

        :param proof_state: Current belief state as a dictionary with 'mean', 'cov', and 'tactics_applied'.
        :param action: Action taken as a tuple (e.g., ('ApplyTactic', 'intro')).
        :param current_goal: The current proof goal as a string.
        :return: Tuple of updated belief state and tokens used.
        """
        logger.info(f"TransitionModel called with proof_state: {proof_state}, action: {action}, current_goal: {current_goal}")

        # Validate proof_state structure
        if not isinstance(proof_state, dict):
            raise TypeError("proof_state must be a dictionary.")
        required_keys = {'mean', 'cov', 'tactics_applied'}
        if not required_keys.issubset(proof_state.keys()):
            raise ValueError(f"proof_state must contain keys: {required_keys}")

        state_mean = proof_state['mean']
        state_cov = proof_state['cov']
        tactics_applied = proof_state['tactics_applied']

        # Generate Coq code using LLMTACWrapper
        coq_code, tokens_used = self.generate_coq_code(action, current_goal)
        logger.info(f"Generated Coq code:2 {coq_code}")
        logger.info(f"Tokens used for Coq code generation: {tokens_used}")

        if not coq_code.strip():
            # If no Coq code was generated, return the state unchanged
            logger.warning("No Coq code generated. Action may not be applicable.")
            return proof_state.copy(), tokens_used

        # Execute Coq code using CoqEngine
        try:
            results, metadata = self.coq_engine.forward(coq_code)
            logger.info(f"Coq execution results: {results}")
            logger.info(f"Coq execution metadata: {metadata}")
        except Exception as e:
            logger.error(f"Error during Coq execution: {e}")
            # Optionally, handle the error by updating the state accordingly
            return self.update_state(proof_state, success=False), tokens_used

        # Process the results to update the belief state
        if metadata.get('status') == 'success':
            # Update state based on successful proof step
            updated_state = self.update_state(proof_state, success=True)
            logger.info("Coq execution successful. Updated belief state.")
        else:
            # Proof step failed; handle accordingly
            logger.warning(f"Coq execution failed: {metadata.get('message', 'No message provided.')}")
            updated_state = self.update_state(proof_state, success=False)

        # Incorporate process noise
        noise = np.random.multivariate_normal(mean=np.zeros(self.state_dim), cov=self.process_noise_cov)
        updated_state['mean'] += noise
        updated_state['cov'] += self.process_noise_cov  # Optionally update covariance

        logger.info(f"Updated belief state after transition: {updated_state}")

        return updated_state, tokens_used

    def generate_coq_code(self, action: Tuple[str, ...], current_goal: str) -> Tuple[str, int]:
        """
        Generates Coq code based on the action and current goal using the LLMTACWrapper.

        :param action: Action taken as a tuple (e.g., ('ApplyTactic', 'intro')).
        :param current_goal: The current proof goal as a string.
        :return: Tuple of Coq code and tokens used.
        """
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
        elif action_type == 'QueryOntology':
            # For 'QueryOntology' actions, Coq code generation is not applicable
            logger.warning(f"Action '{action}' does not require Coq code generation.")
            return "", 0
        else:
            # Unhandled action types
            logger.error(f"Unhandled action type: {action_type}")
            return "", 0

        # Generate Coq code using LLMTACWrapper
        try:
            coq_code, tokens_used = self.llm_tac_wrapper.generate_coq_code_from_prompt(prompt)
            coq_code = re.sub(r'```|coq', '', coq_code)
        except Exception as e:
            logger.error(f"Error during Coq code generation: {e}")
            coq_code = ""
            tokens_used = 0

        return coq_code, tokens_used

    def update_state(self, proof_state: Dict[str, Any], success: bool) -> Dict[str, Any]:
        """
        Updates the belief state based on the success or failure of the proof step.

        :param proof_state: Current belief state as a dictionary with 'mean', 'cov', and 'tactics_applied'.
        :param success: Boolean indicating if the proof step was successful.
        :return: Updated belief state as a dictionary.
        """
        updated_proof_state = proof_state.copy()
        if success:
            # Increase state values to represent progress
            progress = np.random.uniform(0.1, 0.5, size=self.state_dim)
            updated_proof_state['mean'] += progress
            logger.debug(f"State progressed by: {progress}")
        else:
            # Decrease state values to represent setback
            setback = np.random.uniform(0.1, 0.3, size=self.state_dim)
            updated_proof_state['mean'] -= setback
            logger.debug(f"State set back by: {setback}")
        
        # Optionally, you can adjust the covariance here if needed
        # For example, increasing uncertainty on failure
        if not success:
            updated_proof_state['cov'] += np.eye(self.state_dim) * 0.1  # Example adjustment

        return updated_proof_state
