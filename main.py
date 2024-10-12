# main.py

import os
from dotenv import load_dotenv
from agents.agent import LLMAIXITACAgent
from agents.environment_hypothesis import EnvironmentHypothesis
from models.world_model import ProbabilisticWorldModel
from models.reward_function import TokenAwareRewardFunction
from reasoning.llm_tac_wrapper import LLMTACWrapper
from reasoning.coq_engine import CoqEngine
from utils.tokenizer import Tokenizer
from utils.observation_encoder import ObservationEncoder
from knowledge.ontology import Ontology
from environments.ontology_navigation import OntologyNavigationEnv
from models.transition_model import TransitionModel  # Import the TransitionModel class
import numpy as np
from typing import Tuple

def main():
    load_dotenv()
    tokenizer = Tokenizer()
    coq_engine = CoqEngine()
    observation_encoder = ObservationEncoder(input_dim=2, embedding_dim=32)
    observation_encoder.eval()

    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OpenAI API key not found. Please set 'OPENAI_API_KEY' in your environment variables.")
    llm_tac_wrapper = LLMTACWrapper(api_key=api_key, model_name='gpt-4o', tokenizer=tokenizer)

    transition_model_instance = TransitionModel(
        state_dim=2,
        coq_engine=coq_engine,
        tokenizer=tokenizer,
        llm_tac_wrapper=llm_tac_wrapper
    )


    # Load ontology JSON
    ontology_json = '''
    {
      "Universes": [
        {
          "id": "U0",
          "Type": "U0 : Type",
          "Description": "The base universe. Each universe U_i : U_{i+1}."
        },
        {
          "id": "U1",
          "Type": "U1 : Type",
          "Description": "The next universe level, containing U0 and larger types.",
          "Contains": ["U0"]
        }
      ],
      "Types": [
        {
          "id": "Type",
          "Type": "Type : U_i → Type",
          "Description": "Dependent type representing types within a universe."
        },
        {
          "id": "Term",
          "Type": "Term : (A : Type) → Type",
          "Description": "Dependent type representing terms of a given type A."
        }
      ],
      "Theorems": [
        {
          "id": "Univalence",
          "Proposition": "∀ (A B : U_i). IsEquiv(eq_to_path(A, B)) → (A =_{U_i} B)",
          "ProofTerm": "ua",
          "ProvableIn": "HoTT + Univalence Axiom",
          "Description": "Univalence axiom stating that equivalences correspond to equalities between types.",
          "Complexity": "Fundamental"
        },
        {
          "id": "CircleHIType",
          "Proposition": "Circle is a higher inductive type.",
          "ProofTerm": "circle_hitype_proof",
          "ProvableIn": "HoTT",
          "Description": "The circle is defined as a higher inductive type with specific constructors.",
          "Complexity": "Intermediate"
        }
      ],
      "HITypeDefinitions": [
        {
          "id": "Circle",
          "Type": "HIType",
          "Constructors": [
            {
              "point": "base : Term(Circle)"
            },
            {
              "path": "loop : Path(Circle, base, base)"
            }
          ],
          "Description": "Higher inductive type representing the circle."
        }
      ],
      "MetaMathematics": [
        {
          "id": "GodelIncompleteness1",
          "Proposition": "∀ S : FormSys. Consistent(S) ∧ SufficientlyExpressive(S) ⇒ ∃ G : Prop. ¬ ProofIn(G, S) ∧ ¬ ProofIn(¬G, S)",
          "ProofTerm": "godel_incompleteness1_proof",
          "ProvableIn": "Meta-HoTT",
          "Description": "Gödel's First Incompleteness Theorem.",
          "Complexity": "Complex"
        }
      ]
    }
    '''

    ontology = Ontology(ontology_json)

    # Initialize environment with a provable goal
    env = OntologyNavigationEnv(
        ontology=ontology,
        goal_node_id='GodelIncompleteness1', 
        coq_engine=coq_engine
    )

    transition_model_instance = TransitionModel(
        state_dim=2,  
        coq_engine=coq_engine,
        tokenizer=tokenizer, 
        llm_tac_wrapper=llm_tac_wrapper
    )

    # Define observation_model function as before
    def observation_model(observation: np.ndarray, new_state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Defines the observation model.

        :param observation: The observation received as a numpy array.
        :param new_state: The new state as a numpy array.
        :return: A tuple (H, R) where H is the observation matrix and R is the observation noise covariance.
        """
        H = np.eye(len(new_state))  # Identity matrix for simplicity
        R = np.eye(len(new_state)) * 0.1  # Observation noise
        return H, R

    # Create environment hypotheses
    h1 = EnvironmentHypothesis(
        name="H1",
        transition_model=transition_model_instance,  # Use the TransitionModel instance
        observation_model=observation_model,
        process_noise_cov=transition_model_instance.process_noise_cov,
        observation_noise_cov=np.eye(2) * 0.1,
        is_dynamic=False
    )
    initial_state = np.array([0.0, 0.0])  # Example initial state
    h1.initialize_belief_state(initial_state=initial_state)

    h2 = EnvironmentHypothesis(
        name="H2",
        transition_model=transition_model_instance,  # Use the TransitionModel instance
        observation_model=observation_model,
        process_noise_cov=transition_model_instance.process_noise_cov,
        observation_noise_cov=np.eye(2) * 0.1,
        is_dynamic=False
    )
    h2.initialize_belief_state(initial_state=initial_state.copy())

    # Initialize reward function
    reward_function = TokenAwareRewardFunction(
        base_reward_weight=1.0,
        token_penalty_weight=0.05,
        exploration_reward=10.0,
        reasoning_quality_weight=0.1,
        token_budget=1000
    )


    # Initialize world model with goal position and threshold
    goal_position = np.array([5.0, 5.0])  # Example goal position; adjust as needed
    world_model = ProbabilisticWorldModel(
        transition_model=transition_model_instance,  # Pass the TransitionModel instance
        observation_model=observation_model,
        goal_position=goal_position,
        goal_threshold=1.0
    )

    # Generate list of actions based on the new action space
    tactics = ['intro', 'induction', 'apply', 'split', 'destruct', 'contradiction']
    # Remove 'TryLemma' if it's not defined, or ensure it's handled appropriately
    queries = [('FindRelatedType', type_name) for type_name in ['Type', 'Term', 'CircleHIType']]
    strategies = ['SimplifyGoal']  # Removed 'TryLemma' to avoid undefined tactic errors

    actions = [('ApplyTactic', tactic) for tactic in tactics] + \
              [('QueryOntology', *query) for query in queries] + \
              [('ProofStrategy', strategy) for strategy in strategies]

    # Initialize agent with the new actions and environment
    agent = LLMAIXITACAgent(
        hypotheses=[h1, h2],
        actions=actions,
        tokenizer=tokenizer,
        llm_tac_wrapper=llm_tac_wrapper,
        world_model=world_model,
        reward_function=reward_function,
        environment=env,
        observation_encoder=observation_encoder,
        ontology=ontology,
        coq_engine=coq_engine
    )
    agent.set_token_budget(1000)

    # Reset environment
    observation = env.reset()
    env.render()
    goal = '¬(𝑃∨𝑄)⇔¬𝑃∧¬𝑄'  

    # Interaction loop
    while not env.done and agent.token_budget > 200:
        # Select action
        action = agent.select_action(goal)
        if action is None:
            print("No action selected. Terminating.")
            break
        # Execute action in the environment
        observation, reward, done, info = env.step(action)
        # Update agent based on the result
        print("Action Executed:", action)
        agent.update(action, observation, reward, info)
        # Render environment state
        env.render()
        # Display action and reward
        print(f"Action: {action}, Observation: {observation}, Reward: {reward}, Remaining Tokens: {agent.token_budget}\n")


    if env.done:
        print("Agent has successfully completed the proof!")
    else:
        print("Agent failed to complete the proof within the token budget.")


if __name__ == "__main__":
    main()
