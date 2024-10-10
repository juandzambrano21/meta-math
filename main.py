import os
from dotenv import load_dotenv
from environments.continuous_navigation import ContinuousNavigationEnv
from agents.agent import LLMAIXITACAgent
from agents.environment_hypothesis import EnvironmentHypothesis
from models.world_model import ProbabilisticWorldModel
from models.reward_function import TokenAwareRewardFunction
from reasoning.llm_tac_wrapper import LLMTACWrapper
from reasoning.coq_engine import CoqEngine
from scipy.stats import multivariate_normal
from utils.tokenizer import Tokenizer
from utils.observation_encoder import ObservationEncoder
from knowledge.ontology import Ontology
from typing import Tuple 

import numpy as np

def main():
    # Initialize tokenizer
    tokenizer = Tokenizer()

    # Initialize CoqEngine
    coq_engine = CoqEngine()

    # Initialize observation encoder
    observation_encoder = ObservationEncoder(input_dim=2, embedding_dim=32)
    observation_encoder.eval() 

    # Initialize environment
    obstacles = [
        {'position': np.array([15.0, 15.0]), 'radius': 2.0},
        {'position': np.array([30.0, 30.0]), 'radius': 3.0},
        {'position': np.array([45.0, 10.0]), 'radius': 1.5},
    ]
    env = ContinuousNavigationEnv(
        area_size=(50.0, 50.0),
        goal_position=(45.0, 45.0),
        obstacles=obstacles,
        motion_noise_cov=np.eye(2) * 0.2,
        observation_noise_cov=np.eye(2) * 1.0,
        goal_threshold=1.0,
        ontology=None  # Will set the ontology after initialization
    )

    # Define transition and observation models for hypotheses
    class TransitionModel:
        def __init__(self):
            self.noise_cov = np.eye(2) * 0.2

        def __call__(self, state: np.ndarray, action: Tuple[float, float]) -> np.ndarray:
            return state + action

    class ObservationModel:
        def __init__(self):
            self.H = np.eye(2)  # Observation matrix
            self.R = np.eye(2) * 1.0  # Observation noise covariance

        def prob(self, observation: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> float:
            """
            Computes P(observation | state).

            :param observation: Observation vector.
            :param mean: State mean.
            :param cov: State covariance.
            :return: Probability density.
            """
            rv = multivariate_normal(mean, cov + self.R)
            return rv.pdf(observation)

    observation_model = ObservationModel()
    transition_model = TransitionModel()

    # Create environment hypothesis
    h1 = EnvironmentHypothesis("H1", transition_model, observation_model, is_dynamic=True)
    h1.initialize_belief_state(initial_state=np.array([5.0, 5.0]))

    # Initialize world model
    world_model = ProbabilisticWorldModel(
        transition_model=h1.transition_model,
        observation_model=h1.observation_model
    )
    world_model.goal_position = np.array([45.0, 45.0])
    world_model.goal_threshold = 1.0

    # Initialize reward function
    reward_function = TokenAwareRewardFunction(
        base_reward_weight=1.0,
        token_penalty_weight=0.05,
        exploration_reward=10.0,
        reasoning_quality_weight=0.1,
        token_budget=1000  # Increased token budget
    )

    # Initialize LLM-TAC wrapper
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')
    
    llm_tac_wrapper = LLMTACWrapper(api_key, model_name='gpt-4', tokenizer=tokenizer)

    # Load ontology JSON (corrected and without comments)
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

    # Initialize the ontology
    ontology = Ontology(ontology_json)
    env.ontology = ontology  # Assign the ontology to the environment

    # Initialize agent with ontology and coq_engine
    agent = LLMAIXITACAgent(
        hypotheses=[h1],
        actions=[
            (1.0, 1.0),    # A1: Accelerate diagonally up-right
            (-1.0, 1.0),   # A2: Accelerate diagonally up-left
            (1.0, -1.0),   # A3: Accelerate diagonally down-right
            (-1.0, -1.0),  # A4: Accelerate diagonally down-left
            (0.0, 1.0),    # A5: Accelerate up
            (0.0, -1.0),   # A6: Accelerate down
            (1.0, 0.0),    # A7: Accelerate right
            (-1.0, 0.0),   # A8: Accelerate left
            (0.0, 0.0)     # A9: No acceleration
        ],
        tokenizer=tokenizer,
        llm_tac_wrapper=llm_tac_wrapper,
        world_model=world_model,
        reward_function=reward_function,
        environment=env,
        observation_encoder=observation_encoder,
        ontology=ontology,
        coq_engine=coq_engine  # Pass the initialized CoqEngine
    )
    agent.set_token_budget(1000)  # Ensure the token budget is set correctly

    # Reset environment
    observation = env.reset()
    env.render()
    goal = 'Proof Euclids theorem using HoTT only'

    # Interaction loop
    while not env.done and agent.token_budget > 200:  # Reserve at least 200 tokens for prompts
        try:
            # Select action
            action = agent.select_action(goal)
            # Execute action in the environment
            observation, reward, done, info = env.step(action)
            # Update agent based on the result
            agent.update(action, observation, reward, info)
            # Render environment state
            env.render()
            # Display action and reward
            print(f"Action: {action}, Observation: {observation}, Reward: {reward}, Remaining Tokens: {agent.token_budget}\n")
        except Exception as e:
            print(f"An error occurred during interaction: {e}")
            break

    if env.done:
        print("Agent has successfully reached the goal!")
    else:
        print("Agent failed to reach the goal within the token budget.")

if __name__ == "__main__":
    main()
