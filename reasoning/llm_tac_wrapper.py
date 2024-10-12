# reasoning/llm_tac_wrapper.py

import openai
import time
from collections import deque
from typing import Tuple, Dict
from utils.tokenizer import Tokenizer

class LLMTACWrapper:
    def __init__(self, api_key: str, model_name: str = 'gpt-4o', tokenizer: 'Tokenizer' = None, max_retries: int = 3, backoff_factor: int = 2):
        """
        Initializes the LLMTACWrapper.
        """
        openai.api_key = api_key
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.cached_prompts: Dict[Tuple, Dict[str, any]] = {}
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.rate_limit_queue = deque(maxlen=20)

    def generate_reasoning_trace(self, goal: str, current_state: str, current_node: str, token_budget: int) -> Tuple[str, int]:
        """
        Generates a reasoning trace using the LLM.
        """
        # Create a hashable cache key
        cache_key = (goal, str(current_state), current_node)

        # Check if the prompt is in the cache
        if cache_key in self.cached_prompts:
            cached_response = self.cached_prompts[cache_key]
            return cached_response['reasoning_trace'], cached_response['tokens_used']

        prompt = self._create_prompt(goal, current_state, current_node, token_budget)
        print(prompt)

        prompt_tokens = self.tokenizer.count_tokens(prompt)
        max_completion_tokens = token_budget - prompt_tokens

        if max_completion_tokens <= 200:
            max_completion_tokens = 200

        for attempt in range(self.max_retries):
            try:
                self._handle_rate_limit()

                response = openai.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a helpful AI assistant that understands Homotopy Type Theory and can generate Coq code."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    max_tokens=max_completion_tokens,
                    temperature=0.3,
                    n=1,
                    stop=["\n\n"]
                )

                reasoning_trace = response.choices[0].message.content.strip()
                print(reasoning_trace)
                reasoning_tokens = self.tokenizer.count_tokens(reasoning_trace)
                total_tokens_used = reasoning_tokens + prompt_tokens

                self.cached_prompts[cache_key] = {
                    'reasoning_trace': reasoning_trace,
                    'tokens_used': total_tokens_used
                }

                return reasoning_trace, total_tokens_used

            except openai.RateLimitError as e:
                print(f"Rate limit error: {e}. Retrying...")
                time.sleep(self.backoff_factor ** attempt)
            except Exception as e:
                print(f"OpenAI API error: {e}")
                if attempt == self.max_retries - 1:
                    raise

                time.sleep(self.backoff_factor ** attempt)

        print("Failed to generate reasoning trace after multiple attempts.")
        return "", 0

    def _create_prompt(self, goal: str, current_state: str, current_node: str, token_budget: int) -> str:

        return f"""You are a Coq assistant helping to prove the following goal in Coq syntax:

                {goal}

                Current Proof State: {current_state}

                Available Actions:
                - PredictAction(ApplyTactic('tactic_name'))
                - PredictAction(QueryOntology('query_type', 'query_param'))
                - PredictAction(ProofStrategy('strategy_name'))
                - EvaluateGoal(Sx)

                Instructions:
                - Provide a sequence of reasoning steps using the available actions.
                - Use valid Coq tactics and ensure the goal is correctly formatted.
                - Generate the reasoning steps in the exact format, one per line.

                Provide only the reasoning steps without any additional explanation or text.
                """


    def generate_coq_code_for_tactic(self, tactic: str, current_goal: str) -> Tuple[str, int]:
        """
        Generates Coq code for applying a tactic to the current goal.
        """
        prompt = f"""You are a Coq expert assistant.

        Write a Coq proof applying the tactic '{tactic}' to prove the following goal:

        {current_goal}

        Ensure the goal is correctly formatted in Coq syntax. Provide only the Coq code, including the lemma statement and proof, without any additional explanation.
        *IMPORANT* Do not write anything other than the Coq code.

        """

        # Generate Coq code using LLM
        coq_code, tokens_used = self.generate_coq_code_from_prompt(prompt)

        return coq_code, tokens_used


    def generate_coq_code_from_prompt(self, prompt: str) -> Tuple[str, int]:
        """
        Generates Coq code using the LLM based on the provided prompt.
        """
        if prompt in self.cached_prompts:
            cached_response = self.cached_prompts[prompt]
            return cached_response['coq_code'], cached_response['tokens_used']

        prompt_tokens = self.tokenizer.count_tokens(prompt)
        token_budget = 1500
        max_completion_tokens = token_budget - prompt_tokens

        if max_completion_tokens <= 100:
            max_completion_tokens = 100

        for attempt in range(self.max_retries):
            try:
                self._handle_rate_limit()

                response = openai.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are a Coq expert assistant. You implement proper Coq code verifying mathematical theorems with systematic and rigorous implementations."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=max_completion_tokens,
                    temperature=0.2,
                    n=1,
                    stop=None
                )

                coq_code = response.choices[0].message.content.strip()
                coq_code_tokens = self.tokenizer.count_tokens(coq_code)
                total_tokens_used = coq_code_tokens + prompt_tokens

                self.cached_prompts[prompt] = {
                    'coq_code': coq_code,
                    'tokens_used': total_tokens_used
                }

                return coq_code, total_tokens_used

            except openai.RateLimitError as e:
                print(f"Rate limit error: {e}. Retrying...")
                time.sleep(self.backoff_factor ** attempt)
            except Exception as e:
                print(f"OpenAI API error: {e}")
                if attempt == self.max_retries - 1:
                    raise

                time.sleep(self.backoff_factor ** attempt)

        print("Failed to generate Coq code after multiple attempts.")
        return "", 0

    def _handle_rate_limit(self):
        """Handles rate limiting using a deque."""
        current_time = time.time()
        while self.rate_limit_queue and current_time - self.rate_limit_queue[0] < 1:
            time.sleep(0.1)
            current_time = time.time()
        self.rate_limit_queue.append(current_time)
