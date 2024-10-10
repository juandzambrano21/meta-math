from transformers import GPT2TokenizerFast

class Tokenizer:
    def __init__(self, model_name='gpt2'):
        self.tokenizer = GPT2TokenizerFast.from_pretrained(model_name)

    def tokenize(self, text):
        return self.tokenizer.tokenize(text)

    def count_tokens(self, text):
        return len(self.tokenizer.encode(text))