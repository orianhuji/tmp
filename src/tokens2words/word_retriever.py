import torch
from tqdm import tqdm
from abc import ABC, abstractmethod

from utils.enums import MultiTokenKind, RetrievalTechniques
from processor import RetrievalProcessor
from utils.logit_lens import ReverseLogitLens


class WordRetrieverBase(ABC):
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    @abstractmethod
    def retrieve_word(self, new_vector, layer_idx=0, num_tokens_to_generate=3):
        pass


class PatchscopesRetriever(WordRetrieverBase):
    def __init__(self, model, tokenizer):
        super().__init__(model, tokenizer)
        self.source_input_ids = self.generate_states(tokenizer)
        source_tokens = tokenizer.tokenize(self.generate_prompt())
        self.start_index = source_tokens.index(')')
        self.end_index = source_tokens.index('2')

    def generate_states(self, tokenizer, word='Wakanda', with_prompt=True):
        prompt = self.generate_prompt() if with_prompt else word
        input_ids = tokenizer.encode(prompt, return_tensors='pt')
        return input_ids

    def generate_prompt(self, word='Wakanda'):
        return f"Next is the same word twice: 1){word} 2)"

    def retrieve_word(self, new_vector, layer_idx=0, num_tokens_to_generate=3):
        input_ids = self.source_input_ids
        input_embeddings = self.model.get_input_embeddings()(input_ids)
        input_embeddings = torch.cat([
            torch.cat([input_embeddings[:, :self.start_index + 1, :], new_vector.unsqueeze(0)], dim=1),
            input_embeddings[:, self.end_index:, :]
        ], dim=1)
        with torch.no_grad():
            output = self.model.generate(inputs_embeds=input_embeddings, max_new_tokens=num_tokens_to_generate)
            repeated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return repeated_text

class ReverseLogitLensRetriever(WordRetrieverBase):
    def __init__(self, model, tokenizer, device='cuda', dtype=torch.float16):
        super().__init__(model, tokenizer)
        self.reverse_logit_lens = ReverseLogitLens.from_model(model).to(device).to(dtype)

    def retrieve_word(self, new_vector, layer_idx=0, num_tokens_to_generate=3):
        result = self.reverse_logit_lens(new_vector, layer_idx)
        token = self.tokenizer.decode(torch.argmax(result, dim=-1).item())
        return token


class AnalysisWordRetriever:
    def __init__(self, model, tokenizer, multi_token_kind, num_tokens_to_generate=1, add_context=True,
                 model_name='LLaMa-2B', device='cuda', dataset=None):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.multi_token_kind = multi_token_kind
        self.num_tokens_to_generate = num_tokens_to_generate
        self.add_context = add_context
        self.model_name = model_name
        self.device = device
        self.dataset = dataset
        self.retriever = self._initialize_retriever()
        self.RetrievalTechniques = (RetrievalTechniques.Patchscopes if self.multi_token_kind == MultiTokenKind.Natural
                                    else RetrievalTechniques.ReverseLogitLens)
        self.whitespace_token = 'Ġ' if model_name in ['gemma-2-9b', 'pythia-6.9b', 'LLaMA3-8B', 'Yi-6B'] else '▁'
        self.processor = RetrievalProcessor(self.model, self.tokenizer, self.multi_token_kind,
                                            self.num_tokens_to_generate, self.add_context, self.model_name,
                                            self.whitespace_token)

    def _initialize_retriever(self):
        if self.multi_token_kind == MultiTokenKind.Natural:
            return PatchscopesRetriever(self.model, self.tokenizer)
        else:
            return ReverseLogitLensRetriever(self.model, self.tokenizer)

    def retrieving_words_in_dataset(self, number_of_corpora_to_retrieve=2, max_length=1000):
        self.model.eval()
        results = []

        for text in tqdm(self.dataset['train']['text'][:number_of_corpora_to_retrieve], self.model_name):
            tokenized_input = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=max_length).to(
                self.device)
            tokens = tokenized_input.input_ids[0]
            print(f'Processing text: {text}')
            i = 5
            while i < len(tokens):
                if self.multi_token_kind == MultiTokenKind.Natural:
                    j, word_tokens, word, context, tokenized_combined_text, combined_text, original_word = self.processor.get_next_word(
                        tokens, i, device=self.device)
                elif self.multi_token_kind == MultiTokenKind.Typo:
                    j, word_tokens, word, context, tokenized_combined_text, combined_text, original_word = self.processor.get_next_full_word_typo(
                        tokens, i, device=self.device)
                else:
                    j, word_tokens, word, context, tokenized_combined_text, combined_text, original_word = self.processor.get_next_full_word_separated(
                        tokens, i, device=self.device)

                if len(word_tokens) > 1:
                    with torch.no_grad():
                        outputs = self.model(**tokenized_combined_text, output_hidden_states=True)

                    hidden_states = outputs.hidden_states
                    for layer_idx, hidden_state in enumerate(hidden_states):
                        postfix_hidden_state = hidden_states[layer_idx][0, -1, :].unsqueeze(0)
                        retrieved_word_str = self.retriever.retrieve_word(
                            postfix_hidden_state, num_tokens_to_generate=len(word_tokens), layer_idx=layer_idx)
                        results.append({
                            'text': combined_text,
                            'original_word': original_word,
                            'word': word,
                            'word_tokens': self.tokenizer.convert_ids_to_tokens(word_tokens),
                            'num_tokens': len(word_tokens),
                            'layer': layer_idx,
                            'retrieved_word_str': retrieved_word_str,
                            'context': "With Context" if self.add_context else "Without Context"
                        })
                else:
                    i = j
        return results