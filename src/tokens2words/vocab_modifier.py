from tqdm import tqdm
from abc import ABC, abstractmethod
from typing import Iterable, Union, List, Dict
from transformers import PreTrainedModel, PreTrainedTokenizer
import numpy as np
import pandas as pd
import re
import torch
from torch import nn
from collections import defaultdict
from typing import DefaultDict

from .representation_translator import RepresentationTranslators
from .word_retriever import PatchscopesRetriever


class VocabularyModifier(ABC):
    """
    Abstract class for...  # TODO
    """

    def __init__(
            self,
            model: PreTrainedModel,
            tokenizer: PreTrainedTokenizer,
            **kwargs
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.orig_vocab_size = len(tokenizer)
        self.new_tokens_idx: List[int] = list()
        self.new_words: List[str] = list()
        self.failed_words: List[str] = list()
        self.entries_cache = {"embedding": list(), "lm_head": list()}

    @abstractmethod
    def compute_entries_for_word(
            self, word: str
    ) -> (torch.Tensor, torch.Tensor):
        """
        Computes the entries in the embedding and LM head matrices for a given word.

        Args:
            word (str): The word to add to the vocabulary

        Returns:
            embedding entry (torch.Tensor): The transformed representation in embedding space.
            unembedidng entry (torch.Tensor): The transformed representation in LM head space.
        """
        pass

    def add_word_to_vocab(
            self, word: str, finalize: bool = True
    ) -> None:
        embedding_entry, lm_head_entry = self.compute_entries_for_word(word)
        if not embedding_entry or not lm_head_entry:
            # failed to compute new entries for word
            self.failed_words.append(word)
            return

        # Add new tokens to the tokenizer
        num_added_tokens = self.tokenizer.add_tokens([word])

        if num_added_tokens > 0:  # don't add word if it already exists
            self.new_words.append(word)
            if finalize:
                self.model.resize_token_embeddings(len(self.tokenizer) + num_added_tokens)
                new_token_idx = len(self.tokenizer) - 1
                self.new_tokens_idx.append(new_token_idx)

                with torch.no_grad():
                    self.model.get_input_embeddings().weight[new_token_idx] = embedding_entry
                    self.model.lm_head.weight[new_token_idx] = lm_head_entry
            else:
                self.entries_cache["embedding"].append(embedding_entry)
                self.entries_cache["lm_head"].append(lm_head_entry)

    def add_words_to_vocab(
            self, words: Iterable[str]
    ) -> None:
        for word in tqdm(words, total=len(words), desc="Adding words to vocabulary...", unit="word"):
            self.add_word_to_vocab(word, finalize=False)

        self.model.resize_token_embeddings(len(self.tokenizer) + len(self.new_words))

        for new_token_idx, word, embedding_entry, lm_head_entry in zip(
                range(self.orig_vocab_size, len(self.tokenizer)),
                self.new_words,
                self.entries_cache["embedding"],
                self.entries_cache["lm_head"]
            ):
            self.new_tokens_idx.append(new_token_idx)

            with torch.no_grad():
                self.model.get_input_embeddings().weight[new_token_idx] = embedding_entry
                self.model.lm_head.weight[new_token_idx] = lm_head_entry

        self.entries_cache = {"embedding": list(), "lm_head": list()}


class DetokenizationVocabularyExpander(VocabularyModifier):
    # TODO define patchscopes and its cache in init,
    #  then if word is not present in cache add it to cache.
    #  add func to dump and update cache

    def __init__(
            self,
            model: PreTrainedModel,
            tokenizer: PreTrainedTokenizer,
            patchscopes_retriever: PatchscopesRetriever,
            patchscopes_results: Union[np.ndarray, pd.DataFrame, Dict[str, Dict[int, str]]] = None,
            translators: RepresentationTranslators = None,
            detokenization_decision_rule: str = "first_id_layer",
            **kwargs
    ):
        super().__init__(model, tokenizer, translators, **kwargs)

        self.detokenization_decision_rule = detokenization_decision_rule

        self.patchscopes_retriever = patchscopes_retriever
        self.patchscopes_results = patchscopes_results
        if not patchscopes_results:
            # create dict that maps new words (str) to a list of their patchscopes output per layer
            self.patchscopes_results: DefaultDict[str, List[str]] = defaultdict(list)

        self.translators = translators

    def _decide_detokenization_end_layer(self, word: str, patchscope_results: Iterable[str]):
        patchscope_results = np.array(patchscope_results)

        # Count occurrences of word in each layer's result
        # Use word boundary \b to match whole words only
        pattern = f"\\b{re.escape(word)}\\b"
        counts = np.array([len(re.findall(pattern, s)) for s in patchscope_results])

        if self.detokenization_decision_rule == "first_id_layer":
            return np.argmax(counts > 0)
        elif self.detokenization_decision_rule == "last_id_layer":
            return len(counts) - np.argmax((counts > 0)[::-1]) - 1
        elif self.detokenization_decision_rule == "first_layer_with_2_repeats":
            return np.argmax(counts >= 2)
        elif self.detokenization_decision_rule == "last_layer_with_2_repeats":
            return len(counts) - np.argmax((counts >= 2)[::-1]) - 1

        return None

    def compute_entries_for_word(
            self, word: str
    ) -> (torch.Tensor, torch.Tensor):
        """

        Args:
            word (str):
                ...
        """
        if word not in self.patchscope_results:
            patchscopes_description_by_layers, last_token_hidden_states = \
                self.patchscopes_retriever.get_hidden_states_and_retrieve_word(word)
        else:
            patchscopes_description_by_layers = self.patchscopes_results[word]
            last_token_hidden_states = self.patchscopes_retriever.extract_hidden_states(word)

        target_layer = self._decide_detokenization_end_layer(word, patchscopes_description_by_layers)

        if target_layer is None:  # detokenization did not occur
            return None, None

        target_as_embedding = target_as_lm_head = last_token_hidden_states[target_layer]

        target_as_embedding = self.translators.to_embedding(target_as_embedding, target_layer).to(
            self.model.get_input_embeddings().weight.dtype)
        target_as_lm_head = self.translators.to_lm_head(target_as_lm_head, target_layer).to(
            self.model.get_input_embeddings().weight.dtype)

        return target_as_embedding, target_as_lm_head

    def get_patchscopes_results(self):
        return pd.DataFrame.from_records(self.patchscopes_results)


