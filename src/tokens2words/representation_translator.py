import os
from tqdm import tqdm
from abc import ABC, abstractmethod
from typing import Iterable, Union, List
from transformers import PreTrainedModel, PreTrainedTokenizer
import torch
from torch import nn
from torch.utils.data import Dataset, IterableDataset

from .utils.model_utils import learn_linear_map, extract_vocab_hidden_states


class RepresentationTranslators(ABC, nn.Module):
    """
    Abstract class for mapping intermediate model representations to the model's embedding and unembedding spaces.
    """

    def __init__(self):
        super(RepresentationTranslators, self).__init__()
        self.embedding_maps = nn.ModuleDict()
        self.lm_head_maps = nn.ModuleDict()

    @abstractmethod
    def fit_on_dataset(
            self,
            model: PreTrainedModel,
            tokenizer: PreTrainedTokenizer,
            dataset: Union[List[str], Dataset, IterableDataset, "datasets.Dataset"],
    ) -> None:
        """
        Learns transformations that map representations from every layer to the embedding/unembedding spaces,
        by fitting a transformation from intermediate model representations of vocabulary tokens, as computed in
        the dataset's examples, to the corresponding token rows in the embedding and unembedding matrices.

        Args:
            model (PreTrainedModel):
                Model to extract embeddings and intermediate representations from.
            tokenizer (PreTrainedTokenizer):
                Tokenizer to use when extracting embeddings.
            dataset (Union[List[str], Dataset, IterableDataset, "datasets.Dataset"]):
                Data to train transformations on.
        """
        pass

    @abstractmethod
    def fit_on_tokens(
            self,
            model: PreTrainedModel,
            tokenizer: PreTrainedTokenizer,
            token_ids: Iterable[int] = None,
            prompt: str = "{target}",
    ) -> None:
        """
        Learns transformations that map representations from every layer to the embedding/unembedding spaces,
        by fitting a transformation from intermediate model representations of single tokens from the vocabulary,
        as extracted using the given prompt, to the respective rows in the embedding and unembedding matrices.

        Args:
            model (PreTrainedModel):
                Model to extract embeddings and intermediate representations from.
            tokenizer (PreTrainedTokenizer):
                Tokenizer to use when extracting embeddings.
            tokens (Iterable[int]):
                The list of token_ids to use as training data.
                If not given, will train over the entire vocabulary.
            prompt (str):
                The prompt to use when extracting token representations, where the string "{target}"
                will be replaced with the target token.
        """
        pass

    @abstractmethod
    def to_embedding(self, representations: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Transforms the given representations to the embedding space.

        Args:
            representations (torch.Tensor): The intermediate representations to transform.

        Returns:
            torch.Tensor: The transformed representations in embedding space.
        """
        pass

    @abstractmethod
    def to_lm_head(self, representations: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Transforms the given representations to the unembedding space.

        Args:
            representations (torch.Tensor): The intermediate representations to transform.

        Returns:
            torch.Tensor: The transformed representations in unembedding space.
        """
        pass


class LinearRepresentationTranslators(RepresentationTranslators):
    """
    Transforms intermediate model representations to the embedding and unembedding spaces
    using linear maps.
    """

    def fit_on_tokens(
            self,
            model: PreTrainedModel,
            tokenizer: PreTrainedTokenizer,
            token_ids: Iterable[int] = None,
            prompt: str = "{target}",
            batch_size: int = 128,
            layer_batch_size: int = 8,
    ) -> None:
        """
        Learns transformations that map representations from every layer to the embedding/unembedding spaces,
        by fitting a transformation from intermediate model representations of single tokens from the vocabulary,
        as extracted using the given prompt, to the respective rows in the embedding and unembedding matrices.

        Args:
            model (PreTrainedModel):
                Model to extract embeddings and intermediate representations from.
            tokenizer (PreTrainedTokenizer):
                Tokenizer to use when extracting embeddings.
            token_ids (Iterable[int]):
                The list of token_ids to use as training data.
                If not given, will train over the entire vocabulary.
            prompt (str):
                The prompt to use when extracting token representations, where the string "{target}"
                will be replaced with the target token.
            batch_size (int):
                The prompt to use when extracting token representations, where the string "{target}"
                will be replaced with the target token.
            layer_batch_size (int):
                The prompt to use when extracting token representations, where the string "{target}"
                will be replaced with the target token.
        """

        input_embeddings = model.get_input_embeddings().weight.detach().cpu()
        lm_head_weights = model.lm_head.weight.detach().cpu()
        # some models have embeddings for special tokens that aren't included in the "regular" vocabulary
        input_embeddings = input_embeddings[:tokenizer.vocab_size]
        lm_head_weights = lm_head_weights[:tokenizer.vocab_size]

        if token_ids is not None and len(token_ids) != tokenizer.vocab_size:
            input_embeddings = input_embeddings[token_ids]
            lm_head_weights = lm_head_weights[token_ids]

        n_layers = model.config.num_hidden_layers
        layer_batch_size = n_layers if layer_batch_size is None else layer_batch_size
        for start_layer_i in tqdm(range(1, n_layers+1, layer_batch_size), desc="Fitting maps to embedding and unembedding spaces", unit="Layer batch"):
            layers_to_learn = None if layer_batch_size is None else \
                list(range(start_layer_i, min(start_layer_i + layer_batch_size, n_layers+1)))
            all_hidden_states = extract_vocab_hidden_states(model, tokenizer, token_ids, prompt, batch_size, layers_to_learn)
            for layer in tqdm(layers_to_learn, total=len(layers_to_learn), unit="layers", desc="Fitting maps..."):
                hidden_states = all_hidden_states[layer]
                self.embedding_maps[str(layer)] = learn_linear_map(hidden_states, input_embeddings)
                self.lm_head_maps[str(layer)] = learn_linear_map(hidden_states, lm_head_weights)

                all_hidden_states[layer] = None

    def fit_on_dataset(
            self,
            model: PreTrainedModel,
            tokenizer: PreTrainedTokenizer,
            dataset: Union[List[str], Dataset, IterableDataset, "datasets.Dataset"],
    ) -> None:
        """
        Learns transformations that map representations from every layer to the embedding/unembedding spaces,
        by fitting a transformation from intermediate model representations of vocabulary tokens, as computed in
        the dataset's examples, to the corresponding token rows in the embedding and unembedding matrices.

        Args:
            model (PreTrainedModel):
                Model to extract embeddings and intermediate representations from.
            tokenizer (PreTrainedTokenizer):
                Tokenizer to use when extracting embeddings.
            dataset (Union[List[str], Dataset, IterableDataset, "datasets.Dataset"]):
                Data to train transformations on.
        """
        raise NotImplementedError("Fine-tuning linear translators on a dataset is not implemented.")

    def to_embedding(self, representations: torch.Tensor, layer_index: int, **kwargs) -> torch.Tensor:
        """
        Transforms the given representations to the embedding space.

        Args:
            representations (torch.Tensor): The intermediate representations to transform.
            layer_index (int): The index of the model layer the representations were extracted from.

        Returns:
            torch.Tensor: The transformed representations in embedding space.
        """
        if self.embedding_maps is None or str(layer_index) not in self.embedding_maps:
            raise ValueError("The mapping has not been trained yet. Call fit first.")
        if not isinstance(representations, torch.Tensor):
            raise TypeError("Representations must be torch.Tensor.")

        return self.lm_head_maps[str(layer_index)](representations)

    def to_lm_head(self, representations: torch.Tensor, layer_index: int, **kwargs) -> torch.Tensor:
        """
        Transforms the given representations to the unembedding space.

        Args:
            representations (torch.Tensor): The intermediate representations to transform.
            layer_index (int): The index of the model layer the representations were extracted from.

        Returns:
            torch.Tensor: The transformed representations in unembedding space.
        """
        if self.lm_head_maps is None or str(layer_index) not in self.lm_head_maps:
            raise ValueError("The mapping has not been trained yet. Call fit first.")
        if not isinstance(representations, torch.Tensor):
            raise TypeError("Representations must be torch.Tensor.")

        return self.lm_head_maps[str(layer_index)](representations)
