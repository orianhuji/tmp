from tqdm import tqdm
from typing import Iterable, List
from transformers import PreTrainedModel, PreTrainedTokenizer
import torch
from torch import nn


def extract_vocab_hidden_states(
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        tokens_ids_to_extract: Iterable[int] = None,
        prompt: str = "{target}",
        batch_size: int = 128,
        layers_to_extract: List[int] = None,
) -> torch.Tensor:

    all_hidden_states = {layer: [] for layer in layers_to_extract}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    if layers_to_extract is None:
        layers_to_extract = list(range(1, model.config.num_hidden_layers + 1))  # extract all but initial embeddings

    tokens_ids_to_extract = tokens_ids_to_extract if tokens_ids_to_extract is not None else range(tokenizer.vocab_size)
    tokens_to_extract = [tokenizer.decode(tok_id) for tok_id in tokens_ids_to_extract]

    with torch.no_grad():
        for i in tqdm(range(0, len(tokens_to_extract), batch_size), desc="Extracting hidden states", unit="batch"):
            prompts = [prompt.format(target=target) for target in tokens_to_extract[i:i+batch_size]]
            input_ids = tokenizer.encode(prompts, return_tensors="pt")
            outputs = model(input_ids.to(device), output_hidden_states=True)
            for layer in layers_to_extract:
                hidden_states = outputs.hidden_states[layer]
                all_hidden_states[layer].append(hidden_states[:, -1, :].detach().cpu())

    for layer in all_hidden_states:
        all_hidden_states[layer] = torch.concat(all_hidden_states[layer], dim=0)

    return all_hidden_states


def get_vocab_tokens(tokenizer: PreTrainedTokenizer, min_word_len: int = None):
    vocab_size = tokenizer.vocab_size
    tokens = list(range(vocab_size))
    if min_word_len:
        tokens_str = [tokenizer.decode(i) for i in tokens]
        tokens_len = [len(x) for x in tokens_str]
        tokens = [tok for tok, tok_len in zip(tokens, tokens_len) if tok_len >= min_word_len]
    return tokens


def learn_linear_map(X: torch.Tensor, Y: torch.Tensor):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    W = torch.linalg.lstsq(X.to(device), Y.to(device)).solution.cpu()
    linear_map = nn.Linear(X.size(1), Y.size(1), bias=False)
    linear_map.weight.data = W.T
    return linear_map

