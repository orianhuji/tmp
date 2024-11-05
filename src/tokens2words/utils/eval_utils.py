import torch
import string
from collections import defaultdict


def is_not_number(s):
    try:
        float(s)  # Try converting the string to a float
        return False  # If conversion is successful, it's a number
    except ValueError:
        return True  # If conversion fails, it's not a number


def get_contexts_ending_with_word(word, dataset):
    result_contexts = []
    word_len = len(word)

    # Iterate over the dataset
    for example in dataset:
        text = example["text"]

        # Find all occurrences of the word in the text
        start = 0
        while True:
            idx = text.find(word, start)
            if idx == -1:
                break

            # Ensure that the word is isolated (not a substring of another word)
            if (idx == 0 or not text[idx - 1].isalnum()) and (
                    idx + word_len == len(text) or not text[idx + word_len].isalnum()):
                # Text ends with the word
                result_contexts.append(text[:idx + word_len].strip())
            start = idx + word_len

    return result_contexts


def get_texts_containing_word(words, dataset):
    result_texts = []
    words_set = set(words)

    # Iterate over the dataset
    for example in dataset:
        if words_set.intersection(set(example["text"].split())):
            result_texts.append(example["text"])

    return result_texts


def get_words_in_dataset_by_token_length(data, tokenizer, remove_breaks=False):
    # dict to map from tokens length (2, 3, 4...) to multi-token words in that length found in the data
    num_tokens2multi_token_words = defaultdict(list)

    custom_punctuation = string.punctuation + "“”"

    def remove_possessive_s(s):
        return s[:-2] if (s.endswith("'s") or s.endswith("’s")) else s

    # Iterate over the dataset
    for example in data:
        text = example["text"]
        if remove_breaks:
            text.replace("--", " ")
        words = text.split()

        for word in words:
            word = word.strip(custom_punctuation)
            word = remove_possessive_s(word)
            tokens = tokenizer.tokenize(word)
            token_count = len(tokens)
            if token_count > 1:
                num_tokens2multi_token_words[token_count].append(word)

    # Remove duplicates in the lists, but keep insertion order
    for k in num_tokens2multi_token_words:
        num_tokens2multi_token_words[k] = list(dict.fromkeys(num_tokens2multi_token_words[k]))

    return num_tokens2multi_token_words


def compute_topk_token_rank(logits, labels, k=1000):
    # Get the top-k predicted logits and their indices
    topk_logits, topk_indices = torch.topk(logits, k, dim=-1)

    # Expand the labels for comparison
    labels_expanded = labels.unsqueeze(-1).expand_as(topk_indices)

    # Check if the label token is within the top-k predictions
    rank_in_topk = (topk_indices == labels_expanded).nonzero(as_tuple=False)

    # Create a rank tensor initialized with k (max rank is k)
    ranks = torch.full(labels.shape, k, dtype=torch.long, device=logits.device)

    # For labels in top-k, set the rank accordingly
    ranks[rank_in_topk[:, 0], rank_in_topk[:, 1]] = rank_in_topk[:, 2] + 1

    return ranks


def count_tokens_in_dataset(dataset, tokenizer, text_column='text'):
    def tokenize_and_count(examples):
        return {'num_tokens': [len(tokenizer(ex).input_ids) for ex in examples[text_column]]}

    tokenized_dataset = dataset.map(tokenize_and_count, batched=True, remove_columns=dataset.column_names)

    total_tokens = sum(tokenized_dataset['num_tokens'])
    return total_tokens


# TODO make clearer what's its use
def flip_tensor(tensor):
    # Find where consecutive zeros end
    zero_mask = (tensor == 0)
    diff = torch.diff(zero_mask.int(), dim=1)
    last_zero_mask = torch.cat([diff, torch.ones(tensor.size(0), 1, dtype=diff.dtype).to(tensor.device)], dim=1) == -1

    # Create the output
    output = 1 - tensor
    output[zero_mask & ~last_zero_mask] = 0
    return output




