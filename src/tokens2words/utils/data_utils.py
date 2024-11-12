import re
from datasets import load_dataset, Dataset
from itertools import chain

LANGUAGES_TO_DECODE_FROM_BYTES = ["he"]


def load_lm_dataset(dataset_name, language="en", split=None):
    """
    Loads a popular pretraining or perplexity evaluation dataset by name and language.

    Args:
        dataset_name (str): The name of the dataset to load. Options include:
            - 'wikitext' (wikitext-2, smaller WikiText dataset)
            - 'wikitext-103' (larger WikiText dataset)
            - 'pg19' (Project Gutenberg dataset for long-context modeling)
            - 'c4' (Common Crawl-based English corpus)
            - 'wiki40b' (Wikipedia dataset in multiple languages)
            - 'mc4' (Multilingual C4 dataset in various languages)
        language (str): Language code for datasets that support multilingual options (e.g., 'en' for English).
                        Defaults to 'en'.

    Returns:
        Dataset: Loaded Hugging Face dataset.
    """
    if dataset_name.lower() == 'wikitext':
        return load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
    elif dataset_name.lower() == 'wikitext-103':
        return load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split=split)
    elif dataset_name.lower() == 'pg19':
        return load_dataset("deepmind/pg19", split=split)
    elif dataset_name.lower() == 'wiki40b':
        dataset = load_dataset("google/wiki40b", language, split=split)
        if language in LANGUAGES_TO_DECODE_FROM_BYTES:
            dataset = dataset.map(lambda x: {
                "text": bytes(x["text"][2:-1], "utf-8").decode("unicode_escape").encode("latin1").decode("utf-8").replace("_NEWLINE_", "\n")
            })
        return dataset
    else:
        raise ValueError(
            "Dataset not recognized. Available options: 'wikitext-2', 'wikitext-103', 'pg19', 'c4', 'wiki40b', 'mc4'.")


def extract_new_words_from_dataset(
        dataset: Dataset, tokenizer, text_column: str = "text", max_samples: int = None, filter_func=(lambda word, token_count: True)):
    """
    Loads a Hugging Face dataset and extracts all unique words from the specified text column.

    Args:
        dataset (Dataset): Name of the dataset to load.
        split (str): Dataset split to use, typically 'train' for training data. Defaults to 'train'.
        text_column (str): The column in the dataset containing text. Defaults to 'text'.
        max_samples (int): Number of samples from the dataset to go over.

    Returns:
        set: A set of unique words in the dataset.
    """
    if max_samples:
        dataset = dataset.select(range(max_samples))

    # Regular expression to split text into words (adjust as needed for specific languages)
    # word_pattern = re.compile(r"\b\w+\b")
    word_pattern = re.compile(r"\b\w+(?:[-']\w+)*\b")

    # Iterate over each entry in the dataset and extract unique words
    new_words = list()
    for record in dataset:
        text = record.get(text_column, "")
        words = word_pattern.findall(text)
        for word in words:
            tokens = tokenizer.tokenize(word)
            token_count = len(tokens)
            if (token_count > 1) and (not tokenizer.vocab.get(word, False)) and filter_func(word, token_count):
                new_words.append(word)

    # remove duplicates and return
    return list(dict.fromkeys(new_words))


# def get_words_in_dataset_by_token_length(data, tokenizer, remove_breaks=False, filter_func=lambda: True):
#     # dict to map from tokens length (2, 3, 4...) to multi-token words in that length found in the data
#     num_tokens2multi_token_words = defaultdict(list)
#
#     # custom_punctuation = string.punctuation + "“”"
#     #
#     # def remove_possessive_s(s):
#     #     return s[:-2] if (s.endswith("'s") or s.endswith("’s")) else s
#
#     # Iterate over the dataset
#     for example in data:
#         text = example["text"]
#         if remove_breaks:
#             text.replace("--", " ")
#
#         words = re.findall(r'\b\w+\b', text)
#         # words = text.split()
#
#         for word in words:
#             # word = word.strip(custom_punctuation)
#             # word = remove_possessive_s(word)
#             tokens = tokenizer.tokenize(word)
#             token_count = len(tokens)
#             if (token_count > 1) and (not tokenizer.vocab.get(word, False)) and filter_func(word, token_count):
#                 num_tokens2multi_token_words[token_count].append(word)
#
#     # Remove duplicates in the lists, but keep insertion order
#     for k in num_tokens2multi_token_words:
#         num_tokens2multi_token_words[k] = list(dict.fromkeys(num_tokens2multi_token_words[k]))
#
#     return num_tokens2multi_token_words

def get_group_texts_func(block_size=1024):
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
        # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result
    return group_texts
