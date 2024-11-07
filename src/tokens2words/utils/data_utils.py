import re
from datasets import load_dataset, Dataset
import tensorflow_datasets as tfds

LANGUAGES_TO_DECODE_FROM_BYTES = ["he"]


def load_pretrain_dataset(dataset_name, language="en", split=None):
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


def extract_unique_words(dataset: Dataset, split: str = "validation", text_column: str = "text", max_samples: int = None):
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
    if split is not None:
        dataset = dataset[split]
    if max_samples:
        dataset = dataset.select(range(max_samples))

    # Regular expression to split text into words (adjust as needed for specific languages)
    # word_pattern = re.compile(r"\b\w+\b")
    word_pattern = re.compile(r"\b\w+(?:[-']\w+)*\b")

    # Iterate over each entry in the dataset and extract unique words
    unique_words = set()
    for record in dataset:
        text = record.get(text_column, "")
        words = word_pattern.findall(text)
        unique_words.update(words)

    return list(unique_words)
