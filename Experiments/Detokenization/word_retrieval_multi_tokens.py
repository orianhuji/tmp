import os

import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from Utils.MultiTokenKind import MultiTokenKind
from Utils.WordRetriever import WordRetriever
from Utils.utils import save_df_to_dir


def process_dataset_for_models(models_info, dataset, num_tokens_to_generate=1):
    for model_name, model_path in models_info.items():
        print(f"Processing model: {model_name}")

        model = AutoModelForCausalLM.from_pretrained(model_path).to(device).to(dtype)
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        word_retriever = WordRetriever(model, tokenizer, MultiTokenKind.Natural,
                                       num_tokens_to_generate=num_tokens_to_generate, add_context=True,
                                       model_name=model_name, device=device, dataset=dataset)

        for add_context in [True]:
            print(f"Processing model: {model_name} with context: {add_context}")
            results = word_retriever.retrieving_words_in_dataset(number_of_corpora_to_retrieve=4)
            results_df = pd.DataFrame(results)
            print(results_df)

            save_df_to_dir(
                results_df=results_df,
                base_dir="Data",
                sub_dirs=["outputs", "retrieval", "multi_tokens"],
                file_name_format="{model_name}_results_{context}.csv",
                add_context=True,
                model_name="LLaMA3-8B"
            )


if __name__ == "__main__":
    dtype = torch.bfloat16
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    models_info = {
        "LLaMA3-8B": "meta-llama/Meta-Llama-3-8B",
        "Mistral-7B": "mistralai/Mistral-7B-v0.1",
        "Yi-6B": "01-ai/Yi-6B",
        "gemma-2-9b": "google/gemma-2-9b",
        'LLaMa-2B': 'meta-llama/Llama-2-7b-hf',
        "pythia-6.9b": "EleutherAI/pythia-6.9B",
    }

    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', trust_remote_code=True)

    process_dataset_for_models(models_info, dataset)
