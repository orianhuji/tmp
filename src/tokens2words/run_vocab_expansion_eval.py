"""
python -m tokens2words.run_patchscopes --exp_name llama3.1-8b_wiki40b_hebrew_words_prompt=beivrit_x,x,x,x, --dataset_name wiki40b --dataset_language he --dataset_max_samples 1000 --patchscopes_max_words 10000 --patchscopes_prompt "בעברית: X, X, X, X," --words_filter_numeric --words_filter_en
python -m tokens2words.run_patchscopes --exp_name llama3.1-8b_hebrew_twitter_top_1k_words_prompt_beivrit_x,x,x,x, --words_list experiments/top_1k_hebrew_words_twitter.txt --patchscopes_prompt "בעברית: X, X, X, X,"
python -m tokens2words.run_patchscopes --exp_name llama3.1-8b_hebrew_top_5k_words_prompt_beivrit_x,x,x,x, --words_list experiments/top_5k_hebrew_words_without_nikud.txt --patchscopes_prompt "בעברית: X, X, X, X,"
python -m tokens2words.run_patchscopes --exp_name llama3.1-8b_arabic_top_5k_words_prompt_belarabia_x,x,x,x, --words_list experiments/top_5k_arabic_words.txt --patchscopes_prompt "بالعربية: X, X, X, X,"
"""

import argparse
import os
import json
import random
from typing import List
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator
from accelerate import Accelerator
from accelerate.utils import set_seed
from collections import defaultdict
from torch.utils.data import DataLoader
from copy import deepcopy

from .word_retriever import PatchscopesRetriever
from .representation_translator import LinearRepresentationTranslators
from .vocab_modifier import DetokenizationVocabularyExpander
from .utils.model_utils import extract_token_i_hidden_states
from .utils.file_utils import parse_string_list_from_file
from .utils.data_utils import load_lm_dataset, extract_new_words_from_dataset, get_group_texts_func
from .utils.eval_utils import flip_tensor, compute_topk_token_rank

import logging

# Configure logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def eval_lm(
        model, tokenizer, baseline_tokenizer, dataset,
        batch_size: int = 8,
        top_k: int = 5,
        new_token_ids=None, replaced_token_seqs_by_len=None,
        truncate_new_logits: bool = False,
        text_col_name: str = "text",
        max_length=256,
):
    accelerator = Accelerator()

    if tokenizer.bos_token is not None and max_length:
        add_start_token = True
        # leave room for <BOS> token to be added:
        max_tokenized_len = max_length - 1
    else:
        add_start_token = False
        max_tokenized_len = max_length

    def tokenize_function(examples):
        output = tokenizer(
            examples[text_col_name],
            return_token_type_ids=False,
            add_special_tokens=False,
            # padding=False,
            # truncation=True if max_tokenized_len else False,
            # max_length=max_tokenized_len,
            # return_tensors="pt",
            # return_attention_mask=True,
        )
        return output

    def baseline_tokenize_function(examples):
        output = baseline_tokenizer(
            examples[text_col_name],
            return_token_type_ids=False,
            add_special_tokens=False,
            # padding=False,
            # truncation=True if max_tokenized_len else False,
            # max_length=max_tokenized_len,
            # return_tensors="pt",
            # return_attention_mask=True,
        )
        return output

    column_names = dataset.column_names

    with accelerator.main_process_first():
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=column_names,
            load_from_cache_file=False,
            desc="Running tokenizer on dataset",
        )
        group_texts = get_group_texts_func(block_size=max_tokenized_len)
        lm_dataset = tokenized_dataset.map(
            group_texts,
            batched=True,
        )

        baseline_tokenized_dataset = dataset.map(
            baseline_tokenize_function,
            batched=True,
            remove_columns=column_names,
            load_from_cache_file=False,
            desc="Running baseline tokenizer on dataset",
        )
        baseline_lm_dataset = baseline_tokenized_dataset.map(
            group_texts,
            batched=True,
        )

    data_collator = default_data_collator

    # Create data loaders
    expanded_vocab_dataloader = DataLoader(
        lm_dataset, collate_fn=data_collator, batch_size=batch_size, drop_last=False,
    )
    baseline_vocab_dataloader = DataLoader(
        baseline_lm_dataset, collate_fn=data_collator, batch_size=batch_size, drop_last=False,
    )
    model, expanded_vocab_dataloader, baseline_vocab_dataloader = accelerator.prepare(
        model, expanded_vocab_dataloader, baseline_vocab_dataloader)
    model.eval()

    if new_token_ids is not None:
        new_token_ids = torch.tensor(new_token_ids).to(model.device)
    if replaced_token_seqs_by_len is not None:
        replaced_token_seqs_by_len = {token_length: torch.tensor(skip_token_seqs).to(model.device) for token_length, skip_token_seqs in replaced_token_seqs_by_len.items() if len(skip_token_seqs) > 0}

    metrics_per_example = defaultdict(list)
    baseline_metrics_per_example = defaultdict(list)

    # for perplexity
    ce_loss_func = nn.CrossEntropyLoss(reduction="none")

    def _compute_metrics(logits, labels, attention_mask, compute_subsequent_metrics=False):
        results = dict()
        if compute_subsequent_metrics:
            # prepare labels and attentions masks for computing metrics only for the 1st tokens following the new words
            subsequent_labels = logits[..., 1:]
            subsequent_attention_mask = flip_tensor(attention_mask[..., :-1].contiguous())

        results["perplexity"] = perplexity_batch = torch.exp(
            (ce_loss_func(logits.transpose(1, 2), labels) * attention_mask).sum(1)
            / attention_mask.sum(1)
        ).detach().cpu().numpy()

        top1 = logits.argmax(dim=-1)
        results["top1_acc"] = top1_acc = (((
                             labels == top1) * attention_mask).sum() / attention_mask.sum()).detach().cpu().numpy()
        if compute_subsequent_metrics:
            results["subsequent_top1_acc"] = subsequent_top1_acc = (((subsequent_labels == top1[...,
                                                          1:]) * subsequent_attention_mask).sum() / subsequent_attention_mask.sum()).detach().cpu().numpy()

        topk = logits.topk(top_k, dim=-1).indices
        results["topk_acc"] = topk_acc = (((topk == labels.unsqueeze(-1)).any(
            dim=-1) * attention_mask).sum() / attention_mask.sum()).detach().cpu().numpy()
        if compute_subsequent_metrics:
            results["subsequent_topk_acc"] = subsequent_topk_acc = (((topk[..., 1:] == subsequent_labels.unsqueeze(-1)).any(
                dim=-1) * subsequent_attention_mask).sum() / subsequent_attention_mask.sum()).detach().cpu().numpy()

        rank = compute_topk_token_rank(logits, labels.clip(0, new_token_ids.min() - 1))
        results["mrr"] = mrr = (((1 / rank) * attention_mask).sum() / attention_mask.sum()).detach().cpu().numpy()
        if compute_subsequent_metrics:
            results["subsequent_mrr"] = subsequent_mrr = (((1 / rank[...,
                                    1:]) * subsequent_attention_mask).sum() / subsequent_attention_mask.sum()).detach().cpu().numpy()
        del rank

        return results

    def _compute_comparative_metrics(baseline_logits, new_logits, attention_mask, compute_subsequent_metrics=True):
        results = dict()
        # compute agreement between baseline and new-vocab model
        baseline_top1 = baseline_logits.argmax(dim=-1)
        new_top1 = new_logits.argmax(dim=-1)
        results["top1_agreement"] = (
                    ((baseline_top1 == new_top1) * attention_mask).sum() / attention_mask.sum()).detach().cpu().numpy()

        if compute_subsequent_metrics:
            subsequent_attention_mask = flip_tensor(attention_mask[..., :-1].contiguous())
            results["subsequent_top1_agreement"] = (((baseline_top1[..., 1:] == new_top1[...,
                                                                     1:]) * subsequent_attention_mask).sum() / subsequent_attention_mask.sum()).detach().cpu().numpy()
        return results

    for batch_i, batch in tqdm(enumerate(expanded_vocab_dataloader), total=len(expanded_vocab_dataloader), miniters=10):
        if add_start_token:
            bos_tokens_tensor = torch.tensor([[tokenizer.bos_token_id]] * batch["input_ids"].size(dim=0)).to(device)
            batch["input_ids"] = torch.cat([bos_tokens_tensor, batch["input_ids"]], dim=1)
            batch["labels"] = torch.cat([bos_tokens_tensor, batch["labels"]], dim=1)
            batch["attention_mask"] = torch.cat(
                [torch.ones(bos_tokens_tensor.size(), dtype=torch.int64).to(model.device), batch["attention_mask"]], dim=1
            )

        labels = batch["input_ids"]
        attn_mask = batch["attention_mask"]
        with torch.no_grad():
            outputs = model(**batch)
        out_logits = outputs.logits

        shift_logits = out_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_attention_mask_batch = attn_mask[..., 1:].contiguous()

        # Ignore token_ids of new vocabulary words in shift_labels and shift_logits
        if new_token_ids is not None:
            ignore_mask = torch.isin(shift_labels, new_token_ids)
            shift_attention_mask_batch = shift_attention_mask_batch * (~ignore_mask).long()
            if truncate_new_logits:
                shift_logits = shift_logits[:, :, :-len(new_token_ids)]
                shift_labels[ignore_mask] = tokenizer.bos_token_id

        # Ignore multi-token sequences of that were replaced with a single token
        if replaced_token_seqs_by_len is not None:
            # Create a mask that will be updated where sequences match
            ignore_mask = shift_attention_mask_batch.clone()  # Clone the attention mask to modify it
            # Loop over sequences in skip_token_seqs
            for seq_len, seqs in replaced_token_seqs_by_len.items():
                # Create a sliding window of the same size as the skip_seq and check for matches
                for i in range(shift_labels.size(1) - seq_len + 1):
                    # Check if the sequence matches at position i
                    window = shift_labels[:, i:i + seq_len]
                    curr_mask = torch.all(window.unsqueeze(1) == seqs.unsqueeze(0), dim=-1)
                    if curr_mask.any():
                        # Zero out the ignore mask for the length of the sequence
                        ignore_mask[curr_mask.any(dim=-1), i:i + seq_len] = 0
            # Apply the ignore mask to the attention mask
            shift_attention_mask_batch *= ignore_mask

        new_model_logits = shift_logits
        baseline_model_logits = shift_logits
        if new_token_ids is not None and not truncate_new_logits:
            # output vocab is expanded - to compute baseline, need to remove logits for new words
            baseline_model_logits = baseline_model_logits[:, :, :-len(new_token_ids)]

        for metric_name, metric_value in _compute_metrics(baseline_model_logits, shift_labels, shift_attention_mask_batch):
            baseline_metrics_per_example[metric_name].append(metric_value)

        for metric_name, metric_value in _compute_metrics(new_model_logits, shift_labels, shift_attention_mask_batch):
            metrics_per_example[metric_name].append(metric_value)

        for metric_name, metric_value in _compute_comparative_metrics(baseline_model_logits, new_model_logits):
            metrics_per_example[metric_name].append(metric_value)

    # TODO compute total tokens count

    mean_baseline_metrics = {k: np.mean(v) for k, v in baseline_metrics_per_example.items()}
    mean_new_vocab_metrics = {k: np.mean(v) for k, v in metrics_per_example.items()}
    return mean_new_vocab_metrics, mean_baseline_metrics


def compute_input_expansion_metrics(model, vocab_modifier, tokenizer, baseline_tokenizer, eval_dataset):

    # compute baseline
    skip_token_seqs_by_len = dict()
    for k in eval_ks:
        skip_token_seqs_by_len[k] = np.array([word_orig_tokenization[word] for word in set(final_new_words).intersection(token_length2words[k])])

    perplexity_data = dataset["test"]

    all_results['baseline'] = {
        'perplexity': compute_integrity_metrics(model, baseline_tokenizer, perplexity_data, metric=args.metric, batch_size=4, add_start_token=baseline_tokenizer.bos_token_id is not None, skip_token_ids=new_token_indices, skip_token_seqs_by_len=skip_token_seqs_by_len, truncate_new_logits=True, max_eval_samples=args.max_eval_samples),
        'num_tokens': count_tokens_in_dataset(dataset["test"], baseline_tokenizer)
    }
    print(f"Baseline - Perplexity: {all_results['baseline']['perplexity']}, num_tokens: {all_results['baseline']['num_tokens']}")

    ppl = compute_integrity_metrics(model, tokenizer, perplexity_data, metric=args.metric, batch_size=4, add_start_token=tokenizer.bos_token_id is not None, skip_token_ids=new_token_indices, truncate_new_logits=True, max_eval_samples=args.max_eval_samples)
    new_tokens_count = count_tokens_in_dataset(dataset["test"], tokenizer)
    all_results["input_expansion"] = {
        'perplexity': ppl,
        'num_tokens': new_tokens_count,
    }
    print(f" Perplexity: {ppl}, num_tokens: {new_tokens_count}")


def compute_output_expansion_metrics():
    # perplexity
    ppl = compute_integrity_metrics(model, tokenizer, perplexity_data, metric=args.metric, batch_size=8,
                                    add_start_token=tokenizer.bos_token_id is not None,
                                    skip_token_ids=new_token_indices, truncate_new_logits=False,
                                    max_eval_samples=args.max_eval_samples)
    new_tokens_count = count_tokens_in_dataset(dataset["test"], tokenizer)
    perplexity_results[expansion_type] = {
        'perplexity': ppl,
        'num_tokens': new_tokens_count,
    }
    print(f"expansion type {expansion_type} - Perplexity: {ppl}, num_tokens: {new_tokens_count}")

    # ranks and probabilities
    for target_word in tqdm(patchscopes_good_words, desc="Evaluating texts ending with new words...", miniters=10, ):
        eval_texts = get_texts_ending_with_word(target_word, dataset["test"])
        for eval_text in eval_texts:
            curr_result = eval_single_text(model, tokenizer, orig_tokenizer, eval_text, target_word,
                                           new_token_ids=new_token_indices)
            # if curr_result['baseline']['rank'] <= 10:
            # curr_result.pop('top_1')
            results[target_word].append(curr_result)
            num_contexts_per_word[target_word] += 1

    for target_word in results.keys():
        mean_results["mean_target_rank"][target_word] = np.mean(
            [result['target']['rank'] for result in results[target_word]])
        mean_results["mean_orig_rank"][target_word] = np.mean(
            [result['orig']['rank'] for result in results[target_word]])
        mean_results["mean_baseline_rank"][target_word] = np.mean(
            [result['baseline']['rank'] for result in results[target_word]])
        mean_results["mean_target_reciprocal_rank"][target_word] = np.mean(
            [1 / result['target']['rank'] for result in results[target_word]])
        mean_results["mean_orig_reciprocal_rank"][target_word] = np.mean(
            [1 / result['orig']['rank'] for result in results[target_word]])
        mean_results["mean_baseline_reciprocal_rank"][target_word] = np.mean(
            [1 / result['baseline']['rank'] for result in results[target_word]])
        mean_results["mean_target_prob"][target_word] = np.mean(
            [result['target']['probability'] for result in results[target_word]])
        mean_results["mean_orig_prob"][target_word] = np.mean(
            [result['orig']['probability'] for result in results[target_word]])
        mean_results["mean_baseline_prob"][target_word] = np.mean(
            [result['orig']['probability'] for result in results[target_word]])

    print(f"*** expansion type {expansion_type} ***")
    print(f"mean target rank: {np.mean(list(mean_results['mean_target_rank'].values()))}")
    print(f"mean orig rank: {np.mean(list(mean_results['mean_orig_rank'].values()))}")
    print(f"mean baseline rank: {np.mean(list(mean_results['mean_baseline_rank'].values()))}")
    print(f"mean target RR: {np.mean(list(mean_results['mean_target_reciprocal_rank'].values()))}")
    print(f"mean orig RR: {np.mean(list(mean_results['mean_orig_reciprocal_rank'].values()))}")
    print(f"mean baseline RR: {np.mean(list(mean_results['mean_baseline_reciprocal_rank'].values()))}")
    print(f"mean target prob: {np.mean(list(mean_results['mean_target_prob'].values()))}")
    print(f"mean orig prob: {np.mean(list(mean_results['mean_orig_prob'].values()))}")
    print(f"mean baseline prob: {np.mean(list(mean_results['mean_baseline_prob'].values()))}")
    print()

    rank_mean_results[expansion_type] = {
        "mean_target_rank": mean_results['mean_target_rank'],
        "mean_orig_rank": mean_results['mean_orig_rank'],
        "mean_baseline_rank": mean_results['mean_baseline_rank'],
        "mean_target_reciprocal_rank": mean_results['mean_target_reciprocal_rank'],
        "mean_orig_reciprocal_rank": mean_results['mean_orig_reciprocal_rank'],
        "mean_baseline_reciprocal_rank": mean_results['mean_baseline_reciprocal_rank'],
        "mean_target_prob": mean_results['mean_target_prob'],
        "mean_orig_prob": mean_results['mean_orig_prob'],
        "mean_baseline_prob": mean_results['mean_baseline_prob'],
        "N": num_contexts_per_word,
    }
    rank_detailed_results[expansion_type] = results


def get_word_filter(args):

    def word_filter(word, token_count):
        is_valid = True
        if args.words_filter_max_n_tokens and not (token_count <= args.words_filter_max_n_tokens):
            is_valid = False
        if args.words_filter_non_en and not all('a' <= char <= 'z' or 'A' <= char <= 'Z' for char in word):
            is_valid = False
        if args.words_filter_numeric and not word.isalpha():
            is_valid = False
        return is_valid

    return word_filter


def prepare_new_words(
        args, tokenizer):

    _word_filter = get_word_filter(args)

    def _get_token_length(word):
        return len(tokenizer.tokenize(word))

    if not args.words_list:
        new_words = list()
    else:
        new_words = parse_string_list_from_file(args.words_list, args.words_list_delimiter)
        new_words = [w for w in new_words if not tokenizer.vocab.get(w, False) and _word_filter(w, _get_token_length(w))]

    if args.words_dataset_name:
        words_dataset = load_lm_dataset(args.words_dataset_name, args.words_dataset_language)
        words_dataset = words_dataset[args.words_dataset_split]
        if args.words_dataset_max_samples:
            words_dataset = words_dataset.select(range(args.words_dataset_max_samples))

        new_words += extract_new_words_from_dataset(
            words_dataset, tokenizer, args.words_dataset_text_col, filter_func=_word_filter)

    baseline_tokenization = {w: tokenizer.encode(w, add_special_tokens=False, return_tensors="pt")[0]
                             for w in new_words}

    return new_words, baseline_tokenization


def prepare_patchscopes_retriever(args, model, tokenizer):
    patchscopes_retriever = PatchscopesRetriever(
        model, tokenizer,
        args.extraction_prompt,
        args.patchscopes_prompt,
        args.patchscopes_prompt_target,
        args.patchscopes_generate_n_tokens,
    )

    patchscopes_results = None
    if args.patchscopes_results_cache:
        patchscopes_results = pd.read_parquet(args.patchscopes_results_cache)

    return patchscopes_retriever, patchscopes_results


def prepare_translators(args, model, tokenizer):

    if args.translators_path:
        translators = torch.load(args.translators_path, map_location=torch.device('cpu'))
    else:
        translators = LinearRepresentationTranslators()
        translators.fit_on_tokens(
            model, tokenizer,
            prompt=args.extraction_prompt,
        )

    return translators


def eval_input_perplexity(
        args, model, tokenizer,
):
    logger.info("*** Evaluating expanding input vocabulary ***")
    logger.info("Preparing list of words to add to vocabulary...")
    new_words, orig_tokenization = prepare_new_words(args, tokenizer)

    logger.info("Running patchscopes on new words...")
    patchscopes_retriever, patchscopes_results = prepare_patchscopes_retriever(args, model, tokenizer)

    logger.info("Prpearing transformations to embedding and unembedding spaces...")
    translators = prepare_translators(args, model, tokenizer)
    if not args.translators_path:
        torch.save(translators, os.path.join(args.output_dir, f"{args.exp_name}", "translators"))

    logger.info("Adding new words to model vocabulary...")
    accelerator = Accelerator()
    model = accelerator.prepare(model)
    model.eval()
    vocab_modifier = DetokenizationVocabularyExpander(
        model, tokenizer,
        patchscopes_retriever, patchscopes_results,
        translators,
        args.detokenization_decision_rule,
    )
    vocab_modifier.add_words_to_vocab(new_words)
    logger.info("Done adding words! Patchscopes success rate: "
          f"{len(vocab_modifier.new_words) / (len(vocab_modifier.new_words) + len(vocab_modifier.failed_words))}")

    logger.info("Saving updated patchscopes cache to file...")
    updated_patchscopes_results = vocab_modifier.get_patchscopes_results()
    if len(updated_patchscopes_results) > len(patchscopes_results):
        patchscopes_results = updated_patchscopes_results
        if args.patchscopes_results_cache:
            patchscopes_results.to_parquet(args.patchscopes_results_cache)
        else:
            patchscopes_results.to_parquet(os.path.join(args.output_dir, f"{args.exp_name}", "patchscopes_results.parquet"))

    # compute metrics
    eval_dataset = load_lm_dataset(args.eval_dataset_name, args.eval_dataset_language)
    eval_dataset = eval_dataset[args.eval_dataset_split]
    if args.eval_dataset_max_samples:
        eval_dataset = eval_dataset.select(range(args.eval_dataset_max_samples))

    all_results = dict()

    results_df = pd.DataFrame.from_dict(all_results)
    results_df.to_json(os.path.join(args.output_dir, f"{args.exp_name}", "patchscopes_results.parquet"))

    return all_results


def eval_output_expansion(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    word_embeddings = dict()
    word_orig_tokenization = dict()

    if args.patchscopes_results is not None:

        with open(args.patchscopes_results, "rb") as fp:
            patchscopes_results = pickle.load(fp)

    # Initialize weights for new words

    model.eval()
    with torch.no_grad():
        for new_word in tqdm(new_words, total=len(new_words), desc="Adding new words to vocabulary...", miniters=10,):
            # add new word to LM head
            # target_text = orig_tokenizer(new_word, return_tensors="pt").to(device)
            target_text = orig_tokenizer(f"{new_word}", return_tensors="pt").to(device)
            orig_word_tokens = target_text['input_ids'][0, 1:]
            orig_word_first_token = orig_word_tokens[0]
            word_orig_tokenization[new_word] = orig_word_tokens.detach().cpu().numpy()

            if args.extract_repeats:
                target_prompt = "{target} {target} {target} {target}"
            else:
                target_prompt = "{target}"
            target_text = orig_tokenizer(target_prompt.format(target=new_word), return_tensors="pt").to(device)

            # Pass target text through the model to get hidden states
            with torch.no_grad():
                outputs = model(**target_text, output_hidden_states=True)
            # Extract hidden state from the last token of the target text
            last_token_hidden_states = [layer_hidden_states[0, -1, :].detach() for layer_hidden_states in outputs.hidden_states]

            # get identity patchscope results for each word
            if args.patchscopes_results is None or new_word not in patchscopes_all_correct_layers:
                # get identity patchscope results for each word
                patchscopes_correct_layers, patchscopes_outputs = get_identity_patchscopes_results(model, orig_tokenizer, new_word, last_token_hidden_states)  #, mappings=linear_mappings.layer_mappings)
                patchscopes_all_outputs[new_word] = patchscopes_outputs
                patchscopes_all_correct_layers[new_word] = patchscopes_correct_layers
            else:
                patchscopes_correct_layers = patchscopes_all_correct_layers[new_word]

            if len(patchscopes_correct_layers) > 0:
                final_new_words.append(new_word)
                first_correct_layer = patchscopes_correct_layers[0]
                # last_correct_layer = patchscopes_correct_layers[-1]
                # if len(patchscopes_correct_layers) > 1:
                #     second_correct_layer = patchscopes_correct_layers[1]
                # else:
                #     second_correct_layer = first_correct_layer

                if args.patchscopes_results is None or new_word not in patchscopes_good_words:
                    patchscopes_good_words.append(new_word)
                word_embeddings[new_word] = {
                    '1st_correct_patchscopes': (first_correct_layer, outputs.hidden_states[first_correct_layer][0, -1, :].squeeze(0).detach().cpu().numpy().reshape(1, -1)),
                    '1st_correct_patchscopes_no_map': (0, outputs.hidden_states[first_correct_layer][0, -1, :].squeeze(0).detach().cpu().numpy().reshape(1, -1)),
                    'last_token_embedding': (0, outputs.hidden_states[0][0, -1, :].squeeze(0).detach().cpu().numpy().reshape(1, -1)),
                    # '2nd_correct_patchscopes': (second_correct_layer, outputs.hidden_states[second_correct_layer][0, -1, :].squeeze(0).detach().cpu().numpy().reshape(1, -1)),
                    # 'last_correct_patchscopes': (last_correct_layer, outputs.hidden_states[last_correct_layer][0, -1, :].squeeze(0).detach().cpu().numpy().reshape(1, -1)),
                }
            else:
                if args.patchscopes_results is None or new_word not in patchscopes_bad_words:
                    patchscopes_bad_words.append(new_word)

    print(f"Patchscopes success rate: {len(final_new_words)/len(new_words)}")

    patchscopes_results = {
        "outputs": patchscopes_all_outputs,
        "correct_layers": patchscopes_all_correct_layers,
        "identified": patchscopes_good_words,
        "failed": patchscopes_bad_words,
    }
    with open(os.path.join(output_dir, f"{experiment_name}_patchscopes_results.pkl"), 'wb') as fp:
        pickle.dump(patchscopes_results, fp)

    # Add new tokens to the tokenizer
    num_added_tokens = tokenizer.add_tokens(final_new_words)

    # Resize the token embeddings of the model
    model.resize_token_embeddings(len(orig_tokenizer) + num_added_tokens)

    # Initialize the embeddings for the new tokens
    new_token_indices = list(range(len(tokenizer) - num_added_tokens, len(tokenizer)))

    # EVALUATION
    def get_token_rank_and_probability(tokenizer, logits, probabilities, token_id):
        # Get the probability and rank of token_id
        prob = probabilities[token_id].item()
        logit = logits[token_id].item()
        rank = (logits.argsort(descending=True) == token_id).nonzero(as_tuple=True)[0].item() + 1
        token_str = tokenizer.decode(token_id)
        return {
            "token": token_str,
            "probability": prob,
            "score": logit,
            "rank": rank
        }

    def eval_single_text(model, tokenizer, orig_tokenizer, eval_text, target_word, new_token_ids=None):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Pass the input context through the model
        input_context = tokenizer(eval_text, return_tensors="pt").to(device)
        outputs = model(**input_context)

        # Check if the last logit predicted corresponds to the newly added row
        logits = outputs.logits[0, -1, :].detach()
        probs = F.softmax(logits, dim=-1)

        target_token_id = tokenizer.encode(target_word)[-1]
        # top_1_id = logits.argmax()
        orig_first_token = orig_tokenizer.encode(target_word)[1]
        results = {
            'target': get_token_rank_and_probability(tokenizer, logits, probs, target_token_id),
            'orig': get_token_rank_and_probability(tokenizer, logits, probs, orig_first_token),
            # 'top_1': get_token_rank_and_probability(tokenizer, logits, probs, top_1_id),
        }
        if new_token_ids is not None:
            results['baseline'] = get_token_rank_and_probability(tokenizer, logits[:-len(new_token_ids)], F.softmax(logits[:-len(new_token_ids)], dim=-1), orig_first_token)

        return results

    rank_mean_results = dict()
    rank_detailed_results = dict()
    perplexity_results = dict()

    # compute baseline
    skip_token_seqs_by_len = dict()
    for k in eval_ks:
        skip_token_seqs_by_len[k] = np.array([word_orig_tokenization[word] for word in set(final_new_words).intersection(token_length2words[k])])

    perplexity_data = dataset["test"]

    perplexity_results['baseline'] = {
        'perplexity': compute_integrity_metrics(model, orig_tokenizer, perplexity_data, metric=args.metric, batch_size=8, add_start_token=orig_tokenizer.bos_token_id is not None, skip_token_seqs_by_len=skip_token_seqs_by_len, truncate_new_logits=True, max_eval_samples=args.max_eval_samples),
        'num_tokens': count_tokens_in_dataset(dataset["test"], orig_tokenizer)
    }
    print(f"Baseline - Perplexity: {perplexity_results['baseline']['perplexity']}, num_tokens: {perplexity_results['baseline']['num_tokens']}")

    # now for expasnions
    for expansion_type in ['1st_correct_patchscopes', '1st_correct_patchscopes_no_map', 'last_token_embedding']: #, '2nd_correct_patchscopes', 'last_correct_patchscopes']:

        results = defaultdict(list)
        num_contexts_per_word = defaultdict(lambda: 0)
        mean_results = defaultdict(dict)

        for idx, new_word in zip(new_token_indices, final_new_words):
            # add new word to embeddings
            target_layer, target_hidden_state = word_embeddings[new_word][expansion_type]
            target_as_lm_head = target_hidden_state
            target_as_embedding = target_hidden_state

            if target_layer != 0:
                if not args.dont_map_to_lm:
                    target_as_lm_head = linear_mappings.layer_mappings[target_layer]['lm_head'].predict(target_hidden_state)
                target_as_embedding = linear_mappings.layer_mappings[target_layer]['embedding'].predict(target_hidden_state)

            target_as_lm_head = torch.from_numpy(target_as_lm_head).to(
                model.get_input_embeddings().weight.dtype).to(device)
            target_as_embedding = torch.from_numpy(target_as_embedding).to(
                model.get_input_embeddings().weight.dtype).to(device)

            with torch.no_grad():
                model.get_input_embeddings().weight[idx] = target_as_embedding
                model.lm_head.weight[idx] = target_as_lm_head

        if args.renormalize:
            with torch.no_grad():
                lm_head_rows = model.lm_head.weight.detach()
                normalized_lm_head_rows = torch.concat([lm_head_rows[:-len(new_words)], scale_new_rows_to_existing_average_norm(lm_head_rows[:-len(new_words)], lm_head_rows[-len(new_words):])])
                model.lm_head.weight = nn.Parameter(normalized_lm_head_rows)


    results_df = pd.DataFrame.from_dict(rank_mean_results)
    results_df.to_json(os.path.join(output_dir, f"{experiment_name}_results.json"))
    perplexity_df = pd.DataFrame.from_dict(perplexity_results)
    perplexity_df.to_json(os.path.join(output_dir, f"{experiment_name}_perplexity.json"))

    # convert rank_detailed_results to dfs
    rank_detailed_results_dfs = dict()
    for target_layer, layer_results in rank_detailed_results.items():
        rank_detailed_results_dfs[target_layer] = pd.concat([pd.json_normalize(layer_results[word]) for word in layer_results.keys()])

    with open(os.path.join(output_dir, f"{experiment_name}_detailed_results.pkl"), 'wb') as fp:
        pickle.dump(rank_detailed_results_dfs, fp)

    return rank_detailed_results_dfs


def main(args):
    set_seed(args.seed)

    output_dir = os.path.join(args.output_dir, args.exp_name)
    os.makedirs(output_dir, exist_ok=True)

    logger.info("Loading model...")
    accelerator = Accelerator()
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    model = accelerator.prepare(model)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)


    config = vars(args)
    with open(os.path.join(output_dir, f"config.json"), "w") as config_file:
        json.dump(config, config_file, indent=4)

    logger.info(f"Results saved to: {output_dir}")

    return


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate vocabulary expansion.")
    parser.add_argument("--exp_name", type=str)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--detokenization_decision_rule", type=str, default="first_id_layer")
    parser.add_argument("--extraction_prompt", type=str, default="X")
    parser.add_argument("--patchscopes_prompt", type=str, default="X, X, X, X,")
    parser.add_argument("--patchscopes_prompt_target", type=str, default="X")
    parser.add_argument("--patchscopes_results_cache", type=str, default=None)
    parser.add_argument("--patchscopes_generate_n_tokens", type=int, default=20)
    parser.add_argument("--patchscopes_max_words", type=int, default=None)
    parser.add_argument("--translators_path", type=str, default=None)
    parser.add_argument("--eval_dataset", type=str, default=None)
    parser.add_argument("--eval_dataset_language", type=str, default=None)
    parser.add_argument("--eval_dataset_max_samples", type=int, default=None)
    parser.add_argument("--eval_split", type=str, default="test")
    parser.add_argument("--words_dataset", type=str, default=None)
    parser.add_argument("--words_dataset_language", type=str, default=None)
    parser.add_argument("--words_dataset_max_samples", type=int, default=None)
    parser.add_argument("--words_dataset_split", type=str, default="train")
    parser.add_argument("--words_dataset_text_col", type=str, default="text")
    parser.add_argument("--words_list", type=str, default=None)
    parser.add_argument("--words_list_delimiter", type=str, default=None)
    parser.add_argument("--words_filter_max_n_tokens", type=int, default=5)
    parser.add_argument("--words_filter_non_en", action="store_true", default=False)
    parser.add_argument("--words_filter_numeric", action="store_true", default=False)
    parser.add_argument("--output_dir", type=str, default="./experiments/")

    args = parser.parse_args()

    assert args.dataset_name is not None or args.words_list is not None, \
        "Please pass either a dataset name (--dataset_name) " \
        "or a path to a file containing a list of words (--words_list)"

    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
