"""

"""

import argparse
import os
import json
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
from .utils.file_utils import parse_string_list_from_file
from .utils.data_utils import load_lm_dataset, extract_new_words_from_dataset, get_group_texts_func
from .utils.eval_utils import get_last_zero_in_every_seq_mask, get_first_zero_in_every_seq_mask, compute_topk_token_rank
from .utils.eval_utils import count_tokens_in_dataset

import logging

# Configure logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def eval_lm(
        model, accelerator, tokenizer, baseline_tokenizer, dataset,
        batch_size: int = 8,
        top_k: int = 5,
        new_token_ids=None, replaced_token_seqs_by_len=None,
        text_col_name: str = "text",
        max_length: int = 256,
        eval_max_samples: int = None,
):

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

        baseline_vocab_total_tokens = count_tokens_in_dataset(dataset, baseline_tokenizer, text_col_name)
        new_vocab_total_tokens = count_tokens_in_dataset(dataset, tokenizer, text_col_name)

    logger.info(f"Baseline tokenizer - total tokens: {baseline_vocab_total_tokens}")
    logger.info(f"Expanded tokenizer - total tokens: {new_vocab_total_tokens}")

    if eval_max_samples:
        lm_dataset = lm_dataset.select(range(eval_max_samples))
        baseline_lm_dataset = baseline_lm_dataset.select(range(eval_max_samples))

    data_collator = default_data_collator

    # Create data loaders
    expanded_vocab_dataloader = DataLoader(
        lm_dataset, collate_fn=data_collator, batch_size=batch_size, drop_last=True, shuffle=False,
    )
    baseline_vocab_dataloader = DataLoader(
        baseline_lm_dataset, collate_fn=data_collator, batch_size=batch_size, drop_last=True, shuffle=False,
    )
    model, expanded_vocab_dataloader, baseline_vocab_dataloader = accelerator.prepare(
        model, expanded_vocab_dataloader, baseline_vocab_dataloader)
    model.eval()

    if new_token_ids is not None:
        new_token_ids = torch.tensor(new_token_ids).to(model.device)
    if replaced_token_seqs_by_len is not None:
        replaced_token_seqs_by_len = {token_length: torch.tensor(skip_token_seqs).to(model.device) for token_length, skip_token_seqs in replaced_token_seqs_by_len.items() if len(skip_token_seqs) > 0}

    target_metrics = {
        "baseline": defaultdict(list),
        "expanded_E": defaultdict(list),
        "fully_expanded": defaultdict(list),
    }

    background_metrics = deepcopy(target_metrics)

    # for perplexity
    ce_loss_func = nn.CrossEntropyLoss(reduction="none")

    # TODO consider not aggregating results here, to enable metrics for specific words
    def _compute_metrics(logits, labels, attention_mask, original_labels=None, compute_target_metrics=True, compute_subsequent_metrics=True):
        background_results = dict()  # will hold metrics for all background tokens, i.e., not the ones we add or replace
        if compute_subsequent_metrics:
            # prepare labels and attentions masks for computing metrics only for the 1st tokens following the new words
            subsequent_labels = labels[:,  1:]
            subsequent_attention_mask = get_last_zero_in_every_seq_mask(attention_mask[..., :-1].contiguous())

        background_results["perplexity"] = torch.exp(
            (ce_loss_func(logits.transpose(1, 2), labels) * attention_mask).sum(1)
            / attention_mask.sum(1)
        ).mean().detach().cpu().numpy()

        top1 = logits.argmax(dim=-1)
        background_results["top1_acc"] = (((
                             labels == top1) * attention_mask).sum() / attention_mask.sum()).detach().cpu().numpy()
        if compute_subsequent_metrics:
            background_results["subsequent_top1_acc"] = (((subsequent_labels == top1[:, 1:]) * subsequent_attention_mask).sum() / subsequent_attention_mask.sum()).detach().cpu().numpy()

        topk = logits.topk(top_k, dim=-1).indices
        background_results["topk_acc"] = (((topk == labels.unsqueeze(-1)).any(
            dim=-1) * attention_mask).sum() / attention_mask.sum()).detach().cpu().numpy()
        if compute_subsequent_metrics:
            background_results["subsequent_topk_acc"] = (((topk[:, 1:] == subsequent_labels.unsqueeze(-1)).any(
                dim=-1) * subsequent_attention_mask).sum() / subsequent_attention_mask.sum()).detach().cpu().numpy()

        rank = compute_topk_token_rank(logits, labels)
        background_results["mrr"] = (((1 / rank) * attention_mask).sum() / attention_mask.sum()).detach().cpu().numpy()
        if compute_subsequent_metrics:
            background_results["subsequent_mrr"] = (((1 / rank[:,
                                    1:]) * subsequent_attention_mask).sum() / subsequent_attention_mask.sum()).detach().cpu().numpy()
        target_results = dict()  # will hold metrics for all the new words we add or their original tokenization
        if compute_target_metrics:
            target_mask = get_first_zero_in_every_seq_mask(attention_mask)

            target_results["top1_acc"] = (((labels == top1) * target_mask).sum() / target_mask.sum()).detach().cpu().numpy()
            if original_labels is not None:  # TODO check target_mask is for the right token, and not the one after it
                target_results["original_top1_acc"] = (
                            ((original_labels == top1) * target_mask).sum() / target_mask.sum()).detach().cpu().numpy()

            target_results["topk_acc"] = (((topk == labels.unsqueeze(-1)).any(
                dim=-1) * target_mask).sum() / target_mask.sum()).detach().cpu().numpy()
            if original_labels is not None:
                target_results["original_topk_acc"] = (((topk == original_labels.unsqueeze(-1)).any(
                dim=-1) * target_mask).sum() / target_mask.sum()).detach().cpu().numpy()

            target_results["mrr"] = (((1 / rank) * target_mask).sum() / target_mask.sum()).detach().cpu().numpy()
            if original_labels is not None:
                rank = compute_topk_token_rank(logits, original_labels)
                target_results["original_mrr"] = (((1 / rank) * target_mask).sum() / target_mask.sum()).detach().cpu().numpy()

        del rank
        return background_results, target_results

    # def _compute_comparative_metrics(baseline_logits, new_logits, attention_mask, compute_subsequent_metrics=True):
    #     background_results = dict()
    #
    #     # compute agreement between baseline and new-vocab model
    #     baseline_top1 = baseline_logits.argmax(dim=-1)
    #     new_top1 = new_logits.argmax(dim=-1)
    #     background_results["top1_agreement"] = (
    #                 ((baseline_top1 == new_top1) * attention_mask).sum() / attention_mask.sum()).detach().cpu().numpy()
    #     if compute_subsequent_metrics:
    #         subsequent_attention_mask = get_last_zero_in_every_seq_mask(attention_mask[..., :-1].contiguous())
    #         background_results["subsequent_top1_agreement"] = (((baseline_top1[..., 1:] == new_top1[...,
    #                                                                  1:]) * subsequent_attention_mask).sum() / subsequent_attention_mask.sum()).detach().cpu().numpy()
    #     return background_results

    def _add_start_token(batch):
        bos_tokens_tensor = torch.tensor([[tokenizer.bos_token_id]] * batch["input_ids"].size(dim=0)).to(batch["input_ids"].device)
        batch["input_ids"] = torch.cat([bos_tokens_tensor, batch["input_ids"]], dim=1)
        batch["labels"] = torch.cat([bos_tokens_tensor, batch["labels"]], dim=1)
        batch["attention_mask"] = torch.cat(
            [torch.ones(bos_tokens_tensor.size(), dtype=torch.int64).to(batch["attention_mask"].device), batch["attention_mask"]], dim=1)
        return batch

    def _ignore_new_words_in_attention_mask(shift_attention_mask_batch):
        # Ignore token_ids of new vocabulary words in shift_labels and shift_logits
        if new_token_ids is not None:
            ignore_mask = torch.isin(shift_labels, new_token_ids)
            shift_attention_mask_batch = shift_attention_mask_batch * (~ignore_mask).long()

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

        return shift_attention_mask_batch, ignore_mask

    # metrics for baseline
    for batch_i, batch in tqdm(enumerate(baseline_vocab_dataloader), total=len(baseline_vocab_dataloader), miniters=10, desc="Evaluating baseline vocabulary..."):
        if add_start_token:
            batch = _add_start_token(batch)

        labels = batch["input_ids"]
        attn_mask = batch["attention_mask"]
        with torch.no_grad():
            outputs = model(**batch)
        out_logits = outputs.logits

        shift_logits = out_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_attention_mask_batch = attn_mask[..., 1:].contiguous()

        shift_attention_mask_batch, ignore_mask = _ignore_new_words_in_attention_mask(shift_attention_mask_batch)

        # compute metrics for baseline vocabulary
        if new_token_ids is not None:
            # output vocab is expanded - to compute baseline, need to remove logits for new words
            baseline_model_logits = torch.cat([shift_logits[:, :, :min(new_token_ids)], shift_logits[:, :, max(new_token_ids)+1:]], dim=-1)
        else:
            baseline_model_logits = shift_logits

        background_results, target_results = _compute_metrics(baseline_model_logits, shift_labels, shift_attention_mask_batch)
        for metric_name, metric_value in target_results.items():
            target_metrics['baseline'][metric_name].append(metric_value)
        for metric_name, metric_value in background_results.items():
            background_metrics['baseline'][metric_name].append(metric_value)

    # metrics for expanded vocabulary
    for batch_i, batch in tqdm(enumerate(expanded_vocab_dataloader), total=len(expanded_vocab_dataloader),
                               miniters=10, desc="Evaluating expanded vocabulary..."):
        if add_start_token:
            batch = _add_start_token(batch)

        labels = batch["input_ids"]
        attn_mask = batch["attention_mask"]
        with torch.no_grad():
            outputs = model(**batch)
        out_logits = outputs.logits

        shift_logits = out_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_attention_mask_batch = attn_mask[..., 1:].contiguous()

        shift_attention_mask_batch, ignore_mask = _ignore_new_words_in_attention_mask(shift_attention_mask_batch)

        # compute metrics for expanded vocabulary
        if new_token_ids is not None:
            # output vocab is expanded - to compute baseline, need to remove logits for new words
            expanded_E_logits = torch.cat([shift_logits[:, :, :min(new_token_ids)], shift_logits[:, :, max(new_token_ids)+1:]], dim=-1)
        else:
            expanded_E_logits = shift_logits
        background_results, _ = _compute_metrics(expanded_E_logits, shift_labels,
                                                              shift_attention_mask_batch, compute_target_metrics=False)
        for metric_name, metric_value in background_results.items():
            background_metrics['expanded_E'][metric_name].append(metric_value)

        fully_expanded_logits = shift_logits
        # if new_token_ids is not None:
        #     shift_logits = shift_logits[:, :, :-len(new_token_ids)]
        #     shift_labels[ignore_mask] = tokenizer.bos_token_id

        # TODO add original_labels - need map to apply over labels, to change new words to their original first subword token
        background_results, target_results = _compute_metrics(fully_expanded_logits, shift_labels,
                                                              shift_attention_mask_batch, original_labels=None)
        for metric_name, metric_value in target_results.items():
            target_metrics['fully_expanded'][metric_name].append(metric_value)
        for metric_name, metric_value in background_results.items():
            background_metrics['fully_expanded'][metric_name].append(metric_value)

        # for metric_name, metric_value in _compute_comparative_metrics(baseline_model_logits, fully_expanded_logits, shift_attention_mask_batch):
        #     metrics_per_example[metric_name].append(metric_value)


    for eval_type in target_metrics.keys():
        target_metrics[eval_type] = {k: np.nanmean(v) for k, v in target_metrics[eval_type].items()}
    for eval_type in background_metrics.keys():
        background_metrics[eval_type] = {k: np.nanmean(v) for k, v in background_metrics[eval_type].items()}

    other_metircs = {
        "total_tokens": {
            "baseline": baseline_vocab_total_tokens,
            "expanded": expanded_vocab_total_tokens,
        },
    }
    return background_metrics, target_metrics, other_metircs


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

    if args.words_dataset:
        words_dataset = load_lm_dataset(args.words_dataset, args.words_dataset_language)
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
        num_tokens_to_generate=args.patchscopes_generate_n_tokens,
    )

    patchscopes_results = None
    if args.patchscopes_results_cache is not None:
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


def main(args):
    set_seed(args.seed)

    output_dir = os.path.join(args.output_dir, args.exp_name)
    os.makedirs(output_dir, exist_ok=True)

    logger.info("Loading model...")
    accelerator = Accelerator(mixed_precision="bf16" if torch.cuda.is_bf16_supported() else "fp16")
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    model = accelerator.prepare(model)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    baseline_tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    logger.info("*** Evaluating expanding input vocabulary ***")
    logger.info("Preparing list of words to add to vocabulary...")
    new_words, orig_tokenization = prepare_new_words(args, tokenizer)
    logger.info(f"Found {len(new_words)} new words: {new_words[:100]}...")
    # dump new words to file
    with open(os.path.join(output_dir, "new_words.txt"), "w") as fp:
        fp.write("\n".join(new_words))
    # # for debugging
    # new_words = new_words[:100]

    logger.info("Running patchscopes on new words...")
    patchscopes_retriever, patchscopes_results = prepare_patchscopes_retriever(args, model, tokenizer)

    logger.info("Preparing transformations to embedding and unembedding spaces...")
    translators = prepare_translators(args, model, tokenizer)
    if not args.translators_path:
        torch.save(translators, os.path.join(output_dir, "translators.pt"))

    logger.info("Adding new words to model vocabulary...")
    model.eval()
    vocab_modifier = DetokenizationVocabularyExpander(
        model, tokenizer,
        patchscopes_retriever, patchscopes_results,
        translators,
        args.detokenization_decision_rule,
        add_to_core_vocab=args.add_new_words_to_core_vocab,
        add_space_before_lowercase_word=args.add_space_before_lowercase_words,
    )
    vocab_modifier.add_words_to_vocab(new_words)
    tokenizer = vocab_modifier.tokenizer
    logger.info("Done adding words! Patchscopes success rate: "
                f"{len(vocab_modifier.new_words) / (len(vocab_modifier.new_words) + len(vocab_modifier.failed_words))}")
    logger.info("Saving updated patchscopes cache to file...")
    updated_patchscopes_results = vocab_modifier.get_patchscopes_results()
    if patchscopes_results is None or len(updated_patchscopes_results) > len(patchscopes_results):
        patchscopes_results = updated_patchscopes_results
        if args.patchscopes_results_cache:
            patchscopes_results.to_parquet(args.patchscopes_results_cache)
        else:
            patchscopes_results.to_parquet(
                os.path.join(output_dir, "patchscopes_results.parquet"))

    # compute metrics
    eval_dataset = load_lm_dataset(args.eval_dataset, args.eval_dataset_language)
    eval_dataset = eval_dataset[args.eval_dataset_split]

    background_metrics, target_metrics, other_metrics = eval_lm(
        model, accelerator, tokenizer, baseline_tokenizer, eval_dataset,
        batch_size=8, top_k=5, new_token_ids=vocab_modifier.new_token_ids, replaced_token_seqs_by_len=None,
        eval_max_samples=args.eval_max_samples)

    print(other_metrics)
    background_df = pd.DataFrame.from_dict(background_metrics)
    target_df = pd.DataFrame.from_dict(target_metrics)
    other_df = pd.DataFrame.from_dict(other_metrics)
    print(background_df)
    print(target_df)
    background_df.to_json(os.path.join(output_dir, "metrics_background.json"), indent=4)
    target_df.to_json(os.path.join(output_dir, "metrics_target.json"), indent=4)
    other_df.to_json(os.path.join(output_dir, "metrics_other.json"), indent=4)

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
    parser.add_argument("--add_new_words_to_core_vocab", action="store_true", default=False)
    parser.add_argument("--add_space_before_lowercase_words", action="store_true", default=False)
    parser.add_argument("--detokenization_decision_rule", type=str, default="first_id_layer")
    parser.add_argument("--extraction_prompt", type=str, default="X")
    parser.add_argument("--patchscopes_prompt", type=str, default="X, X, X, X,")
    parser.add_argument("--patchscopes_prompt_target", type=str, default="X")
    parser.add_argument("--patchscopes_results_cache", type=str, default=None)
    parser.add_argument("--patchscopes_generate_n_tokens", type=int, default=20)
    parser.add_argument("--patchscopes_max_words", type=int, default=None)
    parser.add_argument("--translators_path", type=str, default=None)
    parser.add_argument("--eval_dataset", type=str, default="wikitext")
    parser.add_argument("--eval_dataset_language", type=str, default=None)
    parser.add_argument("--eval_max_samples", type=int, default=None)
    parser.add_argument("--eval_dataset_split", type=str, default="test")
    parser.add_argument("--words_dataset", type=str, default=None)
    parser.add_argument("--words_dataset_language", type=str, default=None)
    parser.add_argument("--words_dataset_max_samples", type=int, default=None)
    parser.add_argument("--words_dataset_split", type=str, default="test")
    parser.add_argument("--words_dataset_text_col", type=str, default="text")
    parser.add_argument("--words_list", type=str, default=None)
    parser.add_argument("--words_list_delimiter", type=str, default=None)
    parser.add_argument("--words_filter_max_n_tokens", type=int, default=5)
    parser.add_argument("--words_filter_non_en", action="store_true", default=False)
    parser.add_argument("--words_filter_numeric", action="store_true", default=False)
    parser.add_argument("--output_dir", type=str, default="./experiments/")

    args = parser.parse_args()

    assert args.words_dataset is not None or args.words_list is not None, \
        "Please pass either a dataset name (--words_dataset) " \
        "or a path to a file containing a list of words (--words_list)"

    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
