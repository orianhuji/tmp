import torch
from torch import nn
from tqdm import tqdm
import numpy as np

from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator
from transformers import get_scheduler
from accelerate import Accelerator
from accelerate.utils import set_seed
from collections import defaultdict
from torch.utils.data import DataLoader
import torch.optim as optim

from ..utils.data_utils import load_lm_dataset, extract_new_words_from_dataset, get_group_texts_func, get_tokenize_func


class OnlineLogitsStdDev:
    def __init__(self, vocab_size, device='cpu'):
        self.vocab_size = vocab_size
        self.count = 0
        self.mean = torch.zeros(vocab_size).to(device)
        self.m2 = torch.zeros(vocab_size).to(device)

    def update(self, logits_batch):
        """
        Update the online statistics with a new batch of logits.
        Args:
            logits_batch (torch.Tensor): A batch of logits of shape (batch_size, vocab_size).
        """
        batch_size = logits_batch.size(0)*logits_batch.size(1)
        batch_mean = logits_batch.mean(dim=(0,1))
        batch_var = logits_batch.var(dim=(0,1), unbiased=False)

        delta = batch_mean - self.mean
        total_count = self.count + batch_size

        # Update mean
        self.mean += delta * batch_size / total_count

        # Update M2 (sum of squared differences from the mean)
        self.m2 += batch_var * batch_size + (delta ** 2) * self.count * batch_size / total_count

        # Update count
        self.count = total_count

    def std_dev(self):
        """
        Compute the standard deviation for each token in the vocabulary.
        Returns:
            torch.Tensor: A tensor of standard deviations of shape (vocab_size,).
        """
        return torch.sqrt(self.m2 / self.count) if self.count > 0 else torch.zeros(self.vocab_size)


def compute_logits_std_dev(model, tokenizer, orig_vocab_size, dataset, batch_size=8, max_length=256, text_col_name="text"):

    accelerator = Accelerator()

    # Tokenize data
    if tokenizer.bos_token is not None and max_length:
        add_start_token = True
        # leave room for <BOS> token to be added:
        max_tokenized_len = max_length - 1
    else:
        add_start_token = False
        max_tokenized_len = max_length

    def _add_start_token(batch):
        bos_tokens_tensor = torch.tensor([[tokenizer.bos_token_id]] * batch["input_ids"].size(dim=0)).to(batch["input_ids"].device)
        batch["input_ids"] = torch.cat([bos_tokens_tensor, batch["input_ids"]], dim=1)
        batch["attention_mask"] = torch.cat(
            [torch.ones(bos_tokens_tensor.size(), dtype=torch.int64).to(batch["attention_mask"].device), batch["attention_mask"]], dim=1)
        return batch

    tokenize_function = get_tokenize_func(tokenizer, text_col_name)

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

    data_collator = default_data_collator

    # Create data loaders
    dataloader = DataLoader(
        lm_dataset, collate_fn=data_collator, batch_size=batch_size, drop_last=False, shuffle=False,
    )

    model, dataloader = accelerator.prepare(model, dataloader)
    online_std_dev_counter = OnlineLogitsStdDev(len(tokenizer), model.device)

    model.eval()
    for step, batch in tqdm(enumerate(dataloader), total=len(dataloader), miniters=10, unit="batches", desc="Computing logits' standard deviation...."):
        if "labels" in batch:
            batch.pop("labels")
        if add_start_token:
            batch = _add_start_token(batch)
        with torch.no_grad():
            outputs = model(**batch)
        online_std_dev_counter.update(outputs.logits)

    logits_std_dev = online_std_dev_counter.std_dev()
    original_logits_std_dev = logits_std_dev[:orig_vocab_size]
    logits_scales = original_logits_std_dev.mean() / logits_std_dev
    logits_scales[:orig_vocab_size] = 1.0
    return logits_std_dev, logits_scales.squeeze()


class EmbeddingCalibrator(nn.Module):
    def __init__(self, hidden_size, eps=1e-6, dtype=torch.bfloat16):
        super().__init__()
        # self.weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype))
        # self.bias_weight = nn.Parameter(torch.zeros(hidden_size, dtype=dtype))
        # self.bias = nn.Parameter(torch.zeros(1))

        self.weight = nn.Parameter(torch.zeros(hidden_size, hidden_size, dtype=dtype))

        self.eps = eps

    def forward(self, x):
        # norm = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        # return self.weight * (x / norm)
        # return self.weight * x + self.bias
        # return self.weight * x  # + torch.matmul(x, self.bias_weight.unsqueeze(-1))
        return x + torch.matmul(x, self.weight.t())


class LogitsCalibrator(nn.Module):
    def __init__(self, vocab_size, dtype=torch.bfloat16):
        super().__init__()
        self.weight = nn.Linear(vocab_size, 1, dtype=dtype, bias=False)

    def forward(self, logits):
        return self.weight(logits)


class LMHeadWithLogitsCalibration(nn.Module):
    def __init__(self, lm_head, calibration_w, new_tokens_start_i):
        super().__init__()
        self.lm_head = lm_head
        self.calibration_w = calibration_w
        self.new_tokens_start_i = new_tokens_start_i

    def forward(self, h):
        logits = self.lm_head(h)
        gamma = self.calibration_w(logits)
        logits[..., self.new_tokens_start_i:] *= gamma
        return logits


# Add the RMSNorm layer for the new rows
class ModifiedModel(nn.Module):
    def __init__(self, base_model, lm_head, original_vocab_size, num_new_tokens, calibrate_lm_head=True, calibrate_embedding=True, calibrate_logits=False):
        super().__init__()
        self.base_model = base_model
        self.lm_head = lm_head
        self.new_tokens_start = original_vocab_size
        self.new_tokens_end = original_vocab_size + num_new_tokens
        self.unembedding_calibrator = EmbeddingCalibrator(base_model.config.hidden_size)
        self.embedding_calibrator = EmbeddingCalibrator(base_model.config.hidden_size)
        self.logits_calibrator = LogitsCalibrator(original_vocab_size+num_new_tokens)
        self.calibrate_lm_head = calibrate_lm_head
        self.calibrate_embedding = calibrate_embedding
        self.calibrate_logits = calibrate_logits

        # self.loss_fct = nn.CrossEntropyLoss(reduction="mean")
        self.loss_fct = nn.CrossEntropyLoss(reduction="none")
        self.original_tokens_loss_alpha = 0.6
        self.subsequent_tokens_loss_alpha = 0.3
        self.new_tokens_loss_alpha = 0.1

    def forward(self, input_ids, labels, attention_mask=None):
        # shift labels by 1 for CLM
        labels = labels[:, 1:].contiguous()
        input_ids = input_ids[:, :-1].contiguous()

        if self.calibrate_embedding:
            E_weights = self.base_model.get_input_embeddings().weight.data
            E_weights = torch.cat((E_weights[:self.new_tokens_start], self.embedding_calibrator(E_weights[self.new_tokens_start:])))
            input_embeddings = E_weights[input_ids]
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids, dtype=torch.long)
            outputs = self.base_model(inputs_embeds=input_embeddings, attention_mask=attention_mask)
        else:
            with torch.no_grad():
                # Forward pass through the base model
                outputs = self.base_model(input_ids, attention_mask=attention_mask)

        if self.calibrate_lm_head:
            with torch.no_grad():
                lm_head_weights = self.lm_head.weight
                normed_weights = lm_head_weights.clone()
            normed_weights[self.new_tokens_start:self.new_tokens_end] = self.unembedding_calibrator(lm_head_weights[self.new_tokens_start:self.new_tokens_end])
            logits = torch.matmul(outputs['last_hidden_state'], normed_weights.T)
        else:
            if self.calibrate_embedding:
                logits = self.lm_head(outputs['last_hidden_state'])
            else:
                with torch.no_grad():
                    logits = self.lm_head(outputs['last_hidden_state'])

        if self.calibrate_logits:
            gamma = self.logits_calibrator(logits)
            # Multiply only the last N tokens with gamma
            logits = torch.cat((logits[..., :self.new_tokens_start], logits[..., self.new_tokens_start:] * gamma), dim=-1)

        # loss = self.loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        per_example_loss = self.loss_fct(logits.transpose(1,2), labels)
        original_tokens_mask = labels < self.new_tokens_start
        new_tokens_mask = ~original_tokens_mask
        loss = 0.0
        if self.original_tokens_loss_alpha > 0.0:
            loss += self.original_tokens_loss_alpha * per_example_loss[original_tokens_mask].mean()
        if self.new_tokens_loss_alpha > 0.0:
            loss += self.new_tokens_loss_alpha * per_example_loss[new_tokens_mask].mean()
        if self.subsequent_tokens_loss_alpha > 0.0:
            subsequent_tokens_mask = torch.zeros_like(original_tokens_mask, dtype=torch.bool)
            subsequent_tokens_mask[:, 1:][new_tokens_mask[:, :-1]] = True
            loss += self.subsequent_tokens_loss_alpha * per_example_loss[subsequent_tokens_mask].mean()

        return {'loss': loss, 'logits': logits}


def get_calibration_model(model, original_vocab_size, num_new_tokens):
    modified_model = ModifiedModel(model.model, model.lm_head, original_vocab_size, num_new_tokens)
    modified_model.base_model.eval()
    modified_model.lm_head.eval()

    for param in modified_model.base_model.parameters():
        param.requires_grad = False
    for param in modified_model.lm_head.parameters():
        param.requires_grad = False
    for param in modified_model.unembedding_calibrator.parameters():
        param.requires_grad = True
    for param in modified_model.embedding_calibrator.parameters():
        param.requires_grad = True
    for param in modified_model.logits_calibrator.parameters():
        param.requires_grad = True

    return modified_model


def train_calibration_model(modified_model: ModifiedModel, tokenizer, dataset, filter_examples_without_new_tokens=True, lr=1e-4, lr_schedule="linear", num_epochs=1, batch_size=8, max_length=256, n_warmup_steps=0, text_col_name="text", clip_grad_norm=1.0):
    accelerator = Accelerator()
    # Optimizer
    optimizer = optim.AdamW(modified_model.parameters(), lr=lr)

    # Tokenize data
    if tokenizer.bos_token is not None and max_length:
        add_start_token = True
        # leave room for <BOS> token to be added:
        max_tokenized_len = max_length - 1
    else:
        add_start_token = False
        max_tokenized_len = max_length

    def _add_start_token(batch):
        bos_tokens_tensor = torch.tensor([[tokenizer.bos_token_id]] * batch["input_ids"].size(dim=0)).to(batch["input_ids"].device)
        batch["input_ids"] = torch.cat([bos_tokens_tensor, batch["input_ids"]], dim=1)
        batch["attention_mask"] = torch.cat(
            [torch.ones(bos_tokens_tensor.size(), dtype=torch.int64).to(batch["attention_mask"].device), batch["attention_mask"]], dim=1)
        return batch

    tokenize_function = get_tokenize_func(tokenizer, text_col_name)

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

    if filter_examples_without_new_tokens:
        examples_w_new_token = np.arange(len(lm_dataset))[np.any(np.array(lm_dataset['input_ids']) >= modified_model.new_tokens_start, axis=1)]
        lm_dataset = lm_dataset.select(examples_w_new_token)

    data_collator = default_data_collator

    # Create data loaders
    dataloader = DataLoader(
        lm_dataset, collate_fn=data_collator, batch_size=batch_size, drop_last=True, shuffle=True,
    )

    # Learning rate scheduler
    if isinstance(n_warmup_steps, float):
        n_warmup_steps = n_warmup_steps * len(dataloader)
    scheduler = get_scheduler(lr_schedule, optimizer=optimizer, num_warmup_steps=n_warmup_steps, num_training_steps=len(dataloader) * num_epochs)

    modified_model, dataloader = accelerator.prepare(modified_model, dataloader)

    # Freeze the original lm_head weights
    for param in modified_model.lm_head.parameters():
        param.requires_grad = False

    modified_model.train()
    for epoch in tqdm(range(num_epochs), unit="epochs", desc="Fitting calibration"):
        total_loss = 0.0
        for step, batch in tqdm(enumerate(dataloader), total=len(dataloader), miniters=10, unit="batches"):
            if add_start_token:
                batch = _add_start_token(batch)
            batch["labels"] = batch["input_ids"]
            optimizer.zero_grad()
            outputs = modified_model(**batch)
            loss = outputs['loss']
            loss.backward()
            torch.nn.utils.clip_grad_norm_(modified_model.parameters(), max_norm=clip_grad_norm)
            optimizer.step()
            scheduler.step()

            # Log loss
            total_loss += loss.item()

            if step % 10 == 0:
                print(f"Epoch {epoch + 1}, Step {step}, Loss: {loss.item()}")

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1} completed. Average Loss: {avg_loss}")

    return modified_model


def merge_calibrated_weights_to_hf_model(modified_model, hf_model):
    if modified_model.calibrate_lm_head:
        lm_head_weights = modified_model.lm_head.weight
        normed_weights = modified_model.unembedding_calibrator(lm_head_weights[modified_model.new_tokens_start:modified_model.new_tokens_end])
        with torch.no_grad():
            hf_model.lm_head.weight.data[modified_model.new_tokens_start:modified_model.new_tokens_end] = normed_weights
    if modified_model.calibrate_embedding:
        embedding_weights = modified_model.base_model.get_input_embeddings().weight
        normed_weights = modified_model.embedding_calibrator(embedding_weights[modified_model.new_tokens_start:modified_model.new_tokens_end])
        with torch.no_grad():
            hf_model.model.embed_tokens.weight.data[modified_model.new_tokens_start:modified_model.new_tokens_end] = normed_weights
    if modified_model.calibrate_logits:
        new_lm_head = LMHeadWithLogitsCalibration(hf_model.lm_head, modified_model.logits_calibrator.weight, modified_model.new_tokens_start)
        hf_model.lm_head = new_lm_head
    return hf_model