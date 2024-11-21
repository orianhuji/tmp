import torch
from torch import nn
from transformers import AutoModelForCausalLM

# Load a pretrained model
model_name = "gpt2"  # Replace with your model of choice
model = AutoModelForCausalLM.from_pretrained(model_name)

# Define the number of new tokens to add
num_new_tokens = 10

# Pre-computed embeddings for the new tokens
precomputed_embeddings = torch.randn(num_new_tokens, model.config.hidden_size)  # Replace with your embeddings

# Resize the lm_head and initialize new embeddings
model.resize_token_embeddings(model.config.vocab_size + num_new_tokens)
lm_head_weights = model.lm_head.weight.data
lm_head_weights[-num_new_tokens:] = precomputed_embeddings


# Define a custom RMSNorm layer
class CustomRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        norm = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return self.weight * (x / norm)


# Add the RMSNorm layer for the new rows
class ModifiedModel(nn.Module):
    def __init__(self, base_model, lm_head, original_vocab_size, num_new_tokens):
        super().__init__()
        self.base_model = base_model
        self.lm_head = lm_head
        self.new_token_start = original_vocab_size
        self.new_token_end = original_vocab_size + num_new_tokens
        self.rmsnorm = CustomRMSNorm(base_model.config.hidden_size)

    def forward(self, input_ids, labels=None):
        # Forward pass through the base model
        outputs = self.base_model(input_ids, labels=labels, output_hidden_states=True)

        # Apply RMSNorm to the new token embeddings in the LM head
        lm_head_weights = self.lm_head.weight

        with torch.no_grad():  # Freeze other rows during training
            normed_weights = lm_head_weights.clone()
        normed_weights[self.new_token_start:self.new_token_end] = self.rmsnorm(lm_head_weights[self.new_token_start:self.new_token_end])

        logits = torch.matmul(outputs.hidden_states[-1], normed_weights.T)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        return {'loss': loss, 'logits': logits}

# Wrap the model with the modified RMSNorm behavior
modified_model = ModifiedModel(model, num_new_tokens)

# Freeze all parameters except the RMSNorm layer
for param in modified_model.base_model.parameters():
    param.requires_grad = False
for param in modified_model.lm_head.parameters():
    param.requires_grad = False
for param in modified_model.rmsnorm.parameters():
    param.requires_grad = True


def train_calibration(modified_model, data_loader, optimizer, epochs=1):
    # Freeze the original lm_head weights
    for param in modified_model.base_model.lm_head.parameters():
        param.requires_grad = False

    modified_model.train()
    for epoch in range(epochs):
        for batch in data_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(modified_model.base_model.device)
            labels = batch['labels'].to(modified_model.base_model.device)
            outputs = modified_model(input_ids=input_ids, labels=labels)
            loss = outputs['loss']
            loss.backward()
            optimizer.step()


# After training, replace the new rows in the LM head with the RMS-normalized weights
def merge_rmsnorm_weights(modified_model):
    lm_head_weights = modified_model.base_model.lm_head.weight
    normed_weights = modified_model.rmsnorm(lm_head_weights[modified_model.new_token_start:modified_model.new_token_end])
    lm_head_weights.data[modified_model.new_token_start:modified_model.new_token_end] = normed_weights


def merge_rmsnorm_weights_to_hf_model(modified_model, hf_model):
    lm_head_weights = modified_model.base_model.lm_head.weight
    normed_weights = modified_model.rmsnorm(lm_head_weights[modified_model.new_token_start:modified_model.new_token_end])
    with torch.no_grad():
        hf_model.lm_head.weight.data[modified_model.new_token_start:modified_model.new_token_end] = normed_weights
