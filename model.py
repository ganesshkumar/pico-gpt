import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PicoGPTConfig:
  block_size: int = 64
  n_embd: int = 96
  n_layer: int = 2
  n_head: int = 4
  vocab_size: int = 0


class PicoCausalSelfAttention(nn.Module):
  def __init__(self, config: PicoGPTConfig):
    super().__init__()

    assert config.n_embd % config.n_head == 0, "Embedding dimension must be divisible by number of heads"

    self.n_embd = config.n_embd
    self.n_head = config.n_head
    self.head_dim = config.n_embd // config.n_head

    self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd)
    self.proj = nn.Linear(config.n_embd, config.n_embd)

    mask = torch.tril(torch.ones(config.block_size, config.block_size))
    self.register_buffer("mask", mask.view(1, 1, config.block_size, config.block_size))

  def forward(self, x):
    batch_size, sequence_length, embedding_size = x.shape
    qkv = self.qkv(x)
    q, k, v = qkv.split(self.n_embd, dim=-1)

    # Split the embedding into separate attention heads.
    q = q.view(batch_size, sequence_length, self.n_head, self.head_dim).transpose(1, 2)
    k = k.view(batch_size, sequence_length, self.n_head, self.head_dim).transpose(1, 2)
    v = v.view(batch_size, sequence_length, self.n_head, self.head_dim).transpose(1, 2)

    scores = q @ k.transpose(-2, -1)
    scores = scores / math.sqrt(self.head_dim)
    mask = self.mask[:, :, :sequence_length, :sequence_length]
    scores = scores.masked_fill(mask == 0, float("-inf"))
    attention_weights = F.softmax(scores, dim=-1)

    out = attention_weights @ v
    out = out.transpose(1, 2).contiguous()
    out = out.view(batch_size, sequence_length, embedding_size)
    out = self.proj(out)
    return out


class PicoMLP(nn.Module):
  def __init__(self, config: PicoGPTConfig):
    super().__init__()
    hidden = 4 * config.n_embd
    self.fc = nn.Linear(config.n_embd, hidden)
    self.gelu = nn.GELU(approximate="tanh")
    self.proj = nn.Linear(hidden, config.n_embd)

  def forward(self, x):
    x = self.fc(x)
    x = self.gelu(x)
    x = self.proj(x)
    return x


class PicoTransformerBlock(nn.Module):
  def __init__(self, config: PicoGPTConfig):
    super().__init__()
    self.ln1 = nn.LayerNorm(config.n_embd)
    self.attention = PicoCausalSelfAttention(config)
    self.ln2 = nn.LayerNorm(config.n_embd)
    self.mlp = PicoMLP(config)

  def forward(self, x):
    x = x + self.attention(self.ln1(x))
    x = x + self.mlp(self.ln2(x))
    return x


class PicoGPT(nn.Module):
  def __init__(self, config: PicoGPTConfig):
    super().__init__()
    self.config = config
    self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
    self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
    self.blocks = nn.Sequential(*[PicoTransformerBlock(config) for _ in range(config.n_layer)])
    self.ln_f = nn.LayerNorm(config.n_embd)
    self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    self.apply(self._init_weights)
    self.lm_head.weight = self.token_embedding.weight

  def _init_weights(self, module):
    if isinstance(module, nn.Linear):
      nn.init.normal_(module.weight, mean=0.0, std=0.02)
      if module.bias is not None:
        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
      nn.init.normal_(module.weight, mean=0.0, std=0.02)

  def forward(self, input_ids, targets=None):
    batch_size, sequence_length = input_ids.shape
    if sequence_length > self.config.block_size:
      raise ValueError(
        f"Sequence length {sequence_length} exceeds model block size {self.config.block_size}"
      )

    token_embeddings = self.token_embedding(input_ids)
    position_ids = torch.arange(sequence_length, device=input_ids.device)
    position_embeddings = self.position_embedding(position_ids)
    x = token_embeddings + position_embeddings

    for block in self.blocks:
      x = block(x)

    x = self.ln_f(x)
    logits = self.lm_head(x)

    loss = None
    if targets is not None:
      logits_flat = logits.view(batch_size * sequence_length, self.config.vocab_size)
      targets_flat = targets.reshape(batch_size * sequence_length)
      loss = F.cross_entropy(logits_flat, targets_flat)

    return logits, loss

  @torch.no_grad()
  def generate(self, input_ids, max_new_tokens, temperature=1.0, top_k=None):
    if temperature < 0:
      raise ValueError("Temperature cannot be negative")
    if top_k is not None and top_k <= 0:
      raise ValueError("top_k must be greater than zero")

    for _ in range(max_new_tokens):
      # The model can only use the most recent block of tokens.
      context = input_ids[:, -self.config.block_size:]
      logits, _ = self.forward(context)
      logits = logits[:, -1, :]

      if temperature == 0:
        next_token = torch.argmax(logits, dim=-1, keepdim=True)
      else:
        logits = logits / temperature
        if top_k is not None:
          sample_size = min(top_k, logits.size(-1))
          values, _ = torch.topk(logits, sample_size, dim=-1)
          threshold = values[:, -1, None]
          logits = logits.masked_fill(logits < threshold, float("-inf"))

        probabilities = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probabilities, num_samples=1)

      input_ids = torch.cat([input_ids, next_token], dim=1)

    return input_ids
