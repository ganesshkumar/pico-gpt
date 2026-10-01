import torch

if __package__:
  from .data import DEFAULT_TEXT, build_dataset
  from .model import PicoGPT, PicoGPTConfig
  from .training import train_model
else:
  from data import DEFAULT_TEXT, build_dataset
  from model import PicoGPT, PicoGPTConfig
  from training import train_model


def print_section(title):
  print(f"\n--- {title} ---")


def choose_device():
  if torch.cuda.is_available():
    return torch.device("cuda")
  if torch.backends.mps.is_available():
    return torch.device("mps")
  return torch.device("cpu")


def main():
  torch.manual_seed(1337)
  device = choose_device()

  print(f"Using device: {device}")

  text = DEFAULT_TEXT * 200
  config = PicoGPTConfig(
    block_size=64,
    n_embd=96,
    n_head=4,
    n_layer=2
  )

  characters, encode, decode, get_batch = build_dataset(text, config.block_size, device)
  config.vocab_size = len(characters)

  print_section("Tokenizer")
  print("vocabulary size:", config.vocab_size)
  print("characters:", repr("".join(characters)))

  model = PicoGPT(config).to(device)
  parameter_count = sum(parameter.numel() for parameter in model.parameters())

  print_section("Model")
  print(config)
  print("parameters:", f"{parameter_count:,}")
  print("token embedding shape:", tuple(model.token_embedding.weight.shape))
  print("position embedding shape:", tuple(model.position_embedding.weight.shape))
  print("weight tying:", model.lm_head.weight.data_ptr() == model.token_embedding.weight.data_ptr())

  inputs, targets = get_batch("training", batch_size=4)

  print_section("One training example")
  print("Input:")
  print(repr(decode(inputs[0].tolist())))
  print("Target:")
  print(repr(decode(targets[0].tolist())))
  print("input shape:", tuple(inputs.shape))

  # Check the model's output before it has learned anything.
  model.eval()
  with torch.no_grad():
    logits, loss = model(inputs, targets)

  print_section("Forward pass")
  print("logits shape:", tuple(logits.shape))
  print("loss:", loss.item())
  print("shape order: [batch, sequence, vocabulary]")

  prompt = "the "
  prompt_ids = torch.tensor([encode(prompt)], dtype=torch.long, device=device)

  print_section("Before training")
  generated = model.generate(prompt_ids, max_new_tokens=150, temperature=1.0, top_k=10)
  print(decode(generated[0].tolist()))

  optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

  print_section("Training")
  train_model(
    model,
    get_batch,
    optimizer,
    max_steps=1200,
    batch_size=32,
    eval_interval=150,
    eval_iters=30
  )

  model.eval()
  torch.manual_seed(42)

  print_section("After training - sampling")
  generated = model.generate(prompt_ids, max_new_tokens=300, temperature=0.8, top_k=10)
  print(decode(generated[0].tolist()))

  print_section("After training - greedy")
  generated = model.generate(prompt_ids, max_new_tokens=150, temperature=0)
  print(decode(generated[0].tolist()))


if __name__ == "__main__":
  main()
