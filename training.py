import torch


@torch.no_grad()
def estimate_loss(model, get_batch, batch_size, eval_iters):
  result = {}
  model.eval()

  for split in ["training", "validation"]:
    losses = torch.zeros(eval_iters)
    for index in range(eval_iters):
      inputs, targets = get_batch(split, batch_size)
      _, loss = model(inputs, targets)
      losses[index] = loss.item()
    result[split] = losses.mean().item()

  model.train()
  return result


def train_model(model, get_batch, optimizer, max_steps, batch_size, eval_interval, eval_iters):
  model.train()

  for step in range(max_steps + 1):
    # Average a few batches so the reported loss is less noisy.
    if step % eval_interval == 0 or step == max_steps:
      losses = estimate_loss(model, get_batch, batch_size, eval_iters)
      print(
        f"step {step:4d} | "
        f"train {losses['training']:.4f} | "
        f"validation {losses['validation']:.4f}"
      )

    if step == max_steps:
      break

    inputs, targets = get_batch("training", batch_size)
    _, loss = model(inputs, targets)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
