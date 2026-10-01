import io
import unittest
from contextlib import redirect_stdout

import torch

from model import PicoGPT, PicoGPTConfig
from training import estimate_loss, train_model


class TrainingTests(unittest.TestCase):
  def setUp(self):
    torch.manual_seed(42)
    config = PicoGPTConfig(
      block_size=4,
      n_embd=8,
      n_layer=1,
      n_head=2,
      vocab_size=8
    )
    self.model = PicoGPT(config)

  @staticmethod
  def get_batch(split_name, batch_size):
    offset = 0 if split_name == "training" else 1
    row = (torch.arange(5) + offset) % 8
    inputs = row[:-1].repeat(batch_size, 1)
    targets = row[1:].repeat(batch_size, 1)
    return inputs, targets

  def test_estimate_loss_returns_both_splits_and_restores_training_mode(self):
    self.model.train()

    losses = estimate_loss(self.model, self.get_batch, batch_size=2, eval_iters=2)

    self.assertEqual(set(losses), {"training", "validation"})
    self.assertGreater(losses["training"], 0)
    self.assertGreater(losses["validation"], 0)
    self.assertTrue(self.model.training)

  def test_training_step_updates_parameters(self):
    optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-2)
    before = self.model.token_embedding.weight.detach().clone()

    with redirect_stdout(io.StringIO()):
      train_model(
        self.model,
        self.get_batch,
        optimizer,
        max_steps=1,
        batch_size=2,
        eval_interval=1,
        eval_iters=1
      )

    after = self.model.token_embedding.weight.detach()
    self.assertFalse(torch.equal(before, after))


if __name__ == "__main__":
  unittest.main()
