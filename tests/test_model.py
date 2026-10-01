import unittest

import torch

from pico_gpt.model import PicoGPT, PicoGPTConfig


def make_model():
  config = PicoGPTConfig(
    block_size=8,
    n_embd=12,
    n_layer=2,
    n_head=3,
    vocab_size=16
  )
  return PicoGPT(config)


class ModelTests(unittest.TestCase):
  def setUp(self):
    torch.manual_seed(42)
    self.model = make_model().eval()

  def test_forward_returns_logits_and_loss(self):
    inputs = torch.randint(0, 16, (2, 5))
    targets = torch.randint(0, 16, (2, 5))

    logits, loss = self.model(inputs, targets)

    self.assertEqual(logits.shape, (2, 5, 16))
    self.assertEqual(loss.ndim, 0)

  def test_forward_without_targets_has_no_loss(self):
    logits, loss = self.model(torch.randint(0, 16, (2, 5)))

    self.assertEqual(logits.shape, (2, 5, 16))
    self.assertIsNone(loss)

  def test_sequence_cannot_exceed_block_size(self):
    inputs = torch.randint(0, 16, (1, 9))

    with self.assertRaisesRegex(ValueError, "exceeds model block size"):
      self.model(inputs)

  def test_embedding_and_output_weights_are_tied(self):
    self.assertEqual(
      self.model.token_embedding.weight.data_ptr(),
      self.model.lm_head.weight.data_ptr()
    )

  def test_future_token_does_not_change_earlier_logits(self):
    first = torch.tensor([[1, 2, 3, 4]])
    second = torch.tensor([[1, 2, 3, 9]])

    first_logits, _ = self.model(first)
    second_logits, _ = self.model(second)

    self.assertTrue(torch.allclose(first_logits[:, :-1], second_logits[:, :-1]))

  def test_embedding_size_must_be_divisible_by_heads(self):
    config = PicoGPTConfig(n_embd=10, n_head=3, vocab_size=16)

    with self.assertRaisesRegex(AssertionError, "divisible"):
      PicoGPT(config)


class GenerationTests(unittest.TestCase):
  def setUp(self):
    torch.manual_seed(42)
    self.model = make_model().eval()
    self.prompt = torch.tensor([[1, 2, 3]])

  def test_generation_adds_requested_number_of_tokens(self):
    generated = self.model.generate(self.prompt, max_new_tokens=5, temperature=0)

    self.assertEqual(generated.shape, (1, 8))
    self.assertTrue(torch.equal(generated[:, :3], self.prompt))

  def test_zero_tokens_returns_original_input(self):
    generated = self.model.generate(self.prompt, max_new_tokens=0)

    self.assertTrue(torch.equal(generated, self.prompt))

  def test_greedy_generation_is_deterministic(self):
    first = self.model.generate(self.prompt, max_new_tokens=4, temperature=0)
    second = self.model.generate(self.prompt, max_new_tokens=4, temperature=0)

    self.assertTrue(torch.equal(first, second))

  def test_seeded_sampling_is_reproducible(self):
    torch.manual_seed(7)
    first = self.model.generate(self.prompt, max_new_tokens=4, temperature=0.8, top_k=4)
    torch.manual_seed(7)
    second = self.model.generate(self.prompt, max_new_tokens=4, temperature=0.8, top_k=4)

    self.assertTrue(torch.equal(first, second))

  def test_invalid_sampling_values_are_rejected(self):
    with self.assertRaisesRegex(ValueError, "Temperature"):
      self.model.generate(self.prompt, max_new_tokens=1, temperature=-0.1)

    with self.assertRaisesRegex(ValueError, "top_k"):
      self.model.generate(self.prompt, max_new_tokens=1, top_k=0)


if __name__ == "__main__":
  unittest.main()
