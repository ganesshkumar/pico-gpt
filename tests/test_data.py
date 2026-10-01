import unittest

import torch

from pico_gpt.data import build_dataset


class DatasetTests(unittest.TestCase):
  def setUp(self):
    self.device = torch.device("cpu")
    self.text = "abcdefghijklmnopqrstuvwxyz\n" * 20
    self.characters, self.encode, self.decode, self.get_batch = build_dataset(
      self.text,
      block_size=8,
      device=self.device
    )

  def test_encode_decode_round_trip(self):
    value = "tinygpt\n"
    self.assertEqual(self.decode(self.encode(value)), value)

  def test_vocabulary_contains_unique_sorted_characters(self):
    self.assertEqual(self.characters, sorted(set(self.text)))

  def test_batch_has_expected_shape_and_shifted_targets(self):
    inputs, targets = self.get_batch("training", batch_size=4)

    self.assertEqual(inputs.shape, (4, 8))
    self.assertEqual(targets.shape, (4, 8))
    self.assertEqual(inputs.device, self.device)
    self.assertTrue(torch.equal(inputs[:, 1:], targets[:, :-1]))

  def test_unknown_split_is_rejected(self):
    with self.assertRaisesRegex(ValueError, "Unknown split"):
      self.get_batch("test", batch_size=1)

  def test_short_split_is_rejected(self):
    _, _, _, get_batch = build_dataset("abcdefghij", block_size=4, device=self.device)

    with self.assertRaisesRegex(ValueError, "validation split"):
      get_batch("validation", batch_size=1)


if __name__ == "__main__":
  unittest.main()
