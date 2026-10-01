import torch


# A small built-in corpus so the example runs without an input file.
DEFAULT_TEXT = """
the cat sat on the mat.
the dog sat by the door.
the fox ran through the field.
the bird flew over the tree.
the cat chased the mouse.
the dog chased the ball.
the mouse hid under the table.
the bird sang in the morning.
the sun rose over the hill.
the moon shone over the lake.
the child walked through the garden.
the farmer worked in the field.
the cat slept near the window.
the dog slept beside the fire.
the bird rested in the tree.
the fox watched the quiet field.
"""


def build_dataset(text, block_size, device):
  characters = sorted(set(text))
  char_to_id = {char: index for index, char in enumerate(characters)}
  id_to_char = {index: char for index, char in enumerate(characters)}

  def encode(value):
    return [char_to_id[char] for char in value]

  def decode(ids):
    return "".join(id_to_char[index] for index in ids)

  data = torch.tensor(encode(text), dtype=torch.long)
  split = int(0.9 * len(data))
  training_data = data[:split]
  validation_data = data[split:]

  def get_batch(split_name, batch_size):
    if split_name not in {"training", "validation"}:
      raise ValueError(f"Unknown split: {split_name}")

    source = training_data if split_name == "training" else validation_data
    if len(source) <= block_size:
      raise ValueError(
        f"{split_name} split needs more than {block_size} characters to create a batch"
      )

    starts = torch.randint(0, len(source) - block_size - 1, (batch_size,))
    inputs = torch.stack([source[start:start + block_size] for start in starts])
    targets = torch.stack([source[start + 1:start + block_size + 1] for start in starts])
    return inputs.to(device), targets.to(device)

  return characters, encode, decode, get_batch
