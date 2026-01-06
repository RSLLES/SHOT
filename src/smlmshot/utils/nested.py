# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Utility functions to deal with nested tensors (padding/expanding)."""

import torch
from torch import Tensor


@torch.compiler.disable()
def pad_sequence(
    x: list[Tensor],
    target_len: int = None,
    padding_value: float = 0.0,
    strict: bool = True,
    returns_lengths: bool = False,
) -> tuple[Tensor, Tensor]:
    """Pad a list of tensors to a uniform length."""
    device, dtype = x[0].device, x[0].dtype

    if not strict:
        x = [e[:target_len] for e in x]

    if returns_lengths:
        lengths = [e.size(0) for e in x]
        lengths = torch.tensor(lengths, device=device, dtype=torch.int)

    padded = torch.nn.utils.rnn.pad_sequence(
        x, batch_first=True, padding_value=padding_value
    )

    if target_len is None:
        out = (padded, lengths) if returns_lengths else padded
        return out

    delta_len = target_len - padded.size(1)
    if delta_len < 0 and strict:
        raise ValueError(f"Sequence length exceeds max_len={target_len}")
    if delta_len > 0:
        padding_value = torch.full((1,), padding_value, device=device, dtype=dtype)
        pad_size = list(padded.shape)
        pad_size[1] = delta_len
        pad = padding_value.expand(*pad_size)
        padded = torch.cat([padded, pad], dim=1)

    out = (padded, lengths) if returns_lengths else padded
    return out


def expand_to_list(x: Tensor, lengths: Tensor) -> list[Tensor]:
    """Slice a padded tensor back into a list using the given lengths."""
    x = [e[:i] for e, i in zip(x, lengths)]
    return x
