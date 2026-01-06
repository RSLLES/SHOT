# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Utility functions for formatting objects to strings."""

import re
import warnings
from math import isinf, isnan

import numpy as np
import torch
from sigfig import round
from torch import Tensor


def format_metrics(metrics: dict) -> str:
    """Format metrics in a text-logging format."""
    return "; ".join(f"{name}:{format_object(x)}" for name, x in metrics.items())


def format_object(x: any) -> str:
    """Format all sorts of objects to strings."""
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    if isinstance(x, Tensor):
        if x.nelement() > 1:
            strings = [format_object(e) for e in x]
            return "[" + ",".join(strings) + "]"
        x = x.item()
    if isinstance(x, float) or isinstance(x, int):
        return format_number(x)
    return f" {x}"  # other types are printed as they are...


def format_number(x: float | int, n_digits: int = 3) -> str:
    """Format a number with a given number os significant digits."""
    max_characters = n_digits + 5  # ., e, +-, 1 digit in mantissa
    with warnings.catch_warnings():
        # sigfig warns when x has less than n_digits
        warnings.filterwarnings("ignore", category=UserWarning)
        if isinstance(x, float) and not isnan(x) and not isinf(x):
            s = round(x, sigfigs=n_digits, type=str)
        else:
            s = str(x)
        s = (
            round(x, sigfigs=n_digits, type=str, notation="sci")
            if len(s) >= max_characters
            else s
        )
    if s[0] != "-":
        s = f" {s}"
    return s


def format_string(s: str) -> str:
    """Format a string to lowercase with only alphanumeric characters."""
    s = s.lower().replace(" ", "_")
    s = re.sub(r"[^a-z0-9_]", "", s)
    return s


def format_size(size: int, n_digits: int = 3) -> str:
    """Format a number of bytes into a readable string with size unit (KiB, ...)."""
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    for unit in units:
        if size < 1024:
            break
        size /= 1024
    with warnings.catch_warnings():  # sigfig warns when x has less than n_digits
        warnings.filterwarnings("ignore", category=UserWarning)
        return round(size, sigfigs=n_digits, type=str) + unit
