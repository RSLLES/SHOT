# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Utilities for logging; writing and path handling."""

import os
import re


def get_log_dir(name: str, root_dir: str = "outputs/", prefix: str = "v") -> str:
    """Build a log directory path for the specified model name."""
    log_dir = os.path.join(root_dir, name)
    next_dir = f"{prefix}{_get_next_version(log_dir, prefix=prefix)}"
    log_dir = os.path.join(log_dir, next_dir)
    return log_dir


def _get_next_version(root_path: str, prefix: str) -> int:
    """Return the next integer version number for the given prefix."""
    if not os.path.exists(root_path):
        return 0
    last = -1
    pattern = rf"\b{re.escape(prefix)}(\d+)\b"
    dirs = os.listdir(root_path)
    for dir in dirs:
        matches = re.findall(pattern, dir)
        if not matches:
            continue
        version = int(matches[0])
        last = version if version > last else last
    next = last + 1
    return next


def write_file(content: str, filename: str, log_dir: str) -> None:
    """Write the given content to a file inside the specified log directory."""
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, filename)
    with open(path, "w") as f:
        f.write(content)
