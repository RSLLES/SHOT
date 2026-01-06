# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Utility functions for git."""

import subprocess


def get_head_commit_hash(repo_path: str = ".") -> str:
    """Return the hash of the HEAD commit for the given repository."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return "Error while trying to get last commit hash."
