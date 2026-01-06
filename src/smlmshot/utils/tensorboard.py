# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Utility functions for TensorBoard event files."""

import os
import re

import pandas as pd
from tensorboard.backend.event_processing import event_accumulator


def find_events_file(dirpath: str):
    """Return the path to the single events.out.tfevents.* file in dirpath."""
    pattern = re.compile(r"^events\.out\.tfevents\..*")
    matches = [f for f in os.listdir(dirpath) if pattern.match(f)]

    if len(matches) == 0:
        raise FileNotFoundError("No events.out.tfevents.* file found.")
    elif len(matches) > 1:
        raise RuntimeError("Multiple events.out.tfevents.* files found.")

    return os.path.join(dirpath, matches[0])


def load_events_file(filepath: str) -> pd.DataFrame:
    """Load scalar tags from a TensorBoard event file into a DataFrame."""
    ea = event_accumulator.EventAccumulator(filepath)
    ea.Reload()

    scalar_tags = ea.Tags().get("scalars", [])
    if not scalar_tags:
        raise ValueError("No scalar tags found in the event file.")

    data = {}
    for tag in scalar_tags:
        events = ea.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]
        data[tag] = pd.Series(values, index=steps)

    df = pd.DataFrame(data)
    df.index.name = "step"
    df.sort_index(inplace=True)
    return df
