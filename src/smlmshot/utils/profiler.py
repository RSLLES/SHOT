# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Utility functions to profile models."""

import os
from datetime import datetime

import torch


class Profiler:
    """Wrapper around torch.profiler with auto-saving and default scheduling."""

    def __init__(
        self, dirpath="traces/", disable: bool = False, exit_when_done: bool = True
    ):
        self.disable = disable
        self.exit_when_done = exit_when_done
        self.dirpath = dirpath
        self.prof = None

        if not self.disable:
            self.prof = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                schedule=torch.profiler.schedule(
                    wait=10, warmup=10, active=5, repeat=1
                ),
                record_shapes=True,
                with_stack=True,
                on_trace_ready=self._trace_handler,
            )

    def _trace_handler(self, p):
        """Handle to match torch.profiler's signature expectation."""
        os.makedirs(self.dirpath, exist_ok=True)
        filename = datetime.now().strftime("%FT%X") + ".json"
        filepath = os.path.join(self.dirpath, filename)

        output = p.key_averages(group_by_stack_n=5).table(
            sort_by="self_cuda_time_total", row_limit=20
        )
        # output = p.key_averages().table(sort_by="cuda_time_total", row_limit=20)
        print(output)
        p.export_chrome_trace(filepath)
        print(f"Profiler trace saved to {filepath}")
        if self.exit_when_done:
            print("Profiling complete, exiting.")
            exit()

    def step(self):
        """Advance the profiler step."""
        if not self.disable and self.prof:
            self.prof.step()

    def __enter__(self):
        if self.disable:
            return self
        self.prof.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.disable:
            return
        self.prof.__exit__(exc_type, exc_value, traceback)


class MemoryProfiler:
    """Profile GPU memory usage and dump a snapshot.

    Snapshot can be visualized with https://docs.pytorch.org/memory_viz
    """

    def __init__(
        self, prefix="memory", disable: bool = False, exit_on_done: bool = True
    ):
        """Set up profiler with a prefix and options."""
        timestamp = datetime.now().strftime("%FT%X")
        self.filename = f"{prefix}_{timestamp}.pickle"
        self.disable = disable
        self.exit_on_done = exit_on_done

    def __enter__(self):
        """Start recording GPU memory history."""
        if self.disable:
            return
        torch.cuda.memory._record_memory_history(max_entries=100_000)

    def __exit__(self, exc_type, exc_value, traceback):
        """Dump snapshot and stop recording; exit program if requested."""
        if self.disable:
            return
        torch.cuda.memory._dump_snapshot(self.filename)
        torch.cuda.memory._record_memory_history(enabled=None)
        if self.exit_on_done:
            exit()
