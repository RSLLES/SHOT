# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Extension of MeanMetric to dictionnary."""

from collections import defaultdict

from torchmetrics import MeanMetric


class MeanDictMetric:
    """Aggregate multiple MeanMetric instances by key."""

    def __init__(self, device=None, dtype=None):
        self.metrics = defaultdict(lambda: MeanMetric().to(device=device, dtype=dtype))

    def update(self, metrics_dict: dict):  # noqa: D102
        for key, value in metrics_dict.items():
            self.metrics[key].update(value)

    def compute(self) -> dict:  # noqa: D102
        return {k: m.compute() for k, m in self.metrics.items()}

    def reset(self):  # noqa: D102
        for metric in self.metrics.values():
            metric.reset()
