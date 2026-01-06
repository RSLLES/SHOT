# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

import smlmshot


def test_insert_matrice():
    """Test smlmshot.simulation.render_one.batched_insert_2d."""
    output = torch.zeros((2, 4, 5), dtype=torch.int)
    idx = torch.tensor([[0, 1], [1, 3]], dtype=torch.int)
    to_insert = torch.arange(8, dtype=torch.int).view(2, 2, 2)
    expected_output = torch.tensor(
        [
            [
                [0, 0, 1, 0, 0],
                [0, 2, 3, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 4, 5],
                [0, 0, 0, 6, 7],
                [0, 0, 0, 0, 0],
            ],
        ],
        dtype=torch.int,
    )

    smlmshot.a.render_one.batched_insert_2d(to_insert, idx[:, 0], idx[:, 1], output)

    assert torch.allclose(output, expected_output)


def test_cubic_3d_power_series():
    """Test smlmshot.simulation.render_one._cubic_3d_power_series."""
    u = torch.linspace(0, 1, 50 * 3).view(50, 3)
    output = smlmshot.a.render_one._cubic_3d_power_series(u)
    output = smlmshot.utils.torch.hash_tensor(output)
    expected_output = -163858642767066781
    assert output == expected_output
