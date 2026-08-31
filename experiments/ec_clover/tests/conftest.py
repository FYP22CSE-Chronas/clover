from __future__ import annotations

import numpy as np
import pytest

N_BOTTOM = 4


@pytest.fixture
def S() -> np.ndarray:
    """[7, 4] aggregation matrix: total, two pairs, then the identity block."""
    A = np.array(
        [[1, 1, 1, 1], [1, 1, 0, 0], [0, 0, 1, 1]],
        dtype=np.float32,
    )
    return np.vstack([A, np.eye(N_BOTTOM, dtype=np.float32)])
