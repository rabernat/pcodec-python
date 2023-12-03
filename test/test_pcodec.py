import numpy as np
from pcodec import compress, decompress
import pytest

@pytest.mark.parametrize(
    "data",
    [
        np.arange(0, 100, 1000)
    ]
)
@pytest.mark.parametrize("dtype", ['f4', 'f8'])
def test_round_trip(data, dtype):
    data = data.astype(dtype)
    compressed = compress(data)
    out = np.empty_like(data)
    decompress(compressed, out)
    np.testing.assert_array_equal(data, out)
