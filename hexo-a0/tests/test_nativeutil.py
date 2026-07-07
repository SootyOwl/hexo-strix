import math

import pytest


def test_softmax_matches_reference():
    from hexo_a0.serving.nativeutil import softmax

    assert softmax([]) == []
    assert abs(sum(softmax([1.0, 2.0, 3.0])) - 1.0) < 1e-12
    assert softmax([1000.0, 1000.0]) == [0.5, 0.5]
    assert softmax([0.0, 0.0, 0.0]) == pytest.approx([1 / 3, 1 / 3, 1 / 3], abs=1e-12)

    result = softmax([0.0, math.log(2), math.log(3)])
    assert result == pytest.approx([1 / 6, 2 / 6, 3 / 6], abs=1e-9)


def test_native_model_call_raises():
    from hexo_a0.serving.nativeutil import NativeModel

    sentinel = object()
    nm = NativeModel(sentinel)
    assert nm._hexo_native is sentinel
    with pytest.raises(RuntimeError):
        nm()
