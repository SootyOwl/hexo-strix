import os
import sys
import subprocess

import pytest


def test_load_native_model_no_torch():
    path = os.environ.get("HEXO_REAL_WEIGHTS")
    if not path:
        pytest.skip("HEXO_REAL_WEIGHTS unset")

    from hexo_a0.serving.model import load_native_model

    model, mc, meta = load_native_model(path)

    assert model._hexo_native is not None
    assert mc.graph_type == "axis"
    assert isinstance(meta, dict)
    assert "model_config" in meta

    code = (
        "import sys\n"
        "from hexo_a0.serving.model import load_native_model\n"
        f"load_native_model({path!r})\n"
        "assert 'hexo_a0.model' not in sys.modules\n"
        "assert 'torch' not in sys.modules\n"
        "print('CLEAN')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "CLEAN" in result.stdout
