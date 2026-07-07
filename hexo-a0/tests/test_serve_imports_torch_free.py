"""Guard test: ``hexo_a0.cli`` and ``hexo_a0.serving.app`` must import
without pulling in torch/torch_geometric at module scope.

The serve path (native/safetensors inference) should not require torch to be
installed at all; torch is only needed by the train/watch/eval subcommands,
which import it lazily inside their own functions.

Runs in a subprocess with a clean interpreter so the block is enforced from
the very first import, before anything else has a chance to cache torch in
``sys.modules``.
"""
import subprocess
import sys

CODE = """
import sys

BLOCKED = ("torch", "torch_geometric")


class _BlockFinder:
    def find_spec(self, name, path=None, target=None):
        if name in BLOCKED or any(
            name.startswith(prefix + ".") for prefix in BLOCKED
        ):
            raise ImportError(f"blocked: {name}")
        return None


sys.meta_path.insert(0, _BlockFinder())

import hexo_a0.cli
import hexo_a0.serving.app

print("IMPORTS_OK")
assert "torch" not in sys.modules
"""


def test_cli_and_serve_app_import_without_torch():
    result = subprocess.run(
        [sys.executable, "-c", CODE], capture_output=True, text=True
    )
    assert result.returncode == 0, (
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "IMPORTS_OK" in result.stdout
