"""TensorBoard writer wrapper that supports per-scalar descriptions.

Wraps ``torch.utils.tensorboard.SummaryWriter`` and attaches a
``SummaryMetadata`` proto (built via the scalar plugin's canonical
``create_summary_metadata`` helper) on every write of a described tag.
TensorBoard renders ``summary_description`` as an info card on the
scalar chart, useful for documenting non-obvious metrics.
"""

from __future__ import annotations

from typing import Any

from tensorboard.compat.proto.summary_pb2 import Summary
from tensorboard.plugins.scalar.metadata import create_summary_metadata


class DescribedWriter:
    """Wrap a ``SummaryWriter`` so ``add_scalar`` accepts a ``description``.

    The description is remembered per tag: once set, every subsequent
    ``add_scalar`` call for that tag is emitted directly via the underlying
    file writer with a properly-populated ``SummaryMetadata`` proto attached.
    This is required because falling through to ``SummaryWriter.add_scalar``
    writes a vanilla Summary with no plugin metadata, which TensorBoard's
    scalar dashboard will index in preference to the first (described) event.

    Tags without a description fall through to the wrapped writer normally.
    Forwards all other attribute access to the wrapped writer.
    """

    def __init__(self, writer):
        self._writer = writer
        self._described: dict[str, str] = {}

    def add_scalar(
        self,
        tag: str,
        scalar_value,
        global_step=None,
        walltime=None,
        *,
        description: str | None = None,
    ) -> None:
        if description is not None:
            self._described[tag] = description

        desc = self._described.get(tag)
        if desc is None:
            self._writer.add_scalar(tag, scalar_value, global_step, walltime)
            return

        metadata = create_summary_metadata(display_name=tag, description=desc)
        summary = Summary(
            value=[
                Summary.Value(
                    tag=tag,
                    simple_value=float(scalar_value),
                    metadata=metadata,
                )
            ]
        )
        file_writer = self._writer._get_file_writer()
        file_writer.add_summary(summary, global_step=global_step)

    def __getattr__(self, name: str) -> Any:
        # Delegate everything else (add_text, add_histogram, flush, close, etc.)
        return getattr(self._writer, name)
