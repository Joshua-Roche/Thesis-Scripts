"""A bounded undo/redo history for segmentation edits.
"""

from numpy import packbits, stack, unpackbits


def _pack(slices):
    """Bit-pack a list of 2D boolean slices into a compact snapshot."""
    volume = stack(slices, axis=0).astype(bool)
    return volume.shape, packbits(volume)


def _unpack(snapshot):
    """Rebuild the list of 2D boolean slices from a snapshot."""
    shape, packed = snapshot
    count = shape[0] * shape[1] * shape[2]
    volume = unpackbits(packed, count=count).astype(bool).reshape(shape)
    return [volume[i] for i in range(shape[0])]


class UndoHistory:
    """Bounded undo/redo stacks scoped to a single segmentation at a time.

    ``push`` records the current state before an edit. ``undo`` and ``redo``
    take the current state as an argument and return the state to restore, or
    ``None`` when there is nothing to do.
    """

    def __init__(self, max_depth: int = 16):
        self.max_depth = max_depth
        self._undo = []
        self._redo = []

    def clear(self):
        self._undo.clear()
        self._redo.clear()

    def push(self, slices):
        """Record ``slices`` as an undo point, discarding any redo history."""
        self._undo.append(_pack(slices))
        if len(self._undo) > self.max_depth:
            self._undo.pop(0)
        self._redo.clear()

    def can_undo(self):
        return bool(self._undo)

    def can_redo(self):
        return bool(self._redo)

    def undo(self, current_slices):
        """Return the previous state, moving ``current_slices`` onto the redo stack."""
        if not self._undo:
            return None
        self._redo.append(_pack(current_slices))
        return _unpack(self._undo.pop())

    def redo(self, current_slices):
        """Return the next state, moving ``current_slices`` back onto the undo stack."""
        if not self._redo:
            return None
        self._undo.append(_pack(current_slices))
        return _unpack(self._redo.pop())
