"""Vectorised brush stamping.
"""

from numpy import arange


def stamp_brush(slices, cx, cy, cz, radius, adding, three_d=False, z_reach=2):
    """Paint (or erase) a sphere of radius ``radius`` centred on ``(cx, cy, cz)``.

    ``slices`` is the list of 2D boolean arrays making up one segmentation;
    they are modified in place. ``adding`` selects paint (True) vs erase
    (False). When ``three_d`` is false only the centre slice is touched — a
    flat disk — matching the 2D brush; otherwise the sphere spans up to
    ``z_reach`` slices either side.

    The result is exactly the set of voxels satisfying
    ``dx^2 + dy^2 + dz^2 <= radius^2`` that fall inside the volume.
    """
    if not slices:
        return
    depth = len(slices)
    height, width = slices[0].shape

    dz_values = range(-z_reach, z_reach + 1) if three_d else (0,)
    radius_sq = radius * radius

    for dz in dz_values:
        nz = cz + dz
        if not 0 <= nz < depth:
            continue

        # In-plane radius shrinks as we move away from the centre slice; once
        # dz exceeds the radius the sphere no longer reaches this slice.
        remaining_sq = radius_sq - dz * dz
        if remaining_sq < 0:
            continue

        # Only the bounding box of the disk needs to be considered.
        y0 = max(0, cy - radius)
        y1 = min(height, cy + radius + 1)
        x0 = max(0, cx - radius)
        x1 = min(width, cx + radius + 1)
        if y0 >= y1 or x0 >= x1:
            continue

        ys = arange(y0, y1) - cy
        xs = arange(x0, x1) - cx
        disk = (ys[:, None] ** 2 + xs[None, :] ** 2) <= remaining_sq

        # A slice view; assigning through it writes back into the volume.
        window = slices[nz][y0:y1, x0:x1]
        window[disk] = adding
