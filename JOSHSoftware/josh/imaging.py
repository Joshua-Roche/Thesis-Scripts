"""Turning Hounsfield-unit slices into displayable 8-bit images."""

from numpy import clip, float32, ndarray, uint8


def window_to_uint8(slice_hu: ndarray, hu_min: int, hu_max: int) -> ndarray:
    """Map a HU slice into 0-255 using a window/level range.

    This is the standard radiological "windowing" operation: everything below
    ``hu_min`` goes black, everything above ``hu_max`` goes white, and the
    range between is stretched across the full 8-bit scale. Narrow windows are
    what make soft tissue visible in a scan that also contains bone and air.
    """
    if hu_max <= hu_min:
        hu_max = hu_min + 1  # avoid dividing by zero on a degenerate window
    # Cast to float first: ``255 * (windowed - hu_min)`` overflows silently on
    # an integer input dtype (350 * 255 does not fit in int16).
    windowed = clip(slice_hu, hu_min, hu_max).astype(float32)
    return (255 * (windowed - hu_min) / (hu_max - hu_min)).astype(uint8)
