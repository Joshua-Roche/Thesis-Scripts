"""Bulk edits applied to a whole segmentation volume.
"""

import logging

from cv2 import (
    MORPH_CLOSE,
    MORPH_ELLIPSE,
    THRESH_BINARY,
    addWeighted,
    getStructuringElement,
    morphologyEx,
    threshold,
)
from numpy import any as np_any
from numpy import (
    array,
    bool_,
    copy,
    float32,
    percentile,
    stack,
    uint8,
    where,
)
from scipy.ndimage import gaussian_filter

logger = logging.getLogger(__name__)

# Fraction of annotated-region intensities kept when deciding which voxels an
# interpolated slice may claim.
LOWER_PERCENTILE = 2.5
UPPER_PERCENTILE = 97.5


def delete_left_half(segmentation_list):
    """Blank the left half of every slice."""
    volume = stack(segmentation_list, axis=0)
    volume[:, :, : volume.shape[2] // 2] = 0
    return [volume[i] for i in range(volume.shape[0])]


def delete_right_half(segmentation_list):
    """Blank the right half of every slice."""
    volume = stack(segmentation_list, axis=0)
    volume[:, :, volume.shape[2] // 2 :] = 0
    return [volume[i] for i in range(volume.shape[0])]


def _smooth_transitions_3d(volume, sigma=(3, 3, 3)):
    """Blur across slices so interpolated slices do not step abruptly.

    The blurred volume is OR-ed back into the original rather than replacing
    it, so smoothing can only grow the mask — it never erases anything the
    user drew by hand.
    """
    smoothed = gaussian_filter(copy(volume).astype(float), sigma=sigma)
    volume[:, :, :] += smoothed[:, :, :] > 0.5
    return volume > 0.5


def _interpolate_gaps(
    segmentation_volume,
    ct_volume,
    annotated_slices,
    lower_threshold,
    upper_threshold,
    struct_elem,
):
    """Fill every unannotated slice lying between annotated ones.

    Each gap slice is a linear blend of the nearest annotated slice above and
    below, masked down to voxels whose CT intensity falls inside the intensity
    range observed in the annotated regions, then morphologically closed.
    """
    interpolated = copy(segmentation_volume)

    for i in range(annotated_slices[0], annotated_slices[-1] + 1):
        if i in annotated_slices:
            continue

        previous = max(s for s in annotated_slices if s < i)
        following = min(s for s in annotated_slices if s > i)
        factor = (i - previous) / (following - previous)

        blended = addWeighted(
            segmentation_volume[previous].astype(float32),
            1 - factor,
            segmentation_volume[following].astype(float32),
            factor,
            0,
        )

        # Only keep blended voxels whose CT intensity looks like the tissue the
        # user actually annotated, so the mask does not bleed into air or bone.
        ct_slice = ct_volume[i].astype(float32)
        in_range = (ct_slice >= lower_threshold) & (ct_slice <= upper_threshold)
        blended = where(in_range, blended, 0)

        _, binary = threshold(blended, 0.5, 1, THRESH_BINARY)
        closed = morphologyEx(binary.astype(uint8), MORPH_CLOSE, struct_elem)
        interpolated[i] = closed.astype(bool_)

    return _smooth_transitions_3d(interpolated)


def interpolate_segmentation(segmentation_list, ct_scan_list):
    """Fill in the gaps between hand-annotated slices.

    Intended for the workflow where a user contours every Xth slice and lets
    the tool infer the rest. Returns the input unchanged if fewer than two
    slices are annotated, since there is then nothing to interpolate between.
    """
    segmentation_volume = stack(segmentation_list, axis=0)
    ct_volume = stack(ct_scan_list, axis=0)
    num_slices = segmentation_volume.shape[0]

    annotated_slices = [i for i in range(num_slices) if np_any(segmentation_volume[i])]
    if len(annotated_slices) < 2:
        logger.info("Not enough annotated slices to interpolate (need 2, have %d)", len(annotated_slices))
        return segmentation_list

    # Intensity range of what the user annotated, used to constrain the fill.
    annotated_values = []
    for z in annotated_slices:
        annotated_values.extend(ct_volume[z][segmentation_volume[z] > 0.5])
    annotated_values = array(annotated_values)

    lower_threshold = percentile(annotated_values, LOWER_PERCENTILE)
    upper_threshold = percentile(annotated_values, UPPER_PERCENTILE)

    struct_elem = getStructuringElement(MORPH_ELLIPSE, (5, 5))

    filled = _interpolate_gaps(
        segmentation_volume, ct_volume, annotated_slices,
        lower_threshold, upper_threshold, struct_elem,
    )
    return [filled[i] for i in range(filled.shape[0])]


def roi_statistics(segmentation_list, ct_scan_list, voxel_volume_mm3=0.0):
    """Summarise the CT intensities under a segmentation.

    ``voxel_volume_mm3`` is the physical volume of one voxel; pass 0 to omit the
    physical-volume figure. Returns a dict with the voxel count, physical
    volume in cm3, and the mean/min/max/standard-deviation HU inside the mask.
    Returns ``None`` if the mask is empty, since the statistics are undefined.
    """
    mask = stack(segmentation_list, axis=0) > 0.5
    voxel_count = int(mask.sum())
    if voxel_count == 0:
        return None

    values = stack(ct_scan_list, axis=0)[mask].astype(float32)
    return {
        "voxel_count": voxel_count,
        "volume_cm3": voxel_count * voxel_volume_mm3 / 1000.0,
        "mean_hu": float(values.mean()),
        "min_hu": float(values.min()),
        "max_hu": float(values.max()),
        "std_hu": float(values.std()),
    }
