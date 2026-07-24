"""Saving segmentations to disk and cleaning up model output.

Two formats are written side by side for every segmentation:

* ``segmentation_<series>-<name>.npy`` — the raw boolean volume, which the
  viewer reloads verbatim.
* an RTSTRUCT DICOM — contours, for anything else that reads DICOM.

The ``.npy`` is the source of truth; the RTSTRUCT is a lossy export because
contouring discards single-voxel detail.
"""

import logging
from pathlib import Path

import SimpleITK as sitk
from numpy import argmax, bincount, bool_, empty, equal, int32, ndarray, save, uint8, zeros_like
from rt_utils import RTStructBuilder
from scipy.ndimage import generate_binary_structure, label

logger = logging.getLogger(__name__)


def npy_filename(directory: Path, seg_name: str) -> Path:
    """Path the viewer expects for a segmentation called ``seg_name``.

    The series folder name is embedded so a mask stays identifiable if it is
    copied out of its scan folder.
    """
    directory = Path(directory)
    return directory / f"segmentation_{directory.name}-{seg_name}.npy"


def seg_name_from_npy(directory: Path, npy_path: Path) -> str:
    """Recover the segmentation name from a filename made by :func:`npy_filename`.

    Split on the full ``segmentation_<series>-`` prefix rather than on ``-``,
    because segmentation names themselves often contain hyphens.
    """
    directory = Path(directory)
    stem = Path(npy_path).stem
    return stem.split(f"segmentation_{directory.name}-")[-1]


def save_mask_npy(directory: Path, seg_name: str, mask: ndarray) -> Path:
    """Write ``mask`` as the ``.npy`` for ``seg_name`` and return the path."""
    path = npy_filename(directory, seg_name)
    save(path, mask)
    logger.info("Saved mask %s", path.name)
    return path


def save_rtstruct(directory: Path, roi_name: str, mask: ndarray, filename: str) -> Path | None:
    """Write ``mask`` as an RTSTRUCT alongside the DICOM series.

    ``mask`` must be boolean and indexed ``(row, column, slice)`` — the layout
    ``rt_utils`` expects, which is *not* the layout used by the ``.npy`` files.

    Returns the path written, or ``None`` if the export failed. Failure here is
    non-fatal by design: the ``.npy`` has already been saved, so the user's work
    is safe even if contouring chokes on an unusual series.
    """
    directory = Path(directory)
    try:
        rtstruct = RTStructBuilder.create_new(dicom_series_path=str(directory))
        rtstruct.add_roi(
            mask=mask,
            color=[255, 0, 0],
            name=roi_name,
            use_pin_hole=True,
            approximate_contours=False,
            roi_generation_algorithm=2,
        )
        output_path = directory / filename
        rtstruct.save(str(output_path))
        logger.info("Saved RTSTRUCT %s", output_path.name)
        return output_path
    except Exception as exc:
        logger.error("Could not write RTSTRUCT for %s: %s", directory, exc)
        return None


def save_mask_nifti(directory: Path, seg_name: str, mask: ndarray) -> Path | None:
    """Write ``mask`` as a ``.nii.gz`` beside the scan, with matching geometry.

    ``mask`` must be indexed ``(slice, row, column)`` in the original DICOM
    slice order — the same array that :func:`save_mask_npy` receives. The scan
    is re-read to copy its spacing, origin and orientation onto the mask so the
    NIfTI lines up with the source volume in other viewers.

    Returns the path written, or ``None`` on failure (non-fatal: the ``.npy``
    is the source of truth and has already been saved).
    """
    directory = Path(directory)
    output_path = directory / f"segmentation_{directory.name}-{seg_name}.nii.gz"
    try:
        image = sitk.GetImageFromArray(mask.astype(uint8))
        try:
            reader = sitk.ImageSeriesReader()
            names = reader.GetGDCMSeriesFileNames(str(directory))
            if names:
                reader.SetFileNames(names)
                reference = reader.Execute()
                if reference.GetSize()[::-1] == mask.shape:
                    image.CopyInformation(reference)
                else:
                    logger.warning(
                        "NIfTI geometry not copied for %s: mask shape %s does not "
                        "match scan size %s", seg_name, mask.shape, reference.GetSize()[::-1],
                    )
        except Exception as exc:
            logger.warning("Could not copy scan geometry onto NIfTI for %s: %s", seg_name, exc)

        sitk.WriteImage(image, str(output_path))
        logger.info("Saved NIfTI %s", output_path.name)
        return output_path
    except Exception as exc:
        logger.error("Could not write NIfTI for %s: %s", directory, exc)
        return None


def keep_largest_connected_component(mask: ndarray, connectivity: int = 1) -> ndarray:
    """Return only the largest connected blob in ``mask``.

    The model occasionally emits small spurious islands far from the tumour;
    the target is a single contiguous mass, so everything but the largest
    component can be discarded.
    """
    if not mask.any():
        return zeros_like(mask, dtype=bool_)

    structure = generate_binary_structure(rank=mask.ndim, connectivity=connectivity)
    labels = empty(mask.shape, dtype=int32)
    num_components = label(mask, structure=structure, output=labels)

    if num_components == 0:
        return zeros_like(mask, dtype=bool_)

    counts = bincount(labels.ravel())
    largest = argmax(counts[1:]) + 1  # index 0 is background

    out = empty(mask.shape, dtype=bool_)
    equal(labels, largest, out=out)
    return out
