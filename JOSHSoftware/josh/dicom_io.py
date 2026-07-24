"""Finding DICOM series and converting them into the NIfTI volumes the model expects.
"""

import gc
import logging
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pydicom
import SimpleITK as sitk
from numpy import clip, stack

logger = logging.getLogger(__name__)

# HU window the model was trained on. Do not change without retraining.
HU_CLIP_MIN = -200
HU_CLIP_MAX = 800

# Files that live alongside a series but are not scan slices.
_NON_SLICE_NAMES = {"dicomdir", "dicomdir.dcm"}


def _is_candidate_slice(filename: str) -> bool:
    """True if ``filename`` might be a scan slice (not a directory record/RTSTRUCT)."""
    lower = filename.lower()
    if lower in _NON_SLICE_NAMES:
        return False
    if "rtstruct" in lower:
        return False
    return lower.endswith(".dcm")


def locate_all_dicom_series(directory):
    """Find every DICOM series beneath ``directory``.

    Returns ``(series_paths, representative_datasets)`` — two parallel lists,
    one folder path per series and one representative pydicom dataset each,
    matching the interface the UI expects.

    This replaces an earlier compiled version that only inspected the *first*
    file in each folder (so a folder whose first file was not a ``.dcm`` was
    skipped entirely) and keyed purely on the folder path. Here every file is
    considered, unreadable files are skipped rather than fatal, and slices are
    grouped by ``SeriesInstanceUID`` so a folder holding two series is reported
    as two entries instead of being silently merged.
    """
    directory = Path(directory)
    series_paths = []
    representatives = []

    for root, _dirs, files in _walk(directory):
        # Group this folder's readable slices by their SeriesInstanceUID.
        by_series = defaultdict(list)
        for filename in files:
            if not _is_candidate_slice(filename):
                continue
            path = Path(root) / filename
            dataset = _safe_read_header(path)
            if dataset is None:
                continue
            uid = getattr(dataset, "SeriesInstanceUID", None) or "unknown"
            by_series[uid].append(dataset)

        if not by_series:
            continue

        # One entry per distinct series found in this folder. The folder path
        # is still the identifier the rest of the app uses (masks are saved
        # beside the scan), so a folder with multiple series is logged.
        if len(by_series) > 1:
            logger.warning(
                "Folder %s contains %d distinct DICOM series; they share one "
                "folder so masks saved for them would collide.",
                root, len(by_series),
            )
        for uid, datasets in by_series.items():
            series_paths.append(str(root))
            representatives.append(datasets[0])
            logger.info("Found series %s (%d slices) in %s", uid, len(datasets), root)

    return series_paths, representatives


def _walk(directory):
    """Yield ``(root, dirs, files)`` for ``directory`` and every subfolder."""
    import os
    yield from os.walk(directory)


def _safe_read_header(path: Path):
    """Read a DICOM file's header only, returning ``None`` if it is not DICOM."""
    try:
        return pydicom.dcmread(str(path), stop_before_pixels=True, force=False)
    except Exception:
        # Not a DICOM file, or unreadable — expected for stray files in a tree.
        logger.debug("Skipping non-DICOM or unreadable file: %s", path)
        return None


def _slice_position(dataset):
    """Sort key for slices: z of ImagePositionPatient, falling back to SliceLocation."""
    try:
        return float(dataset.ImagePositionPatient[2])
    except (AttributeError, IndexError, TypeError, ValueError):
        return float(getattr(dataset, "SliceLocation", 0.0))


def load_series(folder, series_uid=None):
    """Load the pixel slices of a single series from ``folder``.

    When a folder holds slices from more than one series (or more than one
    patient), reading every ``.dcm`` and sorting by position interleaves them.
    Passing ``series_uid`` restricts loading to one series so the volume is 
    coherent; ``None`` loads everything in the folder.

    Returns a list of pydicom datasets sorted head-to-foot, de-duplicated by
    slice position, each with ``SliceThickness`` set.
    """
    folder = Path(folder)

    # First pass: read headers only and keep the files in the wanted series.
    candidates = []
    for path in folder.iterdir():
        if not _is_candidate_slice(path.name):
            continue
        header = _safe_read_header(path)
        if header is None:
            continue
        if series_uid is not None and getattr(header, "SeriesInstanceUID", None) != series_uid:
            continue
        candidates.append(path)

    if not candidates:
        raise ValueError(
            f"No DICOM slices found in {folder}"
            + (f" for series {series_uid}" if series_uid else "")
        )

    # Second pass: read pixel data only for the slices we are keeping.
    slices = [pydicom.dcmread(str(path), force=True) for path in candidates]
    slices.sort(key=_slice_position, reverse=True)

    # Scanners sometimes emit the same position twice; drop the duplicates.
    unique = []
    seen_positions = set()
    for s in slices:
        position = _slice_position(s)
        if position not in seen_positions:
            seen_positions.add(position)
            unique.append(s)
    slices = unique

    if len(slices) >= 2:
        thickness = abs(_slice_position(slices[0]) - _slice_position(slices[1]))
        for s in slices:
            s.SliceThickness = thickness

    return slices


def process_scan(scan_folder: Path, output_folder: Path) -> Path | None:
    """Convert one folder of DICOM slices to a single ``.nii.gz`` volume.

    Returns the path written, or ``None`` if the folder could not be read.
    The output is named ``<folder name>_0000.nii.gz`` because nnU-Net
    identifies input channels by the ``_0000`` suffix.
    """
    scan_folder = Path(scan_folder)
    output_folder = Path(output_folder)
    patient_id = scan_folder.name

    reader = sitk.ImageSeriesReader()
    try:
        dicom_names = reader.GetGDCMSeriesFileNames(str(scan_folder))
    except RuntimeError as exc:
        logger.error("Could not read DICOM files in %s: %s", scan_folder, exc)
        return None

    if not dicom_names:
        logger.error("No DICOM series found in %s", scan_folder)
        return None

    slices = None
    try:
        # SimpleITK gives us the geometry (spacing, origin, direction); pydicom
        # gives us pixel data we can rescale ourselves and de-duplicate.
        reader.SetFileNames(dicom_names)
        reference_image = reader.Execute()

        slices = [pydicom.dcmread(name) for name in dicom_names]
        slices.sort(key=lambda s: float(s.ImagePositionPatient[2]))

        # Scanners occasionally emit the same slice position twice; keeping
        # duplicates would misalign the volume against its geometry.
        slices = [
            s
            for i, s in enumerate(slices)
            if i == 0
            or s.ImagePositionPatient[2] != slices[i - 1].ImagePositionPatient[2]
        ]

        # Apply each slice's own rescale slope/intercept before stacking
        rescaled = []
        for s in slices:
            slope = float(getattr(s, "RescaleSlope", 1))
            intercept = float(getattr(s, "RescaleIntercept", 0))
            rescaled.append(s.pixel_array * slope + intercept)

        volume = clip(stack(rescaled, axis=0), HU_CLIP_MIN, HU_CLIP_MAX)

        image = sitk.GetImageFromArray(volume)
        image.CopyInformation(reference_image)

        output_file = output_folder / f"{patient_id}_0000.nii.gz"
        sitk.WriteImage(image, str(output_file))

        if not output_file.exists():
            logger.error("SimpleITK reported success but wrote nothing for %s", scan_folder)
            return None

        logger.info("Converted %s -> %s", scan_folder, output_file.name)
        return output_file
    except Exception as exc:
        logger.error("Error processing %s: %s", scan_folder, exc)
        return None
    finally:
        # Scans are hundreds of MB; drop them before the next folder starts.
        del slices
        gc.collect()


def convert_dicom(scan_folders, output_folder, concurrent: int = 1) -> list[Path]:
    """Convert one or many DICOM folders to NIfTI, optionally in parallel.

    ``scan_folders`` may be a single path or an iterable of paths. Returns the
    files that were written successfully; folders that failed are logged and
    omitted rather than raising.
    """
    if isinstance(scan_folders, (str, Path)):
        scan_folders = [scan_folders]
    scan_folders = list(scan_folders)

    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    logger.info("Converting %d scan folder(s) to NIfTI", len(scan_folders))

    written = []
    with ThreadPoolExecutor(max_workers=max(1, concurrent)) as executor:
        futures = {
            executor.submit(process_scan, Path(folder), output_folder): folder
            for folder in scan_folders
        }
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                written.append(result)

    return written
