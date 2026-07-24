"""Background threads for the two jobs that would otherwise freeze the UI.

Both are ``QThread`` subclasses that do their work in :meth:`run` and hand the
result back over a signal. Results are emitted as ``Signal(object)`` so numpy
arrays travel through untouched — ``Signal(list)`` would silently convert every
volume into a Python list of row arrays.
"""

import logging
from pathlib import Path

import nibabel as nib
from PySide6.QtCore import QElapsedTimer, QThread, Signal

from . import config
from .dicom_io import convert_dicom
from .editing import interpolate_segmentation
from .masks import keep_largest_connected_component

from MedNeXt.nnunet_mednext.inference import predict_simple  # noqa: E402

logger = logging.getLogger(__name__)

# Files nnU-Net drops next to its output that are of no interest to the user.
PREDICTION_LEFTOVERS = ("plans.pkl", "postprocessing.json")


class SegmentationWorker(QThread):
    """Run the full DICOM -> NIfTI -> nnU-Net -> mask pipeline for one folder.

    The prediction cannot be interrupted once nnU-Net has started, so
    :meth:`stop` waits for the current folder to finish rather than killing it
    mid-write and leaving a corrupt volume behind.

    On success ``segmentation_complete`` carries a boolean mask; on failure it
    carries ``None`` so the caller can skip the folder instead of saving a
    fabricated empty mask that looks like a real result.
    """

    folder_completed = Signal(int, int)  # (folders done, total folders)
    segmentation_complete = Signal(object)  # boolean ndarray, or None on failure
    stage_changed = Signal(str)  # human-readable description of the current step

    def __init__(self, folder, folder_idx=0, total_folders=1, num_threads=1,
                 fast_mode=True, full_model=False):
        super().__init__()
        self.folder = Path(folder)
        self.folder_idx = folder_idx
        self.total_folders = total_folders
        self.num_threads = num_threads
        self.fast_mode = fast_mode
        self.full_model = full_model
        self.timer = QElapsedTimer()

    def run(self):
        self.timer.start()
        # Each job converts into its own private scratch subdirectory so that
        # concurrent jobs cannot overwrite each other's intermediate files.
        nifti_dir = config.job_temp_dir(self.folder.name)
        prediction_path = None
        mask = None

        try:
            self.stage_changed.emit("Converting scan to NIfTI...")
            convert_dicom(str(self.folder), str(nifti_dir), concurrent=self.num_threads)

            # nnU-Net writes its prediction into the scan folder itself.
            self.stage_changed.emit("Running segmentation model...")
            predict_simple.main(
                str(nifti_dir), str(self.folder),
                self.num_threads, self.fast_mode, self.full_model,
            )

            predictions = sorted(self.folder.glob("*.nii.gz"))
            if not predictions:
                raise FileNotFoundError(f"nnU-Net produced no output in {self.folder}")
            prediction_path = predictions[0]

            self.stage_changed.emit("Post-processing mask...")
            mask = nib.load(str(prediction_path)).get_fdata()
            mask = (mask > 0.5).astype(bool, copy=False)
            try:
                mask = keep_largest_connected_component(mask)
            except Exception as exc:
                logger.warning("Could not prune to largest component, using raw mask: %s", exc)
        except Exception as exc:
            # A single bad folder must not take down a whole batch. Report the
            # failure (mask stays None) and let the caller move on.
            logger.error("Error processing %s, skipping: %s", self.folder, exc)
            mask = None

        self.folder_completed.emit(self.folder_idx + 1, self.total_folders)
        self.segmentation_complete.emit(mask)
        self._cleanup(nifti_dir, prediction_path)

    def _cleanup(self, nifti_dir: Path, prediction_path):
        """Remove the intermediate files this run created.

        ``prediction_path`` may be ``None`` if prediction failed before writing
        anything, which is why it is checked rather than assumed.
        """
        config.remove_job_temp_dir(self.folder.name)

        leftovers = [self.folder / name for name in PREDICTION_LEFTOVERS]
        if prediction_path is not None:
            leftovers.append(Path(prediction_path))
        for path in leftovers:
            try:
                path.unlink()
            except FileNotFoundError:
                pass
            except OSError as exc:
                logger.warning("Could not remove %s: %s", path, exc)

    def stop(self):
        """Ask the thread to finish and block until it has."""
        self.quit()
        self.wait()


class InterpolationWorker(QThread):
    """Interpolate a segmentation across unannotated slices off the UI thread."""

    interpolation_complete = Signal(object)  # list of 2D boolean arrays

    def __init__(self, segmentation_list, ct_scan_list):
        super().__init__()
        self.segmentation_list = segmentation_list
        self.ct_scan_list = ct_scan_list

    def run(self):
        try:
            result = interpolate_segmentation(self.segmentation_list, self.ct_scan_list)
        except Exception as exc:
            # Emit the input unchanged so the UI re-enables its button rather
            # than leaving the user stuck on "Interpolating...".
            logger.error("Interpolation failed: %s", exc)
            result = self.segmentation_list
        self.interpolation_complete.emit(result)

    def stop(self):
        self.quit()
        self.wait()
