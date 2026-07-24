"""The main window: browse scans, and queue folders for batch segmentation.

Two tabs:

* **View Scans** — point at a directory, list every DICOM series found beneath
  it, and open one in an :class:`~josh.ui.image_window.ImageWindow`.
* **Segmentation** — queue up whole folders and run the model over them
  unattended, writing a mask beside each scan.
"""

import logging
from pathlib import Path

from numpy import transpose
from PySide6.QtCore import Qt, QTime
from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from pydicom import dcmread

from .. import config
from ..dicom_io import load_series, locate_all_dicom_series
from ..masks import save_mask_nifti, save_mask_npy, save_rtstruct
from ..workers import SegmentationWorker
from .image_window import ImageWindow

logger = logging.getLogger(__name__)

# Qt.ItemDataRole slots used to stash the real folder path and series UID on a
# list item, so we never have to parse them back out of the display text.
FOLDER_PATH_ROLE = Qt.ItemDataRole.UserRole
SERIES_UID_ROLE = Qt.ItemDataRole.UserRole + 1

AI_MASK_SUFFIX = "Mesothelioma_AI"


def _format_dicom_date(value):
    """DICOM dates are ``YYYYMMDD``; show them as ``YY-MM-DD``."""
    if not value or len(value) < 6:
        return "Unknown"
    return f"{value[2:4]}-{value[4:6]}-{value[0:2]}"


def _format_dicom_time(value):
    """DICOM times are ``HHMMSS.frac``; show them as ``HH:MM:SS``."""
    if not value or len(value) < 6:
        return "Unknown"
    return f"{value[0:2]}:{value[2:4]}:{value[4:6]}"


class CTScanViewer(QMainWindow):
    """Application main window."""

    def __init__(self):
        super().__init__()
        self.directory = None
        self.all_series = []
        self.all_patient_information = []

        # Folders queued for batch segmentation, and the threads working them.
        self.selected_folders = []
        self.folder_queue = []
        self.workers = {}
        self.total_folders = 0
        self.folder_completed_count = 0

        # Viewer windows are kept alive here; dropping the reference would let
        # Qt garbage-collect an open window out from under the user.
        self.open_windows = []

        self.settings = config.load_settings()
        self._build_ui()

    def _build_ui(self):
        self.setWindowTitle("CT Scan Viewer")
        self.setGeometry(100, 100, 800, 600)

        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        self.tab_widget = QTabWidget(self)
        layout.addWidget(self.tab_widget)

        self.view_tab = QWidget()
        self.segmentation_tab = QWidget()
        self.tab_widget.addTab(self.view_tab, "View Scans")
        self.tab_widget.addTab(self.segmentation_tab, "Segmentation")

        self._build_view_tab()
        self._build_segmentation_tab()

    def _build_view_tab(self):
        layout = QVBoxLayout(self.view_tab)

        self.btn_open_folder = QPushButton("Open Folder", self)
        self.btn_open_folder.clicked.connect(self.open_folder)
        layout.addWidget(self.btn_open_folder)

        self.list_widget = QTableWidget(self)
        self.list_widget.setColumnCount(4)
        self.list_widget.setHorizontalHeaderLabels(
            ["Patient Name", "Modality", "Scan Date", "Scan Time"]
        )
        self.list_widget.horizontalHeader().setStretchLastSection(True)
        self.list_widget.setSortingEnabled(True)
        self.list_widget.setShowGrid(False)
        self.list_widget.clicked.connect(self.load_scan)
        layout.addWidget(self.list_widget)

    def _build_segmentation_tab(self):
        layout = QVBoxLayout(self.segmentation_tab)

        self.btn_add_folder = QPushButton("Add Folder for Segmentation", self)
        self.btn_add_folder.clicked.connect(self.add_folder_for_segmentation)
        layout.addWidget(self.btn_add_folder)

        self.segmentation_list_widget = QListWidget(self)
        self.segmentation_list_widget.setToolTip("Click a queued folder to remove it")
        self.segmentation_list_widget.clicked.connect(self.remove_folder_for_segmentation)
        layout.addWidget(self.segmentation_list_widget)

        self.progressBar = QProgressBar(self)
        self.progressBar.setVisible(False)
        layout.addWidget(self.progressBar)

        self.timeLabel = QLabel(self)
        self.timeLabel.setVisible(False)
        layout.addWidget(self.timeLabel)

        self.folderCountLabel = QLabel(self)
        self.folderCountLabel.setVisible(False)
        layout.addWidget(self.folderCountLabel)

        self.concurrent_segmentation_label = QLabel(
            "Concurrent Segmentations: (Experimental)", self
        )
        layout.addWidget(self.concurrent_segmentation_label)

        self.concurrent_segmentation_spinbox = QSpinBox(self)
        self.concurrent_segmentation_spinbox.setMinimum(1)
        self.concurrent_segmentation_spinbox.setValue(1)
        self.concurrent_segmentation_spinbox.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self.concurrent_segmentation_spinbox.setToolTip(
            "Segment several folders at once. Each one needs its own GPU memory, "
            "so raising this can exhaust the card."
        )

        self.full_model_checkbox = QCheckBox("Use Full Model", self)
        self.full_model_checkbox.setChecked(False)
        self.full_model_checkbox.setSizePolicy(
            QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed
        )
        self.full_model_checkbox.setToolTip(
            "Use all model folds for segmentation. Slower but usually more accurate; "
            "uncheck to run a single fold."
        )

        row = QHBoxLayout()
        row.addWidget(self.concurrent_segmentation_spinbox, stretch=1)
        row.addWidget(self.full_model_checkbox, stretch=0)
        layout.addLayout(row)

        self.btn_start_segmentation = QPushButton("Start Segmentation", self)
        self.btn_start_segmentation.clicked.connect(self.generate_segmentation)
        layout.addWidget(self.btn_start_segmentation)

    # ------------------------------------------------------------------
    # Browsing scans
    # ------------------------------------------------------------------
    def open_folder(self):
        """Scan a directory tree for DICOM series and list what was found."""
        directory = QFileDialog.getExistingDirectory(
            self, "Select Directory", self.settings.get("last_open_folder", "")
        )
        if not directory:
            return

        self.directory = directory
        self._remember_folder(directory)
        self.all_series, self.all_patient_information = locate_all_dicom_series(directory)

        # Sorting must be off while populating or Qt reorders rows mid-fill and
        # the row indices stop matching self.all_series.
        self.list_widget.setSortingEnabled(False)
        self.list_widget.setRowCount(len(self.all_patient_information))
        for row, info in enumerate(self.all_patient_information):
            values = [
                str(getattr(info, "PatientName", "Unknown")),
                getattr(info, "Modality", "Unknown"),
                _format_dicom_date(getattr(info, "ContentDate", None)),
                _format_dicom_time(getattr(info, "ContentTime", None)),
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                if column == 0:
                    # Remember which series this row came from, so sorting
                    # cannot decouple the row from its folder. The UID matters
                    # when one folder holds more than one series.
                    item.setData(FOLDER_PATH_ROLE, self.all_series[row])
                    item.setData(SERIES_UID_ROLE, getattr(info, "SeriesInstanceUID", None))
                self.list_widget.setItem(row, column, item)
        self.list_widget.setSortingEnabled(True)

        if not self.all_series:
            QMessageBox.information(
                self, "No Scans Found", f"No DICOM series were found under {directory}."
            )

    def load_scan(self, index):
        """Open the series on the clicked row in a viewer window."""
        anchor = self.list_widget.item(index.row(), 0)
        series_path = anchor.data(FOLDER_PATH_ROLE) if anchor else None
        if series_path is None:
            return
        series_uid = anchor.data(SERIES_UID_ROLE)

        try:
            # Load only the clicked series: a folder can hold several, and
            # mixing them produced the "jumping between images" behaviour.
            slices = load_series(series_path, series_uid)
        except Exception as exc:
            logger.error("Could not load %s: %s", series_path, exc)
            QMessageBox.critical(self, "Could Not Open Scan", f"{series_path}\n\n{exc}")
            return

        window = ImageWindow()
        window.set_directory(series_path)
        window.set_scan_data(slices)
        window.show_image()
        self.open_windows.append(window)

    # ------------------------------------------------------------------
    # Batch segmentation queue
    # ------------------------------------------------------------------
    def add_folder_for_segmentation(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select Directory", self.settings.get("last_open_folder", "")
        )
        if not folder or folder in self.selected_folders:
            return

        self._remember_folder(folder)
        self.selected_folders.append(folder)
        item = QListWidgetItem(self._describe_folder(folder))
        item.setData(FOLDER_PATH_ROLE, folder)
        self.segmentation_list_widget.addItem(item)

    def _remember_folder(self, folder):
        """Persist ``folder`` as the directory to reopen next time."""
        self.settings["last_open_folder"] = folder
        config.save_settings(self.settings)

    def _describe_folder(self, folder):
        """Label a queued folder with patient details, falling back to its name."""
        dcm_files = sorted(Path(folder).glob("*.dcm"))
        if not dcm_files:
            return Path(folder).name
        try:
            info = dcmread(str(dcm_files[0]), stop_before_pixels=True)
        except Exception as exc:
            logger.warning("Could not read metadata from %s: %s", dcm_files[0], exc)
            return Path(folder).name

        patient_name = str(getattr(info, "PatientName", "Unknown"))
        modality = getattr(info, "Modality", "Unknown")
        date = _format_dicom_date(getattr(info, "ContentDate", None))
        return f"{patient_name} - {modality} - {date}: {folder}"

    def remove_folder_for_segmentation(self, index):
        item = self.segmentation_list_widget.item(index.row())
        if item is None:
            return
        folder = item.data(FOLDER_PATH_ROLE)

        if folder in self.workers:
            reply = QMessageBox.question(
                self, "Kill Segmentation Process",
                "This folder is currently being segmented. Stop segmenting it?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
            # The model cannot be interrupted mid-volume, so this waits for the
            # current folder to finish rather than killing it.
            self.workers.pop(folder).stop()
            self._remove_queued_folder(folder, index.row())
            self.start_next_worker()
        else:
            self._remove_queued_folder(folder, index.row())

    def _remove_queued_folder(self, folder, row):
        if folder in self.selected_folders:
            self.selected_folders.remove(folder)
        if folder in self.folder_queue:
            self.folder_queue.remove(folder)
        self.segmentation_list_widget.takeItem(row)

    def _row_for_folder(self, folder):
        for row in range(self.segmentation_list_widget.count()):
            if self.segmentation_list_widget.item(row).data(FOLDER_PATH_ROLE) == folder:
                return row
        return None

    # ------------------------------------------------------------------
    # Running the batch
    # ------------------------------------------------------------------
    def generate_segmentation(self):
        if not self.selected_folders:
            QMessageBox.warning(self, "Segmentation", "No folders selected for segmentation!")
            return

        config.clear_temp_dir()

        self.folder_completed_count = 0
        self.folder_queue = self.selected_folders.copy()
        self.total_folders = len(self.selected_folders)

        # Progress is measured in folders, which is the only granularity the
        # model actually reports back at.
        self.progressBar.setRange(0, self.total_folders)
        self.progressBar.setValue(0)
        self.progressBar.setVisible(True)
        self.timeLabel.setVisible(True)
        self.timeLabel.setText("Estimated time remaining: calculating...")
        self.folderCountLabel.setVisible(True)
        self.folderCountLabel.setText(f"Folders completed: 0/{self.total_folders}")
        self.btn_start_segmentation.setEnabled(False)

        self.batch_timer = QTime.currentTime()
        for _ in range(min(self.concurrent_segmentation_spinbox.value(), len(self.folder_queue))):
            self.start_next_worker()

    def start_next_worker(self):
        if not self.folder_queue:
            return

        folder = self.folder_queue.pop(0)
        worker = SegmentationWorker(
            folder,
            folder_idx=self.selected_folders.index(folder),
            total_folders=self.total_folders,
            full_model=self.full_model_checkbox.isChecked(),
        )
        worker.folder_completed.connect(self.update_folder_count)
        worker.segmentation_complete.connect(
            lambda results, w=worker: self.on_segmentation_complete(results, w)
        )
        self.workers[folder] = worker
        worker.start()

    def update_folder_count(self, completed, total):
        self.folderCountLabel.setText(f"Folders completed: {completed}/{total}")
        self.progressBar.setValue(completed)
        self._update_time_estimate(completed, total)

    def _update_time_estimate(self, completed, total):
        """Extrapolate remaining time from how long the finished folders took."""
        if completed <= 0 or completed >= total:
            self.timeLabel.setText("Estimated time remaining: --:--:--")
            return
        elapsed_ms = self.batch_timer.msecsTo(QTime.currentTime())
        remaining_ms = int((elapsed_ms / completed) * (total - completed))
        remaining = QTime.fromMSecsSinceStartOfDay(remaining_ms).toString("hh:mm:ss")
        self.timeLabel.setText(f"Estimated time remaining: {remaining}")

    def on_segmentation_complete(self, results, worker):
        folder = str(worker.folder)
        if results is None:
            # The worker reports None on failure; skip saving rather than write
            # a fabricated empty mask beside the scan.
            logger.warning("Segmentation failed for %s; no mask written.", folder)
        else:
            self.save_segmentation(folder, results)
        self.folder_completed_count += 1
        self.workers.pop(folder, None)

        row = self._row_for_folder(folder)
        if row is not None:
            self.segmentation_list_widget.takeItem(row)

        if self.folder_completed_count >= self.total_folders:
            self._finish_batch()
        else:
            self.start_next_worker()

    def _finish_batch(self):
        self.folder_queue = []
        self.selected_folders = []
        self.folder_completed_count = 0
        self.workers = {}
        self.segmentation_list_widget.clear()

        # Empty the scratch directory but keep it: a previous version deleted
        # the directory itself, which broke the next batch in the same session.
        config.clear_temp_dir()

        self.progressBar.setValue(0)
        self.progressBar.setVisible(False)
        self.timeLabel.setVisible(False)
        self.folderCountLabel.setVisible(False)
        self.btn_start_segmentation.setEnabled(True)

    def save_segmentation(self, folder, results):
        """Write the model's mask beside the scan it came from."""
        # nibabel hands back (column, row, slice); the .npy convention is
        # (slice, row, column).
        mask = transpose(results, (2, 1, 0))
        save_mask_npy(folder, AI_MASK_SUFFIX, mask)
        save_mask_nifti(folder, AI_MASK_SUFFIX, mask)

        # rt_utils wants (row, column, slice).
        contour_mask = transpose(mask, (1, 2, 0)).astype(bool)
        save_rtstruct(
            folder, "Tumour", contour_mask, f"RTStruct_{Path(folder).name}_AI.dcm"
        )

    def closeEvent(self, event):
        """Wait for in-flight segmentations so no half-written mask is left."""
        for worker in list(self.workers.values()):
            worker.stop()
        event.accept()
