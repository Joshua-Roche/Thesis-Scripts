"""The scan viewer and segmentation editor.
"""

import logging
from pathlib import Path

from cv2 import (
    MORPH_ELLIPSE,
    erode,
    fillPoly,
    getStructuringElement,
)
from numpy import array, flip, int32, load, logical_or, stack, transpose, uint8, zeros, zeros_like
from PySide6.QtCore import QElapsedTimer, QPoint, Qt, QEvent
from PySide6.QtGui import QAction, QColor, QImage, QKeySequence, QPainter, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMenuBar,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QVBoxLayout,
)
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

from .. import config
from ..brush import stamp_brush
from ..editing import delete_left_half, delete_right_half, roi_statistics
from ..history import UndoHistory
from ..imaging import window_to_uint8
from ..masks import save_mask_nifti, save_mask_npy, save_rtstruct, seg_name_from_npy
from ..workers import InterpolationWorker, SegmentationWorker
from . import export, vtk_view

logger = logging.getLogger(__name__)

# Radiological window presets, as (label, min HU, max HU).
WINDOW_PRESETS = [
    ("Soft Tissue (-135 to 215)", -135, 215),
    ("Lung (-1600 to 200)", -1600, 200),
    ("Abdomen (-1500 to 1500)", -1500, 1500),
    ("Bone (-100 to 1000)", -100, 1000),
]

PLANES = ("Transversal", "Sagittal", "Coronal")

# Axis orders for re-slicing a volume stored transversally as (z, row, col).
_PLANE_TRANSPOSE = {
    ("Transversal", "Sagittal"): (2, 0, 1),
    ("Transversal", "Coronal"): (1, 0, 2),
    ("Sagittal", "Transversal"): (1, 2, 0),
    ("Sagittal", "Coronal"): (2, 1, 0),
    ("Coronal", "Transversal"): (1, 0, 2),
    ("Coronal", "Sagittal"): (2, 1, 0),
}

# How many slices either side the 3D brush reaches, and how fast it tapers.
BRUSH_3D_REACH = 2

SEGMENTATION_COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255),
    (255, 255, 0), (255, 0, 255), (0, 255, 255),
]

# Name given to the segmentation the model produces.
AI_SEGMENTATION_NAME = "Mesothelioma"

# Zoom limits and step for the Shift+wheel zoom.
ZOOM_MIN = 1.0
ZOOM_MAX = 8.0
ZOOM_STEP = 1.25

# HU per pixel of middle-drag when adjusting window/level.
WINDOW_DRAG_SENSITIVITY = 3


class ImageWindow(QDialog):
    """Viewer and editor for a single CT series."""

    def __init__(self):
        super().__init__()
        self.setWindowFlag(Qt.WindowType.WindowMaximizeButtonHint, True)
        self.setWindowFlag(Qt.WindowType.WindowMinimizeButtonHint, True)
        self.setWindowTitle("CT Scan Viewer")

        # --- scan state ---
        self.scan_metadata = None
        self.scan_data_hu = None       # list of 2D HU slices, transversal
        self.reoriented_data = None    # same volume re-sliced into current_plane
        self.current_plane = "Transversal"
        self.current_directory = None
        self.volume_conversion_units = 0.0  # mm^3 per voxel

        # --- segmentation state ---
        self.source_segmentation = {}  # name -> list of 2D boolean slices
        self.segmentation_colors = {}
        self.seg_name = None
        self.is_modified = False

        # --- drawing state ---
        self.drawing_contour = False
        self.contour_points = []
        self.mouse_pos = None

        # --- undo/redo ---
        self.history = UndoHistory()

        # --- zoom / window-level drag ---
        self.zoom = 1.0
        self._window_drag_origin = None   # (QPoint, min_hu, max_hu) while dragging

        self._build_ui()

    # ------------------------------------------------------------------
    # Current segmentation
    # ------------------------------------------------------------------
    @property
    def segmentation_data(self):
        """Slices of the currently selected segmentation, or ``None``."""
        if self.seg_name is None:
            return None
        return self.source_segmentation.get(self.seg_name)

    @property
    def display_data(self):
        """Scan slices in the plane currently being viewed."""
        return self.reoriented_data if self.reoriented_data else self.scan_data_hu

    def _set_current_segmentation(self, slices):
        self.source_segmentation[self.seg_name] = slices

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self):
        main_layout = QHBoxLayout(self)
        image_layout = QVBoxLayout()
        control_layout = QVBoxLayout()

        self._build_menu(main_layout)
        self._build_image_area(image_layout)
        self._build_controls()
        self._build_control_panel(control_layout)

        main_layout.addLayout(image_layout, 2)
        main_layout.addLayout(control_layout, 1)

        # 3D view lives beside the 2D one and is revealed on demand.
        self.vtk_widget = QVTKRenderWindowInteractor(self)
        vtk_layout = QVBoxLayout()
        vtk_layout.addWidget(self.vtk_widget)
        self.vtk_frame = QFrame()
        self.vtk_frame.setLayout(vtk_layout)
        self.vtk_frame.hide()
        main_layout.addWidget(self.vtk_frame, 2)

    def _build_menu(self, main_layout):
        self.menu_bar = QMenuBar()
        file_menu = self.menu_bar.addMenu("File")
        view_menu = self.menu_bar.addMenu("View")

        save_action = QAction("Save", self)
        save_action.triggered.connect(self.save_segmentation)
        file_menu.addAction(save_action)

        gif_menu = file_menu.addMenu("Export as GIF")
        for label, handler in (
            ("Export Current View With Scan", self.export_gif_with_scan),
            ("Export Current View Without Scan", self.export_gif_without_scan),
            ("Export 3D Segmentation", self.export_3d_segmentation),
        ):
            action = QAction(label, self)
            action.triggered.connect(handler)
            gif_menu.addAction(action)

        presets_menu = view_menu.addMenu("Window Presets")
        for label, hu_min, hu_max in WINDOW_PRESETS:
            action = QAction(label, self)
            # Bind the values now; a bare closure would capture the loop variable.
            action.triggered.connect(
                lambda _checked=False, lo=hu_min, hi=hu_max: self.set_window_preset(lo, hi)
            )
            presets_menu.addAction(action)

        plane_menu = view_menu.addMenu("Plane")
        for plane in PLANES:
            action = QAction(plane, self)
            action.triggered.connect(lambda _checked=False, p=plane: self.set_plane(p))
            plane_menu.addAction(action)

        top_layout = QHBoxLayout()
        top_layout.addWidget(self.menu_bar)
        main_layout.addLayout(top_layout)

    def _build_image_area(self, image_layout):
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.image_label.setPixmap(QPixmap(800, 600))
        self.image_label.setMouseTracking(True)
        self.image_label.installEventFilter(self)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidget(self.image_label)
        self.scroll_area.setWidgetResizable(False)
        self.scroll_area.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.scroll_area.setMouseTracking(True)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.valueChanged.connect(self.update_image_display)

        self.change_slice_button = QSpinBox()
        self.change_slice_button.setRange(1, 100)
        self.change_slice_button.setValue(1)
        self.change_slice_button.setPrefix("Slice: ")
        self.change_slice_button.setFixedSize(200, 20)
        self.change_slice_button.valueChanged.connect(self.slider.setValue)
        self.slider.valueChanged.connect(self.change_slice_button.setValue)

        self.pixel_value_label = QLabel("HU: N/A")
        self.pixel_value_label.setFixedSize(100, 20)

        slice_control_layout = QHBoxLayout()
        slice_control_layout.addWidget(self.change_slice_button)
        slice_control_layout.addWidget(self.pixel_value_label)
        slice_control_layout.addStretch()

        image_layout.addWidget(self.slider)
        image_layout.addLayout(slice_control_layout)
        image_layout.addWidget(self.scroll_area)

    def _build_controls(self):
        # Start on the window/level the user last left, if remembered.
        settings = config.load_settings()

        self.norm_min_spinbox = QSpinBox()
        self.norm_min_spinbox.setRange(-2000, 300)
        self.norm_min_spinbox.setValue(settings["window_min_hu"])
        self.norm_min_spinbox.valueChanged.connect(self.refresh)

        self.norm_max_spinbox = QSpinBox()
        self.norm_max_spinbox.setRange(-300, 3000)
        self.norm_max_spinbox.setValue(settings["window_max_hu"])
        self.norm_max_spinbox.valueChanged.connect(self.refresh)

        self.generate_segmentation_button = QPushButton("Load/Generate Segmentation")
        self.generate_segmentation_button.clicked.connect(self.load_or_generate_segmentation)

        self.start_segmentating_button = QPushButton("Start Segmenting from Scratch")
        self.start_segmentating_button.clicked.connect(self.start_segmentation_from_scratch)

        self.save_segmentation_button = QPushButton("Save Segmentation")
        self.save_segmentation_button.clicked.connect(self.save_segmentation)

        self.add_segmentation_button = QPushButton("Add Segmentation")
        self.add_segmentation_button.clicked.connect(self.add_segmentation)

        self.segmentation_combo_box = QComboBox()
        self.segmentation_combo_box.currentIndexChanged.connect(self.select_segmentation)

        self.view_all_segmentations_checkbox = QCheckBox("View All Segmentations")
        self.view_all_segmentations_checkbox.toggled.connect(self.refresh)

        self.brush_size_spinbox = QSpinBox()
        self.brush_size_spinbox.setRange(1, 50)
        self.brush_size_spinbox.setValue(5)
        self.brush_size_spinbox_label = QLabel("Brush Size:")

        self.toggle_brush_button = QPushButton("Brush Mode")
        self.toggle_brush_button.clicked.connect(self.toggle_contour_mode)
        self.toggle_contour_button = QPushButton("Contour Mode")
        self.toggle_contour_button.clicked.connect(self.toggle_contour_mode)
        self.toggle_contour_button.setVisible(False)

        self.three_d_brush_tick_box = QCheckBox("3D Drawing")

        self.delete_left_half_button = QPushButton("Delete Left Half")
        self.delete_left_half_button.clicked.connect(self.delete_left)
        self.delete_right_half_button = QPushButton("Delete Right Half")
        self.delete_right_half_button.clicked.connect(self.delete_right)
        self.delete_slice_button = QPushButton("Delete Slice")
        self.delete_slice_button.clicked.connect(self.delete_slice)
        self.delete_all_button = QPushButton("Delete All")
        self.delete_all_button.clicked.connect(self.delete_all)
        self.interpolate_button = QPushButton("Interpolate Segmentation")
        self.interpolate_button.clicked.connect(self.interpolate)

        self.alpha_slider = QSlider(Qt.Orientation.Horizontal)
        self.alpha_slider.setRange(0, 100)
        self.alpha_slider.setValue(50)
        self.alpha_slider.setVisible(False)
        self.alpha_slider.valueChanged.connect(self.refresh)
        self.alpha_slider_label = QLabel("Opacity:")
        self.alpha_slider_label.setVisible(False)

        self.roi_stats_button = QPushButton("ROI Statistics")
        self.roi_stats_button.clicked.connect(self.show_roi_statistics)
        self.rename_segmentation_button = QPushButton("Rename")
        self.rename_segmentation_button.clicked.connect(self.rename_segmentation)
        self.delete_segmentation_button = QPushButton("Delete Segmentation")
        self.delete_segmentation_button.clicked.connect(self.delete_segmentation)
        self.view_3d_button = QPushButton("View 3D Segmentation")
        self.view_3d_button.clicked.connect(self.view_3d_segmentation)
        self.view_3d_button.setVisible(False)
        self.close_3d_button = QPushButton("Close 3D Viewer")
        self.close_3d_button.clicked.connect(self.close_3d_segmentation)
        self.close_3d_button.setVisible(False)

        self.progressBar = QProgressBar()
        # There is no progress signal to report against during prediction, so
        # show a busy indicator rather than a bar that sits at 0%.
        self.progressBar.setRange(0, 0)
        self.progressBar.setVisible(False)
        self.timeLabel = QLabel("Segmenting — this can take several minutes...")
        self.timeLabel.setVisible(False)

        self.save_shortcut = QShortcut(QKeySequence(QKeySequence.StandardKey.Save), self)
        self.save_shortcut.activated.connect(self.save_segmentation)
        self.undo_shortcut = QShortcut(QKeySequence(QKeySequence.StandardKey.Undo), self)
        self.undo_shortcut.activated.connect(self.undo)
        self.redo_shortcut = QShortcut(QKeySequence(QKeySequence.StandardKey.Redo), self)
        self.redo_shortcut.activated.connect(self.redo)

        self.show_segmentation_controls(False)

    def _build_control_panel(self, control_layout):
        norm_group = QGroupBox("Normalisation")
        norm_layout = QFormLayout()
        norm_layout.addRow("Min HU:", self.norm_min_spinbox)
        norm_layout.addRow("Max HU:", self.norm_max_spinbox)
        norm_group.setLayout(norm_layout)

        seg_group = QGroupBox("Segmentation")
        seg_layout = QVBoxLayout()
        seg_layout.addWidget(self.generate_segmentation_button)
        seg_layout.addWidget(self.start_segmentating_button)
        seg_layout.addWidget(self.timeLabel)
        seg_layout.addWidget(self.progressBar)
        seg_layout.addWidget(self.save_segmentation_button)

        additional_segs_layout = QHBoxLayout()
        additional_segs_layout.addWidget(self.add_segmentation_button)
        additional_segs_layout.addWidget(self.segmentation_combo_box)
        additional_segs_layout.addWidget(self.view_all_segmentations_checkbox)
        seg_layout.addLayout(additional_segs_layout)

        manage_segs_layout = QHBoxLayout()
        manage_segs_layout.addWidget(self.rename_segmentation_button)
        manage_segs_layout.addWidget(self.delete_segmentation_button)
        seg_layout.addLayout(manage_segs_layout)

        seg_layout.addWidget(self.view_3d_button)
        seg_layout.addWidget(self.close_3d_button)
        seg_layout.addWidget(self.roi_stats_button)
        seg_group.setLayout(seg_layout)

        self.brush_group = QGroupBox("Brush Settings")
        self.brush_group.setVisible(False)
        brush_layout = QFormLayout()
        brush_layout.addRow(self.three_d_brush_tick_box)
        brush_layout.addRow(self.brush_size_spinbox_label, self.brush_size_spinbox)
        brush_layout.addRow(self.alpha_slider_label, self.alpha_slider)
        brush_layout.addRow(self.toggle_brush_button, self.toggle_contour_button)
        self.brush_group.setLayout(brush_layout)

        self.macro_seg_group = QGroupBox("Macro Segmentation")
        self.macro_seg_group.setVisible(False)
        macro_layout = QFormLayout()
        halves_layout = QHBoxLayout()
        halves_layout.addWidget(self.delete_left_half_button)
        halves_layout.addWidget(self.delete_right_half_button)
        macro_layout.addRow(halves_layout)
        macro_layout.addRow(self.interpolate_button)
        deletes_layout = QHBoxLayout()
        deletes_layout.addWidget(self.delete_slice_button)
        deletes_layout.addWidget(self.delete_all_button)
        macro_layout.addRow(deletes_layout)
        self.macro_seg_group.setLayout(macro_layout)

        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.close)

        control_layout.addWidget(norm_group)
        control_layout.addWidget(seg_group)
        control_layout.addWidget(self.brush_group)
        control_layout.addWidget(self.macro_seg_group)
        control_layout.addWidget(self.close_button)
        control_layout.addStretch()

    def show_segmentation_controls(self, is_visible):
        """Reveal the editing controls once there is something to edit."""
        for widget in (
            self.save_segmentation_button,
            self.add_segmentation_button,
            self.segmentation_combo_box,
            self.view_all_segmentations_checkbox,
            self.brush_size_spinbox_label,
            self.brush_size_spinbox,
            self.alpha_slider,
            self.alpha_slider_label,
            self.toggle_brush_button,
            self.roi_stats_button,
            self.rename_segmentation_button,
            self.delete_segmentation_button,
            self.view_3d_button,
        ):
            widget.setVisible(is_visible)
        self.start_segmentating_button.setVisible(not is_visible)

    def _enter_editing_mode(self):
        """Switch the panel over to editing an existing segmentation."""
        self.show_segmentation_controls(True)
        self.generate_segmentation_button.setVisible(False)
        self.start_segmentating_button.setVisible(False)
        self.brush_group.setVisible(True)
        self.macro_seg_group.setVisible(True)
        self.refresh()

    # ------------------------------------------------------------------
    # Loading a scan
    # ------------------------------------------------------------------
    def set_scan_data(self, scan_metadata):
        """Load a sorted list of pydicom slices into the viewer."""
        from dicom_utils import convert_to_hu  # compiled extension

        self.scan_metadata = scan_metadata
        pixel_spacing = scan_metadata[0].PixelSpacing
        slice_thickness = scan_metadata[0].SliceThickness
        self.volume_conversion_units = pixel_spacing[0] * pixel_spacing[1] * slice_thickness

        volume = stack([convert_to_hu(s, apply_modality_lut=True, normalise=False)
                        for s in scan_metadata])
        # DICOM rows run top-to-bottom in patient coordinates; flip so the
        # image is displayed the way a radiologist expects.
        volume = flip(volume, axis=1)
        self.scan_data_hu = [volume[i] for i in range(volume.shape[0])]

        self._set_slice_range(len(self.scan_data_hu))
        self.update_image_display(0)

    def set_directory(self, directory):
        self.current_directory = directory

    def show_image(self):
        self.showMaximized()

    def _set_slice_range(self, num_slices):
        for widget in (self.slider, self.change_slice_button):
            widget.blockSignals(True)
            widget.setRange(0, max(0, num_slices - 1))
            widget.setValue(0)
            widget.blockSignals(False)

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------
    def set_window_preset(self, hu_min, hu_max):
        self.norm_min_spinbox.setValue(hu_min)
        self.norm_max_spinbox.setValue(hu_max)
        self.refresh()

    def refresh(self):
        """Redraw the current slice."""
        self.update_image_display(self.slider.value())

    def _display_scale(self):
        """Pixels of source image per pixel of displayed image, as (x, y)."""
        pixmap = self.image_label.pixmap()
        scan_data = self.display_data
        if not scan_data or pixmap is None or pixmap.width() == 0 or pixmap.height() == 0:
            return 1.0, 1.0
        slice_shape = scan_data[0].shape
        return slice_shape[1] / pixmap.width(), slice_shape[0] / pixmap.height()

    def _fit_size(self, source_width, source_height):
        """Size (QSize) the slice should occupy at the current zoom.

        At zoom 1.0 the slice fits the scroll-area viewport; higher zoom scales
        up from there, and the surrounding scroll area supplies panning.
        """
        from PySide6.QtCore import QSize

        viewport = self.scroll_area.viewport().size()
        vw = viewport.width() if viewport.width() > 10 else 800
        vh = viewport.height() if viewport.height() > 10 else 600
        base = QSize(source_width, source_height).scaled(
            vw, vh, Qt.AspectRatioMode.KeepAspectRatio
        )
        return QSize(int(base.width() * self.zoom), int(base.height() * self.zoom))

    def update_image_display(self, index):
        scan_data = self.display_data
        if not scan_data or index >= len(scan_data):
            return

        greyscale = window_to_uint8(
            scan_data[index], self.norm_min_spinbox.value(), self.norm_max_spinbox.value()
        )
        qimg = QImage(greyscale.data, greyscale.shape[1], greyscale.shape[0],
                      greyscale.strides[0], QImage.Format.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qimg)

        painter = QPainter(pixmap)
        try:
            self._paint_segmentations(painter, index)
            self._paint_cursor(painter)
        finally:
            painter.end()

        target = self._fit_size(greyscale.shape[1], greyscale.shape[0])
        scaled = pixmap.scaled(
            target, Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.image_label.setPixmap(scaled)
        # Resize the label to match, so the scroll area reports the right
        # content size and shows scrollbars only when zoomed past the viewport.
        self.image_label.resize(scaled.size())

    def _paint_segmentations(self, painter, index):
        """Overlay the visible segmentation(s) on the current slice."""
        if self.view_all_segmentations_checkbox.isChecked():
            visible = list(self.source_segmentation.items())
        elif self.segmentation_data is not None:
            visible = [(self.seg_name, self.segmentation_data)]
        else:
            return

        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
        painter.setOpacity(self.alpha_slider.value() / 100)

        for name, slices in visible:
            if index >= len(slices):
                continue
            colour = self.segmentation_colors.get(name, (0, 255, 0))
            mask = (slices[index] > 0).astype(uint8) * 255
            rgba = zeros((mask.shape[0], mask.shape[1], 4), dtype=uint8)
            rgba[:, :, 0] = colour[0]
            rgba[:, :, 1] = colour[1]
            rgba[:, :, 2] = colour[2]
            rgba[:, :, 3] = mask
            seg_qimg = QImage(rgba.data.tobytes(), rgba.shape[1], rgba.shape[0],
                              QImage.Format.Format_RGBA8888)
            painter.drawImage(0, 0, seg_qimg)

    def _paint_cursor(self, painter):
        """Draw either the contour points being placed or the brush outline."""
        painter.setOpacity(1.0)
        if self.drawing_contour:
            painter.setPen(QColor(255, 0, 0))
            for point in self.contour_points:
                painter.drawEllipse(point[0], point[1], 3, 3)
        elif self.mouse_pos is not None and self.segmentation_data is not None:
            colour = self.segmentation_colors.get(self.seg_name, (0, 0, 255))
            painter.setPen(QColor(*colour))
            radius = self.brush_size_spinbox.value()
            scale_x, scale_y = self._display_scale()
            painter.drawEllipse(
                int(self.mouse_pos.x() * scale_x) - radius,
                int(self.mouse_pos.y() * scale_y) - radius,
                radius * 2, radius * 2,
            )

    # ------------------------------------------------------------------
    # Plane switching
    # ------------------------------------------------------------------
    def set_plane(self, plane):
        """Re-slice the scan and every segmentation into ``plane``."""
        if plane == self.current_plane or not self.scan_data_hu:
            return

        axes = _PLANE_TRANSPOSE.get((self.current_plane, plane))
        if axes is None:
            logger.warning("No transpose defined for %s -> %s", self.current_plane, plane)
            return

        for name, slices in self.source_segmentation.items():
            reoriented = transpose(stack(slices), axes)
            self.source_segmentation[name] = [reoriented[i] for i in range(reoriented.shape[0])]

        # The scan itself is always stored transversally, so re-slice it from
        # scratch rather than chaining transposes and accumulating confusion.
        transversal = stack(self.scan_data_hu)
        if plane == "Transversal":
            self.reoriented_data = None
            num_slices = transversal.shape[0]
        else:
            axes_from_transversal = _PLANE_TRANSPOSE[("Transversal", plane)]
            reoriented = transpose(transversal, axes_from_transversal)
            self.reoriented_data = [reoriented[i] for i in range(reoriented.shape[0])]
            num_slices = reoriented.shape[0]

        self.current_plane = plane
        # Undo snapshots are per-orientation; a re-slice invalidates them.
        self.history.clear()
        self._set_slice_range(num_slices)
        self.update_image_display(0)

    # ------------------------------------------------------------------
    # Segmentation management
    # ------------------------------------------------------------------
    def _blank_segmentation(self):
        return [zeros_like(slice_data, dtype=bool) for slice_data in self.display_data]

    def _assign_colour(self, seg_name):
        index = len(self.segmentation_colors) % len(SEGMENTATION_COLORS)
        self.segmentation_colors[seg_name] = SEGMENTATION_COLORS[index]

    def _prompt_for_name(self):
        seg_name, ok = QInputDialog.getText(
            self, "Segmentation Name", "Enter a name for the new segmentation:"
        )
        if not ok:
            return None
        seg_name = seg_name.strip()
        if not seg_name:
            QMessageBox.warning(self, "Segmentation", "Segmentation name cannot be empty!")
            return None
        if seg_name in self.source_segmentation:
            QMessageBox.warning(self, "Segmentation", f"'{seg_name}' already exists!")
            return None
        return seg_name

    def add_segmentation(self):
        seg_name = self._prompt_for_name()
        if seg_name is None:
            return
        self.seg_name = seg_name
        self._assign_colour(seg_name)
        self._set_current_segmentation(self._blank_segmentation())
        self.segmentation_combo_box.addItem(seg_name)
        self.segmentation_combo_box.setCurrentText(seg_name)
        self.refresh()

    def start_segmentation_from_scratch(self):
        seg_name = self._prompt_for_name()
        if seg_name is None:
            return
        self.seg_name = seg_name
        self._assign_colour(seg_name)
        self._set_current_segmentation(self._blank_segmentation())
        if self.segmentation_combo_box.findText(seg_name) < 0:
            self.segmentation_combo_box.addItem(seg_name)
        self.segmentation_combo_box.setCurrentText(seg_name)
        self._enter_editing_mode()

    def select_segmentation(self):
        seg_name = self.segmentation_combo_box.currentText()
        if seg_name in self.source_segmentation:
            self.seg_name = seg_name
            # Snapshots hold slices for one segmentation at one orientation;
            # switching segmentation makes older snapshots the wrong shape.
            self.history.clear()
            self.refresh()

    def show_roi_statistics(self):
        """Report intensity statistics for the current mask."""
        if self.segmentation_data is None:
            QMessageBox.warning(self, "Segmentation", "No segmentation data available!")
            return
        stats = roi_statistics(
            self.segmentation_data, self.display_data, self.volume_conversion_units
        )
        if stats is None:
            QMessageBox.information(self, "ROI Statistics", "The current segmentation is empty.")
            return
        QMessageBox.information(
            self, "ROI Statistics",
            f"Segmentation: {self.seg_name}\n"
            f"Voxels: {stats['voxel_count']:,}\n"
            f"Volume: {stats['volume_cm3']:.2f} cm3\n\n"
            f"Mean HU: {stats['mean_hu']:.1f}\n"
            f"Std HU:  {stats['std_hu']:.1f}\n"
            f"Min HU:  {stats['min_hu']:.0f}\n"
            f"Max HU:  {stats['max_hu']:.0f}",
        )
        return stats

    def rename_segmentation(self):
        """Rename the currently selected segmentation."""
        if self.seg_name is None:
            return
        new_name, ok = QInputDialog.getText(
            self, "Rename Segmentation",
            "New name:", text=self.seg_name,
        )
        if not ok:
            return
        new_name = new_name.strip()
        if not new_name or new_name == self.seg_name:
            return
        if new_name in self.source_segmentation:
            QMessageBox.warning(self, "Rename", f"'{new_name}' already exists!")
            return

        old_name = self.seg_name
        self.source_segmentation[new_name] = self.source_segmentation.pop(old_name)
        self.segmentation_colors[new_name] = self.segmentation_colors.pop(
            old_name, SEGMENTATION_COLORS[0]
        )
        self.seg_name = new_name

        index = self.segmentation_combo_box.findText(old_name)
        if index >= 0:
            self.segmentation_combo_box.setItemText(index, new_name)
        self.is_modified = True
        self.refresh()

    def delete_segmentation(self):
        """Remove the currently selected segmentation entirely."""
        if self.seg_name is None:
            return
        if not self._confirm(
            "Delete Segmentation",
            f"Remove the segmentation '{self.seg_name}'? This cannot be undone.",
        ):
            return

        removed = self.seg_name
        self.source_segmentation.pop(removed, None)
        self.segmentation_colors.pop(removed, None)
        self.history.clear()

        index = self.segmentation_combo_box.findText(removed)
        if index >= 0:
            # Removing the item triggers select_segmentation for the next one.
            self.segmentation_combo_box.removeItem(index)

        self.seg_name = self.segmentation_combo_box.currentText() or None
        if self.seg_name not in self.source_segmentation:
            self.seg_name = next(iter(self.source_segmentation), None)
        self.is_modified = True
        self.refresh()

    # ------------------------------------------------------------------
    # Bulk edits
    # ------------------------------------------------------------------
    def delete_left(self):
        self._confirm_deletion(
            "Delete Left Half",
            "Are you sure you want to delete the left half of the segmentation data?",
            delete_left_half,
        )

    def delete_right(self):
        self._confirm_deletion(
            "Delete Right Half",
            "Are you sure you want to delete the right half of the segmentation data?",
            delete_right_half,
        )

    def _confirm_deletion(self, title, message, delete_func):
        if self.segmentation_data is None:
            return
        if not self._confirm(title, message):
            return
        self._snapshot()
        edited = delete_func(self.segmentation_data)
        self._set_current_segmentation([s > config.MASK_THRESHOLD for s in edited])
        self.is_modified = True
        self.refresh()

    def delete_slice(self):
        if self.segmentation_data is None:
            return
        self._snapshot()
        index = self.slider.value()
        self.segmentation_data[index] = zeros_like(self.segmentation_data[index])
        self.is_modified = True
        self.refresh()

    def delete_all(self):
        if self.segmentation_data is None:
            return
        if not self._confirm(
            "Deleting Whole Segmentation",
            "Are you sure you want to delete all of the segmentation data "
            "to start again from scratch?",
        ):
            return
        self._snapshot()
        self._set_current_segmentation(self._blank_segmentation())
        self.is_modified = True
        self.refresh()

    def _confirm(self, title, message):
        reply = QMessageBox.question(
            self, title, message,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return reply == QMessageBox.StandardButton.Yes

    def interpolate(self):
        if self.segmentation_data is None:
            return
        self._snapshot()
        self.interpolation_worker = InterpolationWorker(self.segmentation_data, self.display_data)
        self.interpolation_worker.interpolation_complete.connect(self.on_interpolation_complete)
        self.interpolate_button.setText("Interpolating...")
        self.interpolate_button.setEnabled(False)
        self.interpolation_worker.start()

    def on_interpolation_complete(self, interpolated_data):
        self._set_current_segmentation(
            [s > config.MASK_THRESHOLD for s in interpolated_data]
        )
        self.is_modified = True
        self.interpolate_button.setText("Interpolate Segmentation")
        self.interpolate_button.setEnabled(True)
        self.refresh()

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------
    def toggle_contour_mode(self):
        self.drawing_contour = not self.drawing_contour
        self.toggle_brush_button.setVisible(not self.drawing_contour)
        self.toggle_contour_button.setVisible(self.drawing_contour)
        if not self.drawing_contour:
            self.contour_points = []
        self.refresh()

    @property
    def three_d_brush(self):
        return self.three_d_brush_tick_box.isChecked()

    def add_contour_point(self, point):
        scale_x, scale_y = self._display_scale()
        self.contour_points.append((int(point.x() * scale_x), int(point.y() * scale_y)))
        self.refresh()

    def complete_contour(self):
        if len(self.contour_points) > 2:
            self.fill_contour()
            self.contour_points = []
        self.refresh()

    def fill_contour(self):
        """Fill the polygon the user has clicked out on the current slice."""
        index = self.slider.value()
        if self.segmentation_data is None or index >= len(self.segmentation_data):
            return

        self._snapshot()
        filled = zeros_like(self.segmentation_data[index], dtype=uint8)
        contour = array(self.contour_points, dtype=int32).reshape((-1, 1, 2))
        fillPoly(filled, [contour], 255)
        self.segmentation_data[index] = logical_or(
            self.segmentation_data[index], filled == 255
        )

        # With the 3D brush on, bleed a shrinking copy into neighbouring slices
        # so a contour drawn on one slice produces a rounded 3D blob.
        if self.three_d_brush:
            for dz in range(-BRUSH_3D_REACH, BRUSH_3D_REACH + 1):
                if dz == 0:
                    continue
                slice_idx = index + dz
                if not 0 <= slice_idx < len(self.segmentation_data):
                    continue
                strength = BRUSH_3D_REACH - abs(dz)
                size = max(1, strength * 2 + 1)
                kernel = getStructuringElement(MORPH_ELLIPSE, (size, size))
                eroded = erode(filled, kernel)
                self.segmentation_data[slice_idx] = logical_or(
                    self.segmentation_data[slice_idx], eroded == 255
                )

        self.is_modified = True
        self.update_image_display(index)

    def edit_segmentation(self, pos, adding):
        """Paint or erase a sphere of voxels centred on ``pos``."""
        if self.segmentation_data is None:
            return

        radius = self.brush_size_spinbox.value()
        scale_x, scale_y = self._display_scale()
        img_x = int(pos.x() * scale_x)
        img_y = int(pos.y() * scale_y)
        img_z = self.slider.value()

        # Vectorised stamp: identical result to the old triple loop, but numpy
        # applies each slice's disk in one operation instead of per-voxel.
        stamp_brush(
            self.segmentation_data, img_x, img_y, img_z, radius, adding,
            three_d=self.three_d_brush, z_reach=BRUSH_3D_REACH,
        )

        self.is_modified = True
        self.update_image_display(img_z)

    def eventFilter(self, source, event):
        if source is self.image_label:
            self._handle_image_event(event)

        if event.type() == QEvent.Type.Wheel:
            if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                # Shift+wheel zooms; scrollbars then pan the enlarged image.
                self.set_zoom(self.zoom * (ZOOM_STEP if event.angleDelta().y() > 0 else 1 / ZOOM_STEP))
            elif event.modifiers() & Qt.KeyboardModifier.ControlModifier:
                if self.segmentation_data is not None:
                    step = 1 if event.angleDelta().y() > 0 else -1
                    self.brush_size_spinbox.setValue(self.brush_size_spinbox.value() + step)
                    self.refresh()
            else:
                step = -1 if event.angleDelta().y() > 0 else 1
                self.slider.setValue(self.slider.value() + step)
        elif event.type() == QEvent.Type.KeyPress:
            if event.key() == Qt.Key.Key_Left:
                self.slider.setValue(self.slider.value() - 1)
            elif event.key() == Qt.Key.Key_Right:
                self.slider.setValue(self.slider.value() + 1)

        return super().eventFilter(source, event)

    def _handle_image_event(self, event):
        # Middle-button drag adjusts the HU window/level, the way clinical
        # viewers do, and takes priority over everything else.
        if self._handle_window_drag(event):
            return

        if self.drawing_contour:
            if event.type() == QEvent.Type.MouseButtonPress:
                if event.buttons() & Qt.MouseButton.LeftButton:
                    self.add_contour_point(event.position().toPoint())
                elif event.buttons() & Qt.MouseButton.RightButton and len(self.contour_points) > 2:
                    self.complete_contour()
            return

        if event.type() not in (QEvent.Type.MouseMove, QEvent.Type.MouseButtonPress):
            return
        if self.segmentation_data is None:
            return

        painting = bool(event.buttons() & (Qt.MouseButton.LeftButton | Qt.MouseButton.RightButton))

        # Snapshot once at the start of a stroke so a whole drag is one undo.
        if event.type() == QEvent.Type.MouseButtonPress and painting:
            self._snapshot()

        if event.type() == QEvent.Type.MouseMove:
            self.mouse_pos = event.position().toPoint()
            self._update_pixel_readout()
            self.refresh()

        # Left button paints, right button erases.
        if painting:
            self.edit_segmentation(
                event.position().toPoint(),
                bool(event.buttons() & Qt.MouseButton.LeftButton),
            )

    def _handle_window_drag(self, event):
        """Adjust window width/level from a middle-button drag. Returns handled."""
        if event.type() == QEvent.Type.MouseButtonPress and event.button() == Qt.MouseButton.MiddleButton:
            self._window_drag_origin = (
                event.position().toPoint(),
                self.norm_min_spinbox.value(),
                self.norm_max_spinbox.value(),
            )
            return True
        if event.type() == QEvent.Type.MouseMove and self._window_drag_origin is not None:
            start, start_min, start_max = self._window_drag_origin
            pos = event.position().toPoint()
            # Horizontal drag widens/narrows the window; vertical shifts level.
            width_delta = (pos.x() - start.x()) * WINDOW_DRAG_SENSITIVITY
            level_delta = (pos.y() - start.y()) * WINDOW_DRAG_SENSITIVITY
            new_min = start_min + level_delta - width_delta // 2
            new_max = start_max + level_delta + width_delta // 2
            if new_max - new_min >= 2:
                self.norm_min_spinbox.setValue(int(new_min))
                self.norm_max_spinbox.setValue(int(new_max))
            return True
        if event.type() == QEvent.Type.MouseButtonRelease and event.button() == Qt.MouseButton.MiddleButton:
            self._window_drag_origin = None
            return True
        return False

    # ------------------------------------------------------------------
    # Zoom + undo/redo
    # ------------------------------------------------------------------
    def set_zoom(self, zoom):
        """Set the zoom factor, clamped, and redraw."""
        self.zoom = max(ZOOM_MIN, min(ZOOM_MAX, zoom))
        self.refresh()

    def _snapshot(self):
        """Record the current segmentation so the next edit can be undone."""
        if self.segmentation_data is not None:
            self.history.push(self.segmentation_data)

    def undo(self):
        if self.segmentation_data is None:
            return
        restored = self.history.undo(self.segmentation_data)
        if restored is None:
            return
        self._set_current_segmentation(restored)
        self.is_modified = True
        self.refresh()

    def redo(self):
        if self.segmentation_data is None:
            return
        restored = self.history.redo(self.segmentation_data)
        if restored is None:
            return
        self._set_current_segmentation(restored)
        self.is_modified = True
        self.refresh()

    def _update_pixel_readout(self):
        scan_data = self.display_data
        height, width = scan_data[0].shape
        scale_x, scale_y = self._display_scale()
        img_x = int(self.mouse_pos.x() * scale_x)
        img_y = int(self.mouse_pos.y() * scale_y)
        if 0 <= img_x < width and 0 <= img_y < height:
            self.pixel_value_label.setText(f"HU: {scan_data[self.slider.value()][img_y, img_x]:.0f}")
        else:
            self.pixel_value_label.setText("HU: N/A")

    # ------------------------------------------------------------------
    # 3D view
    # ------------------------------------------------------------------
    def _segmentations_to_render(self):
        if self.view_all_segmentations_checkbox.isChecked():
            return dict(self.source_segmentation)
        if self.segmentation_data is None:
            return {}
        return {self.seg_name: self.segmentation_data}

    def view_3d_segmentation(self):
        segmentations = self._segmentations_to_render()
        if not segmentations:
            QMessageBox.warning(self, "Warning", "No segmentation data available to visualise!")
            return

        renderer = vtk_view.build_renderer(segmentations, self.segmentation_colors)
        self.vtk_frame.show()
        vtk_view.show_in_widget(self.vtk_widget, renderer)
        self.view_3d_button.setVisible(False)
        self.close_3d_button.setVisible(True)

    def close_3d_segmentation(self):
        self.vtk_frame.hide()
        self.view_3d_button.setVisible(True)
        self.close_3d_button.setVisible(False)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    def _ask_for_gif_path(self):
        if self.segmentation_data is None:
            QMessageBox.warning(self, "Warning", "No segmentation data available to export!")
            return None
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Segmentation as GIF", "", "GIF Files (*.gif)"
        )
        return path or None

    def export_gif_with_scan(self):
        path = self._ask_for_gif_path()
        if path is None:
            return
        export.export_overlay_gif(
            path, self.segmentation_data, self.display_data,
            self.norm_min_spinbox.value(), self.norm_max_spinbox.value(),
        )
        QMessageBox.information(self, "Export Complete", f"Segmentation exported as {path}")

    def export_gif_without_scan(self):
        path = self._ask_for_gif_path()
        if path is None:
            return
        export.export_mask_gif(path, self.segmentation_data)
        QMessageBox.information(self, "Export Complete", f"Segmentation exported as {path}")

    def export_3d_segmentation(self):
        path = self._ask_for_gif_path()
        if path is None:
            return
        # The rotation is captured from the live render window, so make sure
        # there is one showing the current segmentation first.
        self.view_3d_segmentation()
        export.export_rotation_gif(path, self.vtk_widget.GetRenderWindow())
        QMessageBox.information(self, "Export Complete", f"3D segmentation exported as {path}")

    # ------------------------------------------------------------------
    # Generating a segmentation with the model
    # ------------------------------------------------------------------
    def load_or_generate_segmentation(self):
        """Offer to load saved masks for this scan, else run the model."""
        npy_files = sorted(Path(self.current_directory).glob("*.npy"))

        if not npy_files:
            self.start_ai_segmentation()
            return

        reply = QMessageBox.question(
            self, "Segmentation Exists",
            "Segmentation files already exist for this scan. Do you want to load them?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )

        if reply == QMessageBox.StandardButton.Yes:
            self._load_saved_segmentations(npy_files)
        elif reply == QMessageBox.StandardButton.No:
            self.start_ai_segmentation()

    def _load_saved_segmentations(self, npy_files):
        # Saved masks are stored transversally, so make sure that is what is
        # on screen before loading them.
        self.set_plane("Transversal")

        loaded = []
        for npy_file in npy_files:
            seg_name = seg_name_from_npy(self.current_directory, npy_file)
            volume = load(npy_file).astype(bool)
            # Undo the flips applied on save (see _save_single_segmentation).
            volume = flip(volume[::-1, :, :], axis=1)
            self.source_segmentation[seg_name] = [volume[i] for i in range(volume.shape[0])]
            self._assign_colour(seg_name)
            if self.segmentation_combo_box.findText(seg_name) < 0:
                self.segmentation_combo_box.addItem(seg_name)
            loaded.append(seg_name)

        if not loaded:
            return

        self.seg_name = loaded[0]
        self.history.clear()
        self.segmentation_combo_box.setCurrentText(self.seg_name)
        QMessageBox.information(
            self, "Segmentation", f"Loaded existing segmentations: {', '.join(loaded)}!"
        )
        self._enter_editing_mode()

    def start_ai_segmentation(self):
        self.seg_name = AI_SEGMENTATION_NAME
        self._assign_colour(self.seg_name)
        self.generate_segmentation_button.setVisible(False)
        self.start_segmentating_button.setVisible(False)
        self.progressBar.setVisible(True)
        self.timeLabel.setVisible(True)

        self.elapsed_timer = QElapsedTimer()
        self.elapsed_timer.start()
        self.worker = SegmentationWorker(folder=self.current_directory)
        self.worker.stage_changed.connect(self.timeLabel.setText)
        self.worker.segmentation_complete.connect(self.on_segmentation_complete)
        self.worker.start()

    def on_segmentation_complete(self, results):
        self.progressBar.setVisible(False)
        self.timeLabel.setVisible(False)

        # The worker reports None when segmentation failed; don't fabricate a
        # mask, just tell the user and let them draw one or try again.
        if results is None:
            QMessageBox.warning(
                self, "Segmentation Failed",
                "The model could not segment this scan. See the log for details.\n"
                "You can still draw a segmentation by hand.",
            )
            self.generate_segmentation_button.setVisible(True)
            self.start_segmentating_button.setVisible(True)
            return

        self.history.clear()
        self._set_current_segmentation([results[i] for i in range(len(results))])
        if self.segmentation_combo_box.findText(self.seg_name) < 0:
            self.segmentation_combo_box.addItem(self.seg_name)
        self.segmentation_combo_box.setCurrentText(self.seg_name)

        self._enter_editing_mode()
        self.save_segmentation()

    # ------------------------------------------------------------------
    # Saving
    # ------------------------------------------------------------------
    def save_segmentation(self):
        if self.segmentation_data is None:
            QMessageBox.warning(self, "Segmentation", "No segmentation data to save!")
            return

        # Offer the current segmentation first, then the rest, then "All".
        choices = [self.seg_name]
        choices += [n for n in self.source_segmentation if n != self.seg_name]
        if len(self.source_segmentation) > 1:
            choices.append("All")

        seg_name, ok = QInputDialog.getItem(
            self, "Select Segmentation", "Choose segmentation to save:", choices, 0, False
        )
        if not ok or not seg_name:
            return

        # Masks are always written in the transversal orientation.
        self.set_plane("Transversal")

        if seg_name == "All":
            for name in self.source_segmentation:
                self._save_single_segmentation(name)
            QMessageBox.information(self, "Segmentation", "All segmentations saved successfully!")
        else:
            self._save_single_segmentation(seg_name)
            QMessageBox.information(
                self, "Segmentation", f"Segmentation '{seg_name}' saved successfully!"
            )
        self.is_modified = False

    def _save_single_segmentation(self, seg_name):
        """Write one segmentation as both ``.npy`` and RTSTRUCT.

        The flips here are the inverse of those applied when the scan was
        loaded, so the saved volume matches the original DICOM slice order.
        """
        mask = stack(self.source_segmentation[seg_name])
        mask = flip(mask, axis=1)[::-1, :, :]

        save_mask_npy(self.current_directory, seg_name, mask)
        save_mask_nifti(self.current_directory, seg_name, mask)

        # rt_utils wants (row, column, slice); the .npy is (slice, row, column).
        contour_mask = (transpose(mask, (1, 2, 0)) > config.MASK_THRESHOLD).astype(bool)
        save_rtstruct(
            self.current_directory, seg_name, contour_mask, f"rtstruct_{seg_name}.dcm"
        )

    # ------------------------------------------------------------------
    # Closing
    # ------------------------------------------------------------------
    def _persist_window_level(self):
        """Remember the current window/level for the next scan opened."""
        settings = config.load_settings()
        settings["window_min_hu"] = self.norm_min_spinbox.value()
        settings["window_max_hu"] = self.norm_max_spinbox.value()
        config.save_settings(settings)

    def closeEvent(self, event):
        """Offer to save before closing, however the window was closed."""
        self._persist_window_level()
        if not self.is_modified:
            event.accept()
            return

        reply = QMessageBox.question(
            self, "Save Changes",
            "Segmentation has been modified. Do you want to save changes?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )

        if reply == QMessageBox.StandardButton.Yes:
            self.save_segmentation()
            event.accept()
        elif reply == QMessageBox.StandardButton.No:
            event.accept()
        else:
            event.ignore()
