"""JOSH — a CT viewer and mesothelioma segmentation tool.

The package is laid out as:

``config``      Paths, environment variables and compute device. Must be
                imported before anything under ``MedNeXt``.
``dicom_io``    Reading DICOM series and converting them to NIfTI for the model.
``masks``       Saving segmentations to ``.npy`` / DICOM RTSTRUCT, and
                post-processing them.
``editing``     Numpy-level segmentation edits (delete half, interpolate).
``workers``     ``QThread`` wrappers so long jobs do not block the UI.
``ui``          The Qt widgets: scan browser and the viewer/editor window.

Run it with ``python main.py`` from the project root.
"""

__version__ = "2.0.0"
