"""Runtime configuration: paths, nnU-Net environment variables, compute device.
"""

import json
import logging
import logging.handlers
import os
import re
import sys
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

# When frozen by PyInstaller the bundled data lives in the extraction dir
# (``sys._MEIPASS``); in a source checkout it lives next to the package.
if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
    BUNDLE_DIR = Path(sys._MEIPASS)
else:
    BUNDLE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = BUNDLE_DIR / "model"
ASSETS_DIR = BUNDLE_DIR / "assets"

# nnU-Net looks these up by name, so they must be set before it is imported.
os.environ["RESULTS_FOLDER"] = str(MODEL_DIR / "trained_models")
os.environ["nnUNet_preprocessed"] = str(MODEL_DIR / "preprocessed")
os.environ["nnUNet_raw_data_base"] = str(MODEL_DIR / "raw_data")

# Scratch space for the NIfTI files handed to the model. Lives under the
# system temp dir so a crash cannot leave junk in the user's scan folders.
TEMP_PROCESSED_DIR = Path(tempfile.gettempdir()) / "josh_processed"

# The task the bundled weights were trained for.
TASK_NAME = "Task284_Mesothelioma"

# Voxels below this value in a float mask are treated as background. The model
# emits near-binary output, so this is deliberately permissive.
MASK_THRESHOLD = 0.008


def ensure_temp_dir() -> Path:
    """Create the scratch directory if it is missing and return it."""
    TEMP_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    return TEMP_PROCESSED_DIR


def clear_temp_dir() -> None:
    """Empty the scratch directory but keep the directory itself.

    Keeping the directory matters: a previous version removed it after each
    batch, which made the second batch in a session fail.
    """
    ensure_temp_dir()
    for path in sorted(TEMP_PROCESSED_DIR.rglob("*"), reverse=True):
        try:
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
        except OSError as exc:
            logger.warning("Could not remove temp path %s: %s", path, exc)


def _slugify(name: str) -> str:
    """Make ``name`` safe to use as a directory name."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", name) or "job"


def job_temp_dir(job_name: str) -> Path:
    """Return (creating if needed) a private scratch subdirectory for one job.

    Each segmentation job gets its own subdirectory so concurrent jobs cannot
    overwrite each other's intermediate NIfTI files, and so clearing one job's
    scratch space cannot touch another's.
    """
    path = ensure_temp_dir() / _slugify(job_name)
    path.mkdir(parents=True, exist_ok=True)
    return path


def remove_job_temp_dir(job_name: str) -> None:
    """Remove a job's private scratch subdirectory and its contents."""
    path = TEMP_PROCESSED_DIR / _slugify(job_name)
    if not path.exists():
        return
    for child in sorted(path.rglob("*"), reverse=True):
        try:
            child.unlink() if child.is_file() else child.rmdir()
        except OSError as exc:
            logger.warning("Could not remove %s: %s", child, exc)
    try:
        path.rmdir()
    except OSError as exc:
        logger.warning("Could not remove job temp dir %s: %s", path, exc)


# --- User settings -----------------------------------------------------------
# Small JSON file remembering things like the last-opened folder and window
# presets. Kept in the user's home so it survives reinstalls of the app.
SETTINGS_PATH = Path.home() / ".josh" / "settings.json"

DEFAULT_SETTINGS = {
    "last_open_folder": "",
    "window_min_hu": -135,
    "window_max_hu": 215,
}


def load_settings() -> dict:
    """Load persisted settings, falling back to defaults for missing keys."""
    settings = dict(DEFAULT_SETTINGS)
    try:
        with open(SETTINGS_PATH, encoding="utf-8") as handle:
            stored = json.load(handle)
        if isinstance(stored, dict):
            settings.update({k: v for k, v in stored.items() if k in DEFAULT_SETTINGS})
    except FileNotFoundError:
        pass
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not read settings from %s: %s", SETTINGS_PATH, exc)
    return settings


def save_settings(settings: dict) -> None:
    """Persist ``settings`` (only known keys are written)."""
    to_write = {k: settings[k] for k in DEFAULT_SETTINGS if k in settings}
    try:
        SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(SETTINGS_PATH, "w", encoding="utf-8") as handle:
            json.dump(to_write, handle, indent=2)
    except OSError as exc:
        logger.warning("Could not write settings to %s: %s", SETTINGS_PATH, exc)


def log_file_path() -> Path:
    """Where the rotating log file lives."""
    return Path.home() / ".josh" / "josh.log"


def configure_logging(level=logging.INFO) -> Path:
    """Send logs to both the console and a rotating file, and return its path.

    A file log matters most for the frozen executable, where the console may
    close on exit: it gives users something to send when reporting a problem.
    """
    log_path = log_file_path()
    handlers = [logging.StreamHandler()]
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            log_path, maxBytes=2_000_000, backupCount=3, encoding="utf-8"
        )
        handlers.append(file_handler)
    except OSError as exc:  # pragma: no cover - only if home is unwritable
        logger.warning("Could not open log file %s: %s", log_path, exc)

    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )
    return log_path


def describe_device() -> str:
    """Return a human-readable description of what the model will run on."""
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a hard dependency
        return "unknown (torch not importable)"
    if torch.cuda.is_available():
        return f"CUDA ({torch.cuda.get_device_name(0)})"
    return "CPU (no CUDA device found — segmentation will be slow)"
