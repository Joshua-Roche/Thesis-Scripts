"""Entry point for the JOSH CT viewer and segmentation tool.
"""

import logging
import multiprocessing as mp
import sys

from PySide6.QtWidgets import QApplication

from josh import config
from josh.ui.main_window import CTScanViewer

logger = logging.getLogger(__name__)


def main():
    log_path = config.configure_logging()
    logger.info("Logging to %s", log_path)
    logger.info("Model weights: %s", config.MODEL_DIR / "trained_models")
    logger.info("Compute device: %s", config.describe_device())

    config.clear_temp_dir()

    app = QApplication(sys.argv)
    window = CTScanViewer()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    mp.freeze_support()
    mp.set_start_method("spawn", force=True)
    main()
