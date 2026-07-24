"""Exporting segmentations as animated GIFs.
"""

import logging

from numpy import ndarray, uint8, zeros
from PIL import Image
import vtkmodules.util.numpy_support as vnp
from vtk import vtkWindowToImageFilter

from ..imaging import window_to_uint8

logger = logging.getLogger(__name__)

SLICE_FRAME_MS = 70
# Frame delay for the 3D spin
ROTATION_FRAME_MS = 50
ROTATION_FRAMES = 72
OVERLAY_ALPHA = 0.5
OVERLAY_COLOR = (0, 255, 0)  # green reads clearly against greyscale CT


def _mask_to_rgba(mask_slice: ndarray, color=OVERLAY_COLOR, alpha=1.0) -> Image.Image:
    """Render one boolean mask slice as an RGBA image, transparent off-mask."""
    filled = (mask_slice > 0).astype(uint8) * 255
    rgba = zeros((filled.shape[0], filled.shape[1], 4), dtype=uint8)
    rgba[:, :, 0] = color[0]
    rgba[:, :, 1] = color[1]
    rgba[:, :, 2] = color[2]
    rgba[:, :, 3] = (filled * alpha).astype(uint8)
    return Image.fromarray(rgba)


def export_overlay_gif(output_path, segmentation_slices, scan_slices, hu_min, hu_max):
    """Write a GIF of the mask overlaid on the windowed scan.

    Compositing is done with the mask's alpha channel so unmasked areas keep
    the scan at full brightness.
    """
    frames = []
    for mask_slice, scan_slice in zip(segmentation_slices, scan_slices):
        background = Image.fromarray(
            window_to_uint8(scan_slice, hu_min, hu_max)
        ).convert("RGBA")
        overlay = _mask_to_rgba(mask_slice, alpha=OVERLAY_ALPHA)
        frames.append(Image.alpha_composite(background, overlay).convert("RGB"))

    _save_gif(output_path, frames, SLICE_FRAME_MS)
    return output_path


def export_mask_gif(output_path, segmentation_slices):
    """Write a GIF of the mask alone, white on black."""
    frames = [
        Image.fromarray((mask_slice > 0).astype(uint8) * 255)
        for mask_slice in segmentation_slices
    ]
    _save_gif(output_path, frames, SLICE_FRAME_MS)
    return output_path


def export_rotation_gif(output_path, render_window, num_frames=ROTATION_FRAMES):
    """Write a GIF of the 3D surface rotating once about the vertical axis."""
    capture = vtkWindowToImageFilter()
    capture.SetInput(render_window)
    capture.SetScale(1)
    capture.SetInputBufferTypeToRGB()
    # Read the back buffer: the front buffer may be occluded by other windows.
    capture.ReadFrontBufferOff()

    renderer = render_window.GetRenderers().GetFirstRenderer()
    camera = renderer.GetActiveCamera()
    renderer.ResetCamera()
    renderer.ResetCameraClippingRange()

    angle_step = 360.0 / num_frames
    frames = []
    for _ in range(num_frames):
        camera.Azimuth(angle_step)
        render_window.Render()
        capture.Modified()
        capture.Update()

        image = capture.GetOutput()
        width, height, _ = image.GetDimensions()
        pixels = vnp.vtk_to_numpy(image.GetPointData().GetScalars())
        frames.append(Image.fromarray(pixels.reshape(height, width, 3).astype(uint8)))

    _save_gif(output_path, frames, ROTATION_FRAME_MS)
    return output_path


def _save_gif(output_path, frames, duration_ms):
    if not frames:
        raise ValueError("Nothing to export: no frames were produced")
    frames[0].save(
        str(output_path),
        save_all=True,
        append_images=frames[1:],
        optimize=False,
        duration=duration_ms,
        loop=0,
    )
    logger.info("Exported %d frames to %s", len(frames), output_path)
