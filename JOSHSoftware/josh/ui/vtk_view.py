"""Turning boolean segmentation volumes into 3D surfaces for the VTK viewer.

Marching cubes gives a triangle mesh of the mask boundary; the rest is the
boilerplate needed to hand that mesh to VTK.
"""

import logging

from numpy import ascontiguousarray, empty, float32, int64, stack
from skimage import measure
from vtk import (
    vtkActor,
    vtkCellArray,
    vtkPolyData,
    vtkPolyDataMapper,
    vtkPoints,
    vtkRenderer,
)
from vtkmodules.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray

logger = logging.getLogger(__name__)

DEFAULT_OPACITY = 0.65
DEFAULT_COLOR = (128, 128, 255)


def build_actor(slices, color=DEFAULT_COLOR, opacity=DEFAULT_OPACITY):
    """Build a semi-transparent surface actor for one segmentation.

    ``slices`` is the list of 2D boolean slices held by the viewer. Returns
    ``None`` for an empty mask, since marching cubes cannot mesh nothing.

    The mesh from marching cubes is handed to VTK in bulk via numpy: an earlier
    version inserted every vertex and every triangle one at a time in a Python
    loop, which dominated the cost of opening the 3D view on a large tumour.
    """
    volume = stack(slices)
    if not volume.any():
        return None

    vertices, faces, _, _ = measure.marching_cubes(volume, level=0)

    points = vtkPoints()
    points.SetData(numpy_to_vtk(ascontiguousarray(vertices, dtype=float32), deep=True))

    # VTK's legacy cell layout is a flat [3, i0, i1, i2, 3, j0, j1, j2, ...]
    # array; build it with numpy rather than a per-face Python loop.
    num_faces = len(faces)
    cells = empty((num_faces, 4), dtype=int64)
    cells[:, 0] = 3
    cells[:, 1:] = faces
    polygons = vtkCellArray()
    polygons.SetCells(num_faces, numpy_to_vtkIdTypeArray(cells.ravel(), deep=True))

    polydata = vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetPolys(polygons)

    mapper = vtkPolyDataMapper()
    mapper.SetInputData(polydata)

    actor = vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetOpacity(opacity)
    actor.GetProperty().SetColor(color[0] / 255, color[1] / 255, color[2] / 255)
    return actor


def build_renderer(segmentations, colors=None, opacity=DEFAULT_OPACITY):
    """Build a renderer showing every segmentation in ``segmentations``.

    ``segmentations`` maps name -> list of 2D slices. ``colors`` maps the same
    names to RGB tuples; names missing from it fall back to a neutral blue.
    """
    renderer = vtkRenderer()
    renderer.SetBackground(1, 1, 1)

    colors = colors or {}
    for name, slices in segmentations.items():
        actor = build_actor(slices, colors.get(name, DEFAULT_COLOR), opacity)
        if actor is None:
            logger.info("Skipping empty segmentation %r in 3D view", name)
            continue
        renderer.AddActor(actor)

    return renderer


def show_in_widget(vtk_widget, renderer):
    """Display ``renderer`` in ``vtk_widget``, replacing whatever was there.

    Clearing first matters: without it, every "View 3D" click stacks another
    renderer on the window and old meshes never disappear.
    """
    render_window = vtk_widget.GetRenderWindow()

    existing = render_window.GetRenderers()
    existing.InitTraversal()
    for _ in range(existing.GetNumberOfItems()):
        render_window.RemoveRenderer(existing.GetNextItem())

    render_window.AddRenderer(renderer)
    renderer.ResetCamera()
    render_window.Render()
    vtk_widget.GetRenderWindow().GetInteractor().Initialize()
