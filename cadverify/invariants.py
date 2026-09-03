"""Stage A — load a STEP file and measure quantities that survive rigid motion.

Everything here comes off the B-rep directly. No tessellation: OpenCascade
integrates the exact surfaces, so these are true values, not mesh approximations.
"""

from dataclasses import dataclass, field

from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCP.BRepGProp import BRepGProp
from OCP.GProp import GProp_GProps
from OCP.IFSelect import IFSelect_RetDone
from OCP.STEPControl import STEPControl_Reader
from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_SHELL, TopAbs_SOLID, TopAbs_VERTEX
from OCP.TopExp import TopExp_Explorer
from OCP.TopoDS import TopoDS_Shape
from OCP.gp import gp_Trsf


def load_step(path) -> TopoDS_Shape:
    reader = STEPControl_Reader()
    if reader.ReadFile(str(path)) != IFSelect_RetDone:
        raise IOError(f"STEP read failed: {path}")
    reader.TransferRoots()
    shape = reader.OneShape()
    if shape.IsNull():
        raise ValueError(f"STEP produced a null shape: {path}")
    return shape


def _count(shape, topo_type) -> int:
    exp, n = TopExp_Explorer(shape, topo_type), 0
    while exp.More():
        n += 1
        exp.Next()
    return n


@dataclass
class Invariants:
    """Quantities that do not change under rotation or translation."""

    volume: float
    area: float
    moments: tuple           # principal moments of inertia, about the centre of mass
    n_faces: int
    n_edges: int
    n_vertices: int
    n_shells: int
    n_solids: int
    has_symmetry_axis: bool  # principal axes are ambiguous -> alignment needs a fallback
    has_symmetry_point: bool
    # not invariant, carried for stage B only:
    com: tuple = field(default=(0.0, 0.0, 0.0))


def invariants(shape) -> Invariants:
    # OpenCascade already references the inertia tensor to the centre of mass,
    # so PrincipalProperties() needs no parallel-axis correction. Verified: a
    # 1000mm translation moves the COM by exactly 1000mm and leaves Ixx and the
    # principal moments bit-identical. Do not "fix" this by re-referencing.
    g = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, g)
    c = g.CentreOfMass()
    pp = g.PrincipalProperties()

    g_area = GProp_GProps()
    BRepGProp.SurfaceProperties_s(shape, g_area)

    return Invariants(
        volume=g.Mass(),
        area=g_area.Mass(),
        moments=tuple(sorted(pp.Moments())),   # sorted: axis labelling is arbitrary
        n_faces=_count(shape, TopAbs_FACE),
        n_edges=_count(shape, TopAbs_EDGE),
        n_vertices=_count(shape, TopAbs_VERTEX),
        n_shells=_count(shape, TopAbs_SHELL),
        n_solids=_count(shape, TopAbs_SOLID),
        has_symmetry_axis=pp.HasSymmetryAxis(),
        has_symmetry_point=pp.HasSymmetryPoint(),
        com=(c.X(), c.Y(), c.Z()),
    )


def transformed(shape, trsf: gp_Trsf) -> TopoDS_Shape:
    """Apply a rigid transform. Used to test that invariants really are invariant."""
    return BRepBuilderAPI_Transform(shape, trsf, True).Shape()
