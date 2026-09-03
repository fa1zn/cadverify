"""Point-to-surface distance.

The first prefilter written for this code was silently wrong: a plain
k-nearest-centroid search reported 0.80% of the bbox diagonal for points known
to lie exactly on the surface. Every test here exists to keep that from
returning.
"""

import numpy as np
import pytest

from cadverify.invariants import load_step
from cadverify.sampling import Surface, _point_triangle_distance, bbox_diagonal, sample_surface

from .conftest import ref_step


def test_point_triangle_matches_brute_force():
    """Analytic distance never overshoots a dense sampling of the triangle."""
    rng = np.random.default_rng(0)
    n = 2000
    tri = rng.normal(size=(n, 3, 3)) * 10
    pts = rng.normal(size=(n, 3)) * 10
    mine = _point_triangle_distance(pts, tri)

    steps = 40
    u, v = [], []
    for i in range(steps + 1):
        for j in range(steps + 1 - i):
            u.append(i / steps)
            v.append(j / steps)
    u = np.array(u)[None, :, None]
    v = np.array(v)[None, :, None]
    a, b, c = tri[:, 0:1], tri[:, 1:2], tri[:, 2:3]
    grid = a + u * (b - a) + v * (c - a)
    brute = np.linalg.norm(grid - pts[:, None, :], axis=2).min(axis=1)

    assert np.all(mine <= brute + 1e-9), "analytic distance exceeded a point on the triangle"
    assert np.abs(mine - brute).mean() < 0.01


@pytest.mark.parametrize(
    "tri,pt,expected",
    [
        ([[0, 0, 0], [1, 0, 0], [2, 0, 0]], [0.5, 1.0, 0.0], 1.0),   # collinear
        ([[0, 0, 0], [0, 0, 0], [0, 0, 0]], [3.0, 4.0, 0.0], 5.0),   # zero area
        ([[0, 0, 0], [1, 0, 0], [0, 1, 0]], [0.25, 0.25, 2.0], 2.0),  # face interior
    ],
)
def test_point_triangle_degenerate(tri, pt, expected):
    d = _point_triangle_distance(np.array([pt], float), np.array([tri], float))
    assert d[0] == pytest.approx(expected, abs=1e-9)


def test_points_on_surface_measure_zero(sample):
    """The noise floor must be identically zero, not merely small."""
    shape = load_step(ref_step(sample))
    surf = Surface(shape)
    pts = sample_surface(shape, 2000, seed=1)
    assert surf.distance(pts).max() == pytest.approx(0.0, abs=1e-9)


def test_fast_path_equals_exhaustive(corpus):
    """Pruning is a guarantee, not a heuristic — results must be identical."""
    rng = np.random.default_rng(4)
    for path in [corpus[i] for i in rng.choice(len(corpus), 3, replace=False)]:
        shape = load_step(ref_step(path))
        surf = Surface(shape)
        diag = bbox_diagonal(shape)
        pts = sample_surface(shape, 1200, seed=1)
        pts = pts + rng.normal(scale=diag * 0.01, size=pts.shape)
        assert np.abs(surf.distance(pts) - surf._exhaustive(pts)).max() < 1e-9
