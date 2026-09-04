"""The audit has to catch the defects that actually shipped.

Fixtures are built on disk rather than mocked, because every one of these
corresponds to a real delivery and the check needs to survive the real layout.
"""

import json
import os

import pytest

from cadverify.audit import (NOTE, QUARANTINE, REJECT, Corpus, audit,
                             check_completeness, check_density_consistency,
                             check_duplicates, check_index_integrity, verdicts)


def _write_sample(root, sid, folder=None, *, hollow=False, density=0.0027,
                  material="Aluminum 6061", mass=None, volume=1000.0,
                  drawing=b"%PDF-1.4 fake"):
    folder = folder or f"track--{sid}"
    p = os.path.join(root, "samples", "track", folder)
    for sub in ("input", "submission", "ground_truth"):
        os.makedirs(os.path.join(p, sub), exist_ok=True)

    json.dump([] if hollow else [{"i": 1, "op": "add"}],
              open(os.path.join(p, "submission", "steps.json"), "w"))
    json.dump({"features": [] if hollow else [{"featureType": "extrude"}]},
              open(os.path.join(p, "submission", "final.features.json"), "w"))
    open(os.path.join(p, "input", "drawing.pdf"), "wb").write(drawing)

    meas = ({"volumeMm3": None, "massGrams": None} if hollow
            else {"volumeMm3": volume,
                  "massGrams": mass if mass is not None else volume * density})
    json.dump({"id": sid, "densityGPerMm3": density, "material": material,
               "difficulty": "easy", "units": "mm", "tolerancePercent": 3,
               "track": "track", "provenance": {"captureMode": "recipe-rebuild"},
               "submission": {"measurements": meas},
               "answerKey": {"targetVolumeMm3": volume}},
              open(os.path.join(p, "sample.json"), "w"))
    return p


@pytest.fixture
def good(tmp_path):
    root = str(tmp_path / "good")
    for i in range(4):
        _write_sample(root, f"aaaa{i:04d}-0000-0000-0000-000000000000",
                      drawing=b"%PDF-1.4 unique-" + str(i).encode())
    return root


def test_clean_corpus_produces_no_rejects(good):
    c, findings = audit(good, geometry=False)
    assert not [f for f in findings if f.severity == REJECT]
    assert set(verdicts(c, findings).values()) == {"accept"}


def test_catches_hollow_samples(tmp_path):
    """30 of these shipped in a delivery labelled '100 approved'."""
    root = str(tmp_path / "hollow")
    _write_sample(root, "aaaa0001-0000-0000-0000-000000000000", hollow=True,
                  drawing=b"%PDF-a")
    _write_sample(root, "aaaa0002-0000-0000-0000-000000000000",
                  drawing=b"%PDF-b")
    c = Corpus.load(root)
    f = list(check_completeness(c))
    assert len(f) == 1
    assert f[0].severity == REJECT
    assert "trajectory" in f[0].message and "measurements" in f[0].message


def test_catches_folder_name_mismatch(tmp_path):
    """6 folders in one delivery were named with an id absent from their contents."""
    root = str(tmp_path / "mismatch")
    _write_sample(root, "aaaa0001-0000-0000-0000-000000000000",
                  folder="track--bbbb0001-0000-0000-0000-000000000000")
    c = Corpus.load(root)
    f = [x for x in check_index_integrity(c) if x.check == "index"]
    assert len(f) == 1 and f[0].severity == QUARANTINE
    assert "folder is named" in f[0].message


def test_catches_wrong_density(tmp_path):
    """A steel part whose mass was computed at aluminium density, re-shipped
    unfixed after being reported."""
    root = str(tmp_path / "density")
    _write_sample(root, "aaaa0001-0000-0000-0000-000000000000",
                  material="Q235", density=0.00785, volume=1000.0, mass=2.7)
    c = Corpus.load(root)
    f = list(check_density_consistency(c))
    assert len(f) == 1 and f[0].severity == REJECT
    assert "Q235" in f[0].message


def test_density_check_tolerates_rounding(tmp_path):
    root = str(tmp_path / "ok")
    _write_sample(root, "aaaa0001-0000-0000-0000-000000000000",
                  density=0.00785, volume=1000.0, mass=7.851)
    assert not list(check_density_consistency(Corpus.load(root)))


def test_catches_duplicate_drawings(tmp_path):
    """70 of 150 samples in one delivery were re-ships from the previous one."""
    root = str(tmp_path / "dupes")
    for i in range(3):
        _write_sample(root, f"aaaa{i:04d}-0000-0000-0000-000000000000",
                      drawing=b"%PDF-identical")
    f = list(check_duplicates(Corpus.load(root)))
    assert len(f) == 1 and f[0].severity == QUARANTINE
    assert len(f[0].detail["ids"]) == 3


def test_dataset_jsonl_disagreement_is_a_reject(tmp_path):
    root = str(tmp_path / "idx")
    _write_sample(root, "aaaa0001-0000-0000-0000-000000000000", drawing=b"%PDF-a")
    with open(os.path.join(root, "DATASET.jsonl"), "w") as fh:
        fh.write(json.dumps({"id": "aaaa0001-0000-0000-0000-000000000000"}) + "\n")
        fh.write(json.dumps({"id": "cccc9999-0000-0000-0000-000000000000"}) + "\n")
    f = [x for x in check_index_integrity(Corpus.load(root)) if x.severity == REJECT]
    assert len(f) == 1
    assert "no sample on disk" in f[0].message


def test_geometry_check_skips_cleanly_without_opencascade(good, monkeypatch):
    """Structural checks must still run when the geometry stack is absent, and
    the geometry check must say it was skipped rather than silently pass."""
    import builtins

    real_import = builtins.__import__

    def blocked(name, *a, **k):
        if name.startswith("OCP") or name.endswith("invariants"):
            raise ImportError("simulated: no OpenCascade")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", blocked)
    from cadverify.audit import check_answer_key_matches_reference
    f = list(check_answer_key_matches_reference(Corpus.load(good)))
    assert len(f) == 1 and f[0].severity == NOTE and "skipped" in f[0].message


def test_verdict_severity_is_the_worst_finding(tmp_path):
    root = str(tmp_path / "mixed")
    _write_sample(root, "aaaa0001-0000-0000-0000-000000000000", hollow=True,
                  folder="track--bbbb0001-0000-0000-0000-000000000000",
                  drawing=b"%PDF-a")
    c, findings = audit(root, geometry=False)
    assert verdicts(c, findings)["aaaa0001-0000-0000-0000-000000000000"] == REJECT


# --------------------------------------------------------------------------
# baseline comparison — pinned so the headline claim cannot silently rot
# --------------------------------------------------------------------------

def test_standard_preprocessing_cannot_separate_correct_from_degenerate(corpus):
    """The comparison-against-prior-art claim, as a regression test.

    Centre-and-scale preprocessing — what most benchmarks document — scores a
    correct-but-rotated part about the same as a hollow box. Alignment is what
    restores the distinction, not a better metric.
    """
    import numpy as np
    from cadverify.baselines import compare
    from cadverify.exploit import POLICIES
    from cadverify.invariants import load_step, transformed
    from OCP.gp import gp_Ax1, gp_Dir, gp_Pnt, gp_Trsf

    from .conftest import answer_key, ref_step

    def rotate(s):
        t = gp_Trsf()
        t.SetRotation(gp_Ax1(gp_Pnt(0, 0, 0), gp_Dir(0.37, 0.61, 0.70)), 1.05)
        return transformed(s, t)

    rng = np.random.default_rng(42)
    naive, aligned = [], []
    for path in [corpus[i] for i in rng.choice(len(corpus), 4, replace=False)]:
        ref = load_step(ref_step(path))
        good, bad = rotate(ref), POLICIES["pocketed_block"](answer_key(path))
        naive.append(compare(ref, good, "center-scale")["fscore"]
                     - compare(ref, bad, "center-scale")["fscore"])
        aligned.append(compare(ref, good, "full-align")["fscore"]
                       - compare(ref, bad, "full-align")["fscore"])

    naive_gap, aligned_gap = float(np.median(naive)), float(np.median(aligned))
    assert aligned_gap > 0.3, "alignment should separate correct from degenerate"
    assert naive_gap < 0.2, "centre-scale should NOT separate them"
    assert aligned_gap > naive_gap * 3, (
        f"alignment widened the gap only {aligned_gap / max(naive_gap, 1e-9):.1f}x"
    )
