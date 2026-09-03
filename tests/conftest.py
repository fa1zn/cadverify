import glob
import json
import os

import pytest

CORPUS = os.environ.get(
    "CADVERIFY_CORPUS",
    os.path.expanduser("~/Downloads/collection-selection"),
)


def _samples():
    return sorted(glob.glob(os.path.join(CORPUS, "samples", "*", "*")))


@pytest.fixture(scope="session")
def corpus():
    paths = _samples()
    if not paths:
        pytest.skip(
            f"corpus not found at {CORPUS}. "
            "Set CADVERIFY_CORPUS to the collection-selection directory."
        )
    return paths


@pytest.fixture(scope="session")
def sample(corpus):
    """One representative part: a bracket with both planar and cylindrical faces."""
    return corpus[0]


def answer_key(path):
    return json.load(open(os.path.join(path, "sample.json")))["answerKey"]


def ref_step(path):
    return os.path.join(path, "ground_truth", "reference.step")


def sub_step(path):
    return os.path.join(path, "submission", "final.step")
