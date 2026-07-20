"""
Unit tests. Deliberately use pretrained=False (randomly initialized weights)
so these run fast and offline in CI without needing to download ~270MB of
ImageNet weights on every push. This tests the *pipeline logic* (shapes,
calibration, thresholding, coreset subsampling) independent of feature
quality — see eval_harness.py for the fuller, semantically meaningful checks
intended to be run locally with real pretrained weights.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch  # noqa: E402
from PIL import Image  # noqa: E402
import numpy as np  # noqa: E402

from feature_extractor import PatchFeatureExtractor  # noqa: E402
from memory_bank import MemoryBank  # noqa: E402
from pipeline import AnomalyDetector  # noqa: E402
from eval_harness import run_eval  # noqa: E402


def _random_image(seed=0, size=224):
    rng = np.random.default_rng(seed)
    arr = (rng.random((size, size, 3)) * 255).astype("uint8")
    return Image.fromarray(arr)


def test_feature_extractor_output_shape():
    extractor = PatchFeatureExtractor(pretrained=False)
    patches, grid = extractor.extract(_random_image())
    h, w = grid
    assert patches.shape[0] == h * w
    assert patches.shape[1] > 0


def test_memory_bank_respects_max_size():
    torch.manual_seed(0)
    embeddings = [torch.randn(500, 32) for _ in range(3)]  # 1500 total patches
    bank = MemoryBank(max_size=200)
    bank.build(embeddings)
    assert bank.bank.shape[0] == 200


def test_memory_bank_no_subsampling_when_under_limit():
    torch.manual_seed(0)
    embeddings = [torch.randn(50, 32) for _ in range(2)]  # 100 total patches
    bank = MemoryBank(max_size=200)
    bank.build(embeddings)
    assert bank.bank.shape[0] == 100


def test_memory_bank_scores_are_nonnegative():
    torch.manual_seed(0)
    bank = MemoryBank(max_size=200)
    bank.build([torch.randn(100, 32)])
    scores = bank.anomaly_scores(torch.randn(50, 32))
    assert (scores >= 0).all()


def test_pipeline_requires_at_least_one_reference():
    detector = AnomalyDetector(pretrained=False)
    import pytest

    with pytest.raises(ValueError):
        detector.detect([], _random_image())


def test_full_eval_harness_passes_with_random_weights():
    # Even with random (untrained) weights, planted synthetic anomalies
    # should separate cleanly from the calibrated "normal" noise floor --
    # if this fails, something is broken in the pipeline logic itself,
    # independent of feature quality.
    detector = AnomalyDetector(pretrained=False)
    report = run_eval(detector)
    assert report.passed, report.summary()
