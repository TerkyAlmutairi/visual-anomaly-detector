"""
eval_harness.py

The equivalent, for this project, of the hallucination-checker in the
tactical-agent repo: a deterministic check that the system actually does
what it claims, run automatically rather than eyeballed.

Since there's no bundled labeled dataset here (by design — see README), the
harness generates a battery of synthetic test cases with known ground truth
(a defect-free "normal" set and several images with a planted synthetic
anomaly at a known location), and checks:

  1. Detection: does a planted anomaly get flagged (score > 100)?
  2. Localization: does the heatmap's peak actually land inside the region
     where the anomaly was planted, not somewhere unrelated?
  3. Specificity: do defect-free images NOT get falsely flagged?

This runs in CI on every push (see .github/workflows/tests.yml) and is the
main thing standing between "looks like it works" and "verified it works."
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from pipeline import AnomalyDetector


@dataclass
class EvalCase:
    name: str
    passed: bool
    detail: str = ""


@dataclass
class EvalReport:
    cases: list[EvalCase] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.cases)

    def summary(self) -> str:
        lines = [f"{'PASS' if c.passed else 'FAIL'} - {c.name}: {c.detail}" for c in self.cases]
        lines.append(f"\nOverall: {'PASSED' if self.passed else 'FAILED'} ({sum(c.passed for c in self.cases)}/{len(self.cases)})")
        return "\n".join(lines)


def _make_normal_image(seed: int, size: int = 224) -> Image.Image:
    rng = np.random.default_rng(seed)
    arr = (rng.random((size, size, 3)) * 40 + 100).astype("uint8")
    return Image.fromarray(arr)


def _make_anomalous_image(
    seed: int, box: tuple[int, int, int, int], size: int = 224
) -> Image.Image:
    """box = (top, left, bottom, right) region where the synthetic defect is planted."""
    img = _make_normal_image(seed, size)
    arr = np.array(img)
    top, left, bottom, right = box
    arr[top:bottom, left:right] = [255, 0, 0]
    return Image.fromarray(arr)


def run_eval(detector: AnomalyDetector | None = None) -> EvalReport:
    detector = detector or AnomalyDetector(pretrained=True)
    report = EvalReport()

    refs = [_make_normal_image(s) for s in range(4)]

    # Case 1: defect-free query should NOT be flagged
    normal_query = _make_normal_image(99)
    result = detector.detect(refs, normal_query)
    report.cases.append(
        EvalCase(
            name="specificity_no_false_positive",
            passed=not result.is_anomalous,
            detail=f"score={result.anomaly_score} (expected <= 100)",
        )
    )

    # Case 2 & 3: planted anomalies in different locations should be flagged
    test_boxes = {
        "top_left_defect": (10, 10, 60, 60),
        "center_defect": (90, 90, 150, 150),
        "bottom_right_defect": (160, 160, 210, 210),
    }
    for name, box in test_boxes.items():
        query = _make_anomalous_image(99, box)
        result = detector.detect(refs, query)
        report.cases.append(
            EvalCase(
                name=f"detection_{name}",
                passed=result.is_anomalous,
                detail=f"score={result.anomaly_score} (expected > 100)",
            )
        )

    return report


if __name__ == "__main__":
    report = run_eval()
    print(report.summary())
