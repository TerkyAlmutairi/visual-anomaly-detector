"""
pipeline.py

Wires together feature extraction, memory bank construction, and anomaly
scoring into a single callable pipeline. Also handles calibration: since a
raw patch-distance score has no inherent "this is anomalous" threshold, we
calibrate it using leave-one-out comparisons among the reference images
themselves (each reference image scored against a bank built from the
*other* reference images). This gives a data-driven noise floor — the score
a completely normal image gets just from natural variation — so the final
0-100 score is meaningful rather than arbitrary.
"""

from __future__ import annotations

from dataclasses import dataclass

from PIL import Image

from anomaly_scorer import AnomalyResult
from feature_extractor import PatchFeatureExtractor
from memory_bank import MemoryBank


@dataclass
class DetectionResult:
    anomaly_score: float  # 0-100+, calibrated
    is_anomalous: bool  # score > 100 threshold
    heatmap_overlay: Image.Image
    raw_max_score: float
    calibration_max: float


class AnomalyDetector:
    def __init__(self, pretrained: bool = True, bank_max_size: int = 5000):
        self.extractor = PatchFeatureExtractor(pretrained=pretrained)
        self.bank_max_size = bank_max_size

    def _extract_all(self, images: list[Image.Image]):
        embeddings, grids = [], []
        for img in images:
            patches, grid = self.extractor.extract(img)
            embeddings.append(patches)
            grids.append(grid)
        return embeddings, grids

    def calibrate(self, reference_images: list[Image.Image], embeddings: list) -> float:
        """
        Leave-one-out calibration: for each reference image, build a bank
        from the *other* references and score this one against it. The max
        across all these "normal vs normal" comparisons becomes the noise
        floor for the 0-100 scale.
        """
        if len(reference_images) < 2:
            # Can't do leave-one-out with a single reference image; fall back
            # to a conservative default. Documented limitation — see README.
            return 1.0

        calibration_scores = []
        for i in range(len(embeddings)):
            others = [e for j, e in enumerate(embeddings) if j != i]
            bank = MemoryBank(max_size=self.bank_max_size)
            bank.build(others)
            scores = bank.anomaly_scores(embeddings[i])
            calibration_scores.append(float(scores.max()))

        return max(calibration_scores) if calibration_scores else 1.0

    def detect(
        self, reference_images: list[Image.Image], query_image: Image.Image
    ) -> DetectionResult:
        if len(reference_images) < 1:
            raise ValueError("At least one reference image is required.")

        ref_embeddings, _ = self._extract_all(reference_images)
        calibration_max = self.calibrate(reference_images, ref_embeddings)

        bank = MemoryBank(max_size=self.bank_max_size)
        bank.build(ref_embeddings)

        query_patches, query_grid = self.extractor.extract(query_image)
        scores = bank.anomaly_scores(query_patches)
        result = AnomalyResult(scores, query_grid)

        heatmap = result.heatmap(output_size=(query_image.height, query_image.width))
        from anomaly_scorer import overlay_heatmap

        overlay = overlay_heatmap(query_image, heatmap)
        normalized = result.normalized_score(calibration_max)

        return DetectionResult(
            anomaly_score=normalized,
            is_anomalous=normalized > 100,
            heatmap_overlay=overlay,
            raw_max_score=result.raw_max_score,
            calibration_max=calibration_max,
        )


if __name__ == "__main__":
    import numpy as np

    def make_normal_image(seed: int) -> Image.Image:
        rng = np.random.default_rng(seed)
        arr = (rng.random((224, 224, 3)) * 40 + 100).astype("uint8")  # muted grey noise
        return Image.fromarray(arr)

    def make_anomalous_image(seed: int) -> Image.Image:
        img = make_normal_image(seed)
        arr = np.array(img)
        arr[80:140, 80:140] = [255, 0, 0]  # a bright red "defect" patch
        return Image.fromarray(arr)

    refs = [make_normal_image(s) for s in range(3)]
    detector = AnomalyDetector(pretrained=False)  # random weights: pipeline smoke test only

    normal_query = make_normal_image(99)
    anomaly_query = make_anomalous_image(99)

    r1 = detector.detect(refs, normal_query)
    r2 = detector.detect(refs, anomaly_query)
    print(f"Normal query score: {r1.anomaly_score}")
    print(f"Anomalous query score: {r2.anomaly_score}")
