# PatchSense — Few-Shot Visual Anomaly Detector

A few-shot visual anomaly detector: show it a handful of photos of what "normal" looks like, and it flags anything unlike them in a new photo — with a heatmap showing exactly where, and why.

No labeled defect data. No fixed object category. No training run required.

## Why this approach

Most anomaly detection demos either (a) need a large labeled dataset of both normal and defective examples, or (b) only work on one specific object category they were trained on. Neither matches how this actually gets used in the real world — inspection teams almost always have plenty of photos of "normal," but rarely have labeled examples of every possible defect type, because defects are by definition the things you haven't seen before.

This implements a simplified version of **[PatchCore](https://arxiv.org/abs/2106.08265)** (Roth et al., 2022) — a widely-cited, genuinely state-of-the-art approach on the standard industrial anomaly detection benchmark (MVTec AD). The core idea: use a pretrained CNN's mid-level features (which capture local texture and structure, not high-level semantics) as a general-purpose "does this patch look normal" detector, with no anomaly-specific training needed at all.

## How it works

```
reference images → feature extraction → memory bank (coreset)
                                                  │
query image → feature extraction → nearest-neighbor distance → heatmap + score
```

1. **Feature extraction** (`src/feature_extractor.py`) — a pretrained WideResNet-50 extracts patch-level embeddings from mid-network layers (layer2 + layer3, concatenated), following the paper's finding that this captures the right level of detail for texture/structure anomalies.
2. **Memory bank** (`src/memory_bank.py`) — patch embeddings from all reference images are pooled into a "memory" of normal patches. If there are more patches than the size cap, greedy coreset subsampling picks a representative subset (prioritizing coverage of the embedding space over redundant near-duplicates) rather than randomly discarding data.
3. **Scoring** (`src/anomaly_scorer.py`, `src/pipeline.py`) — each patch in the query image gets a distance to its nearest neighbor in the memory bank. This is reshaped into a heatmap and upsampled to the original resolution, and the overall score is calibrated against the natural variation *among the reference images themselves* (via leave-one-out comparison), so 100 means "as different as your own normal photos are from each other" — a meaningful, data-driven threshold rather than an arbitrary number.

## Example

Upload 3-5 photos of an undamaged surface (a wall, a product, a road, anything), then a photo to check. The output shows the original image next to a heatmap overlay, plus a numeric anomaly score and full scoring breakdown.

## Running it

```bash
git clone <this-repo>
cd visual-anomaly-detector
pip install -r requirements.txt

# Run tests (uses random weights, no download needed, fast + offline)
pytest tests/ -v

# Run the fuller semantic eval (needs internet, downloads ~270MB pretrained
# weights on first run only)
python src/eval_harness.py

# Launch the app
streamlit run app.py
```

## Testing approach — two tiers, deliberately

- **`tests/`** (CI, every push): uses `pretrained=False` (randomly initialized weights) so it runs in seconds with zero network dependency. This checks the *pipeline logic* — shapes, calibration math, coreset subsampling, thresholding — independent of feature quality.
- **`src/eval_harness.py`** (run locally, or manually in CI with weight caching): uses real pretrained ImageNet weights and checks actual detection + localization on synthetic planted anomalies at known locations. This is the meaningful check that the *detector* works, not just that the code runs.

Splitting it this way was a deliberate tradeoff: CI that needs to download 270MB on every push either gets slow or flaky. Separating "does the code work" from "does the model work" keeps CI fast while keeping the real check available.

## Deploying the live demo

[Streamlit Community Cloud](https://share.streamlit.io) — connect this repo, set `app.py` as the entrypoint. No API key or secrets needed (unlike an LLM-based project — this one only needs the pretrained weights, downloaded automatically on first load).

## Limitations (documented on purpose)

- **No bundled benchmark evaluation.** The standard benchmark for this technique (MVTec AD) has a non-commercial research license that makes it awkward to bundle in a public repo, so this ships dataset-agnostic instead — you bring your own reference images. The tradeoff: there's no single benchmark accuracy number to quote (e.g. "98% AUROC on MVTec AD"), because it's not evaluated against that fixed benchmark. If you want that number, `src/eval_harness.py` is straightforward to point at a locally downloaded copy of MVTec AD under its own license terms.
- **Calibration needs 2+ reference images** to compute a meaningful noise floor via leave-one-out comparison; with only one reference, it falls back to a conservative default (flagged clearly in the app).
- **Sensitive to viewpoint/lighting mismatch** between reference and query images — this is a known limitation of patch-matching approaches generally, not specific to this implementation. Reference and query images should be reasonably similar in framing and lighting for reliable results.
- **CNN backbone, not a foundation model** — a CLIP or DINOv2-based version would likely generalize better across very different domains, at the cost of a larger, slower backbone. WideResNet-50 was chosen to match the original PatchCore paper's setup.

## Stack

Python, PyTorch, torchvision (pretrained WideResNet-50), Streamlit, NumPy, Pillow.

## Reference

Roth, K., Pemula, L., Zepeda, J., Schölkopf, B., Brox, T., & Gehler, P. (2022). *Towards Total Recall in Industrial Anomaly Detection.* CVPR. [arXiv:2106.08265](https://arxiv.org/abs/2106.08265)
