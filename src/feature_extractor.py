"""
feature_extractor.py

Extracts patch-level feature embeddings from images using a pretrained
ImageNet backbone (WideResNet-50), following the approach used in PatchCore
(Roth et al., 2022, "Towards Total Recall in Industrial Anomaly Detection").

Core idea: mid-level CNN feature maps (not the final classification layer)
capture local texture/structure patterns that generalize well to detecting
*unseen* anomaly types, without ever training on defect examples. Each
spatial location in the feature map corresponds to a "patch" of the original
image, giving us a grid of patch embeddings we can compare against a memory
bank of "normal" patches.

We concatenate features from two mid-network layers (layer2 and layer3),
matching the original paper's finding that this captures both fine detail
and broader context better than any single layer alone.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Standard ImageNet preprocessing, since the backbone was trained on ImageNet
PREPROCESS = T.Compose(
    [
        T.Resize((256, 256)),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class PatchFeatureExtractor:
    """Wraps a pretrained WideResNet-50 and exposes mid-layer patch embeddings."""

    def __init__(self, pretrained: bool = True):
        weights = models.Wide_ResNet50_2_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = models.wide_resnet50_2(weights=weights)
        backbone.eval()
        self.backbone = backbone.to(DEVICE)

        self._features: dict[str, torch.Tensor] = {}
        backbone.layer2.register_forward_hook(self._make_hook("layer2"))
        backbone.layer3.register_forward_hook(self._make_hook("layer3"))

    def _make_hook(self, name: str):
        def hook(_module, _input, output):
            self._features[name] = output.detach()

        return hook

    @torch.no_grad()
    def extract(self, image: Image.Image) -> torch.Tensor:
        """
        Returns a (H*W, C) tensor of patch embeddings for one image, where
        H, W is the spatial grid size of layer2's feature map (the coarser
        layer3 map is upsampled to match, then concatenated channel-wise).
        """
        x = PREPROCESS(image.convert("RGB")).unsqueeze(0).to(DEVICE)
        self.backbone(x)

        f2 = self._features["layer2"]  # (1, C2, H2, W2)
        f3 = self._features["layer3"]  # (1, C3, H3, W3), H3 = H2/2

        f3_upsampled = F.interpolate(f3, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        combined = torch.cat([f2, f3_upsampled], dim=1)  # (1, C2+C3, H2, W2)

        # Local average pooling (3x3) smooths patch embeddings, matching the
        # paper's "locally aware patch features" — helps robustness to small
        # misalignments between reference and query images.
        combined = F.avg_pool2d(combined, kernel_size=3, stride=1, padding=1)

        _, c, h, w = combined.shape
        patches = combined.permute(0, 2, 3, 1).reshape(h * w, c)  # (H*W, C)
        return patches.cpu(), (h, w)


if __name__ == "__main__":
    from PIL import Image
    import numpy as np

    # Smoke test with a synthetic image (no network needed for the image itself)
    img = Image.fromarray((np.random.rand(224, 224, 3) * 255).astype("uint8"))
    extractor = PatchFeatureExtractor(pretrained=False)  # random weights: shape test only
    patches, grid = extractor.extract(img)
    print(f"Patch embeddings: {patches.shape}, grid size: {grid}")
