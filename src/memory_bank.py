"""
memory_bank.py

Builds a "memory bank" of patch embeddings from reference (known-normal)
images. At inference time, a query image is flagged as anomalous wherever
its patches are far (in embedding space) from anything in this bank.

We apply greedy coreset subsampling (a simplified version of the k-center
greedy approach in the PatchCore paper) so the bank stays a manageable size
even with several reference images — this keeps nearest-neighbor lookups
fast without materially hurting detection quality, since nearby patches in
embedding space are redundant anyway.
"""

from __future__ import annotations

import torch


class MemoryBank:
    def __init__(self, max_size: int = 5000, seed: int = 0):
        self.max_size = max_size
        self.seed = seed
        self.bank: torch.Tensor | None = None

    def build(self, patch_embeddings: list[torch.Tensor]) -> None:
        """patch_embeddings: list of (N_i, C) tensors, one per reference image."""
        all_patches = torch.cat(patch_embeddings, dim=0)  # (N_total, C)

        if all_patches.shape[0] <= self.max_size:
            self.bank = all_patches
            return

        self.bank = self._greedy_coreset(all_patches, self.max_size)

    def _greedy_coreset(self, patches: torch.Tensor, target_size: int) -> torch.Tensor:
        """
        Greedy k-center subsampling: iteratively picks the patch furthest
        from the current selected set, ensuring good coverage of the
        embedding space rather than a random (potentially redundant) subset.

        Runs in a normalized embedding space for numerical stability.
        """
        torch.manual_seed(self.seed)
        n = patches.shape[0]
        normed = torch.nn.functional.normalize(patches, dim=1)

        selected_idx = [torch.randint(0, n, (1,)).item()]
        min_dists = torch.cdist(normed, normed[selected_idx]).squeeze(1)

        for _ in range(target_size - 1):
            next_idx = torch.argmax(min_dists).item()
            selected_idx.append(next_idx)
            new_dists = torch.cdist(normed, normed[[next_idx]]).squeeze(1)
            min_dists = torch.minimum(min_dists, new_dists)

        return patches[selected_idx]

    def anomaly_scores(self, query_patches: torch.Tensor) -> torch.Tensor:
        """
        For each query patch, returns its L2 distance to the nearest patch
        in the memory bank. Higher distance = more anomalous (i.e. unlike
        anything seen in the "normal" reference images).
        """
        if self.bank is None:
            raise RuntimeError("Memory bank is empty — call build() first.")

        dists = torch.cdist(query_patches, self.bank)  # (N_query, N_bank)
        min_dists, _ = dists.min(dim=1)
        return min_dists


if __name__ == "__main__":
    import torch

    torch.manual_seed(0)
    ref_patches = [torch.randn(784, 64) for _ in range(3)]

    bank = MemoryBank(max_size=1000)
    bank.build(ref_patches)
    print(f"Bank size: {bank.bank.shape}")

    query = torch.randn(784, 64)
    scores = bank.anomaly_scores(query)
    print(f"Query scores: min={scores.min():.3f}, max={scores.max():.3f}, mean={scores.mean():.3f}")
