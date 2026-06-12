"""PCA-KMeans change detection (Celik 2009 standard formulation).

Pipeline: difference image → non-overlapping ``h x h`` blocks → eigenvector
basis from the block covariance matrix → projection of every pixel's
``h x h`` neighbourhood onto that basis → 2-class KMeans → the cluster whose
centre is farther from the no-change origin is labelled change.

KMeans (implemented here in pure NumPy with k-means++ initialisation) is fit
on a stratified random sample on huge images, then the two fitted centres
are applied to the full image chunk by chunk so memory stays bounded. Chunk
sizes adapt to the :class:`~changemaster.core.hardware.HardwareTier` —
smaller windows on weak machines, never lower accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from changemaster.core.exceptions import EngineError
from changemaster.core.hardware import HardwareInfo
from changemaster.preprocessing._common import adaptive_tile_size

if TYPE_CHECKING:
    import numpy as np


@dataclass
class PCAKMeansResult:
    """Outcome of PCA-KMeans change detection.

    Attributes
    ----------
    probability:
        Float32 ``(H, W)`` soft change probability in ``[0, 1]``
        (``NaN`` on invalid pixels).
    change_mask:
        Boolean ``(H, W)`` hard cluster assignment (change cluster).
    eigenvectors:
        ``(h*h, n_components)`` PCA basis from the block covariance.
    centers:
        ``(2, n_components)`` KMeans centres ``[no_change, change]``.
    warnings:
        Bilingual warnings.
    """

    probability: "np.ndarray"
    change_mask: "np.ndarray"
    eigenvectors: "np.ndarray"
    centers: "np.ndarray"
    warnings: list[str] = field(default_factory=list)


def _kmeans2(
    samples: "np.ndarray",
    max_iterations: int = 100,
    tolerance: float = 1e-6,
    seed: int = 42,
) -> "np.ndarray":
    """Pure-NumPy 2-class KMeans with k-means++ initialisation.

    Parameters
    ----------
    samples:
        ``(N, d)`` feature matrix.
    max_iterations:
        Iteration cap for Lloyd's algorithm.
    tolerance:
        Convergence threshold on total centre movement.
    seed:
        Random seed for reproducible initialisation.

    Returns
    -------
    numpy.ndarray
        ``(2, d)`` cluster centres.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    n = samples.shape[0]
    # k-means++ init: first centre uniform, second proportional to D^2.
    first = samples[rng.integers(n)]
    d2 = ((samples - first) ** 2).sum(axis=1)
    total = float(d2.sum())
    if total <= 0:
        return np.stack([first, first])
    second = samples[rng.choice(n, p=d2 / total)]
    centers = np.stack([first, second]).astype(np.float64)

    for _ in range(max_iterations):
        d0 = ((samples - centers[0]) ** 2).sum(axis=1)
        d1 = ((samples - centers[1]) ** 2).sum(axis=1)
        assign = d1 < d0
        new_centers = centers.copy()
        if (~assign).any():
            new_centers[0] = samples[~assign].mean(axis=0)
        if assign.any():
            new_centers[1] = samples[assign].mean(axis=0)
        shift = float(np.abs(new_centers - centers).sum())
        centers = new_centers
        if shift < tolerance:
            break
    return centers


def pca_kmeans_change(
    reference: "np.ndarray",
    moving: "np.ndarray",
    valid_mask: "np.ndarray",
    block_size: int = 4,
    n_components: int | None = None,
    max_samples: int = 200_000,
    hardware: HardwareInfo | None = None,
    seed: int = 42,
) -> PCAKMeansResult:
    """Run the full PCA-KMeans change-detection algorithm.

    Parameters
    ----------
    reference / moving:
        ``(bands, H, W)`` co-registered, normalized arrays.
    valid_mask:
        Boolean ``(H, W)`` mask; invalid pixels are excluded from the block
        covariance, the KMeans fit and the output (``NaN`` / ``False``).
    block_size:
        Edge length ``h`` of the non-overlapping blocks (default 4).
    n_components:
        Number of eigenvectors kept; default ``min(h*h, 3)``.
    max_samples:
        Maximum pixels used to fit KMeans; a stratified random sample
        (uniform over row strata) is drawn on bigger images and the fitted
        centres are then applied to all pixels chunk by chunk.
    hardware:
        Hardware snapshot for adaptive chunk sizing.
    seed:
        Random seed for sampling and KMeans initialisation.

    Returns
    -------
    PCAKMeansResult
        Soft probability, hard mask, PCA basis and KMeans centres.
    """
    import numpy as np

    ref = np.asarray(reference, dtype=np.float64)
    mov = np.asarray(moving, dtype=np.float64)
    if ref.shape != mov.shape or ref.ndim != 3:
        raise EngineError(
            f"PCA-KMeans needs identical (bands, H, W) arrays; got "
            f"{ref.shape} vs {mov.shape}.",
            f"يتطلب PCA-KMeans مصفوفتين متطابقتين (bands, H, W)؛ وجد "
            f"{ref.shape} و{mov.shape}.",
        )
    if block_size < 2:
        raise EngineError(
            f"Block size must be >= 2, got {block_size}.",
            f"يجب أن يكون حجم الكتلة 2 على الأقل، وجد {block_size}.",
        )
    bands, height, width = ref.shape
    h = block_size
    if height < h or width < h:
        raise EngineError(
            f"Image {height}x{width} is smaller than one {h}x{h} block.",
            f"الصورة {height}x{width} أصغر من كتلة واحدة {h}x{h}.",
            suggestion_en="Use a smaller block size.",
            suggestion_ar="استخدم حجم كتلة أصغر.",
        )
    warnings: list[str] = []
    rng = np.random.default_rng(seed)

    # Difference image: multi-band Euclidean magnitude (single plane).
    # Invalid pixels are replaced by the valid mean so they never bias the
    # covariance, and are re-masked at the end.
    diff = np.sqrt(((mov - ref) ** 2).sum(axis=0))
    valid_values = diff[valid_mask]
    if valid_values.size < h * h:
        raise EngineError(
            "Too few valid pixels for PCA-KMeans.",
            "عدد البكسلات الصالحة غير كافٍ لتشغيل PCA-KMeans.",
            suggestion_en="Check the validity mask coverage.",
            suggestion_ar="تحقق من تغطية قناع الصلاحية.",
        )
    fill_value = float(valid_values.mean())
    diff_filled = np.where(valid_mask, diff, fill_value)

    # --- PCA basis from non-overlapping h x h blocks --------------------------
    n_block_rows = height // h
    n_block_cols = width // h
    dim = h * h
    stats_sum = np.zeros(dim)
    stats_sq = np.zeros((dim, dim))
    n_blocks = 0
    chunk_rows = max(h, (adaptive_tile_size(hardware) // h) * h)
    for r0 in range(0, n_block_rows * h, chunk_rows):
        r1 = min(n_block_rows * h, r0 + chunk_rows)
        rows = diff_filled[r0:r1, : n_block_cols * h]
        vrows = valid_mask[r0:r1, : n_block_cols * h]
        nbr = (r1 - r0) // h
        blocks = rows.reshape(nbr, h, n_block_cols, h).transpose(0, 2, 1, 3)
        blocks = blocks.reshape(-1, dim)
        vblocks = vrows.reshape(nbr, h, n_block_cols, h).transpose(0, 2, 1, 3)
        keep = vblocks.reshape(-1, dim).all(axis=1)
        blocks = blocks[keep]
        if blocks.shape[0] == 0:
            continue
        stats_sum += blocks.sum(axis=0)
        stats_sq += blocks.T @ blocks
        n_blocks += blocks.shape[0]
    if n_blocks < dim:
        warnings.append(
            "Few fully-valid blocks; PCA basis estimated from partially "
            "filled data. | كتل صالحة بالكامل قليلة؛ قُدّر أساس PCA من بيانات معبأة جزئياً."
        )
        blocks = diff_filled[: n_block_rows * h, : n_block_cols * h]
        blocks = blocks.reshape(n_block_rows, h, n_block_cols, h)
        blocks = blocks.transpose(0, 2, 1, 3).reshape(-1, dim)
        stats_sum = blocks.sum(axis=0)
        stats_sq = blocks.T @ blocks
        n_blocks = blocks.shape[0]
    mean_vec = stats_sum / n_blocks
    cov = stats_sq / n_blocks - np.outer(mean_vec, mean_vec)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    n_comp = n_components if n_components is not None else min(dim, 3)
    n_comp = max(1, min(n_comp, dim))
    basis = eigvecs[:, order[:n_comp]]  # (dim, n_comp)

    # --- Project every pixel's h x h neighbourhood onto the basis -------------
    pad = h // 2
    padded = np.pad(diff_filled, pad, mode="reflect")

    def _project_rows(r0: int, r1: int) -> "np.ndarray":
        """Project pixels in rows [r0, r1) onto the PCA basis."""
        out = np.empty((r1 - r0, width, n_comp))
        window = padded[r0 : r1 + 2 * pad, :]
        strided = np.lib.stride_tricks.sliding_window_view(window, (h, h))
        patch = strided[: r1 - r0, :width].reshape(r1 - r0, width, dim)
        out[:] = (patch - mean_vec) @ basis
        return out

    # --- Stratified sample for KMeans fitting ----------------------------------
    n_valid = int(valid_mask.sum())
    features_sample: list["np.ndarray"] = []
    sample_fraction = min(1.0, max_samples / max(1, n_valid))
    proc_rows = max(h, adaptive_tile_size(hardware) // 4)
    for r0 in range(0, height, proc_rows):
        r1 = min(height, r0 + proc_rows)
        feats = _project_rows(r0, r1)
        vm = valid_mask[r0:r1]
        chunk_feats = feats[vm]
        if chunk_feats.shape[0] == 0:
            continue
        if sample_fraction < 1.0:
            take = max(1, int(round(chunk_feats.shape[0] * sample_fraction)))
            idx = rng.choice(chunk_feats.shape[0], size=take, replace=False)
            chunk_feats = chunk_feats[idx]
        features_sample.append(chunk_feats)
    samples = np.concatenate(features_sample, axis=0)
    centers = _kmeans2(samples, seed=seed)

    # The change cluster is the one farther from the projected no-change
    # origin (the mean block maps to the feature-space origin).
    norms = np.sqrt((centers**2).sum(axis=1))
    change_idx = int(np.argmax(norms))
    ordered = np.stack([centers[1 - change_idx], centers[change_idx]])

    # --- Apply centres to all pixels, chunked ----------------------------------
    probability = np.full((height, width), np.nan, dtype=np.float32)
    change_mask = np.zeros((height, width), dtype=bool)
    for r0 in range(0, height, proc_rows):
        r1 = min(height, r0 + proc_rows)
        feats = _project_rows(r0, r1)
        d_nc = np.sqrt(((feats - ordered[0]) ** 2).sum(axis=2))
        d_ch = np.sqrt(((feats - ordered[1]) ** 2).sum(axis=2))
        prob = d_nc / (d_nc + d_ch + 1e-12)
        vm = valid_mask[r0:r1]
        probability[r0:r1] = np.where(vm, prob, np.nan).astype(np.float32)
        change_mask[r0:r1] = vm & (d_ch < d_nc)

    return PCAKMeansResult(
        probability=probability,
        change_mask=change_mask,
        eigenvectors=basis,
        centers=ordered,
        warnings=warnings,
    )
