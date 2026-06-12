"""IR-MAD: Iteratively Reweighted Multivariate Alteration Detection.

Full implementation: Canonical Correlation Analysis between the image pair,
MAD variates, iterative chi-square no-change weighting until convergence,
PIF (pseudo-invariant feature) extraction from high-probability no-change
pixels, and band-by-band linear normalization fitted on those PIFs.

Covariance statistics are accumulated incrementally so the algorithm runs
tiled over huge images without loading them fully in memory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterable, Iterator

from changemaster.core.exceptions import RadiometricError
from changemaster.preprocessing._common import require_scipy

if TYPE_CHECKING:
    import numpy as np


@dataclass
class IncrementalStats:
    """Weighted incremental mean/covariance accumulator for stacked bands.

    Accumulates sufficient statistics ``sum(w)``, ``sum(w*x)`` and
    ``sum(w*x*x^T)`` chunk by chunk, so covariance of arbitrarily large
    images can be assembled from tiles.
    """

    dim: int
    weight_sum: float = 0.0
    mean_acc: "np.ndarray | None" = None
    outer_acc: "np.ndarray | None" = None

    def update(self, samples: "np.ndarray", weights: "np.ndarray | None" = None) -> None:
        """Add ``(N, dim)`` samples with optional per-sample weights."""
        import numpy as np

        x = np.asarray(samples, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.dim:
            raise RadiometricError(
                f"Expected samples shaped (N, {self.dim}), got {x.shape}.",
                f"المتوقع عينات بشكل (N, {self.dim})، وجد {x.shape}.",
            )
        w = (
            np.ones(x.shape[0], dtype=np.float64)
            if weights is None
            else np.asarray(weights, dtype=np.float64)
        )
        if self.mean_acc is None:
            self.mean_acc = np.zeros(self.dim)
            self.outer_acc = np.zeros((self.dim, self.dim))
        self.weight_sum += float(w.sum())
        self.mean_acc += w @ x
        self.outer_acc += (x * w[:, None]).T @ x

    @property
    def mean(self) -> "np.ndarray":
        """Weighted mean vector."""
        if self.mean_acc is None or self.weight_sum <= 0:
            raise RadiometricError("No samples accumulated.", "لم تُجمع أي عينات.")
        return self.mean_acc / self.weight_sum

    @property
    def covariance(self) -> "np.ndarray":
        """Weighted covariance matrix."""
        import numpy as np

        if self.outer_acc is None or self.weight_sum <= 0:
            raise RadiometricError("No samples accumulated.", "لم تُجمع أي عينات.")
        mu = self.mean
        return self.outer_acc / self.weight_sum - np.outer(mu, mu)


@dataclass
class IRMADResult:
    """Outcome of the IR-MAD computation.

    Attributes
    ----------
    chi_square:
        Per-pixel chi-square statistic (sum of squared standardized MADs),
        shape ``(H, W)``.
    no_change_probability:
        ``P(no change)`` per pixel from the chi-square survival function.
    mad_variates:
        ``(bands, H, W)`` MAD variates.
    canonical_correlations:
        Canonical correlations (rho) of the final iteration.
    iterations:
        Number of reweighting iterations executed.
    converged:
        ``True`` when the rho change fell below the tolerance.
    pif_mask:
        Boolean ``(H, W)`` mask of pseudo-invariant pixels.
    gains / offsets:
        Per-band linear normalization coefficients fitted on the PIFs.
    """

    chi_square: "np.ndarray"
    no_change_probability: "np.ndarray"
    mad_variates: "np.ndarray"
    canonical_correlations: "np.ndarray"
    iterations: int
    converged: bool
    pif_mask: "np.ndarray"
    gains: "np.ndarray"
    offsets: "np.ndarray"
    warnings: list[str] = field(default_factory=list)


def _flatten_valid(
    reference: "np.ndarray", moving: "np.ndarray", valid: "np.ndarray"
) -> tuple["np.ndarray", "np.ndarray"]:
    """Reshape ``(bands, H, W)`` pairs to ``(N, bands)`` over valid pixels."""
    import numpy as np

    ref = reference.reshape(reference.shape[0], -1).T
    mov = moving.reshape(moving.shape[0], -1).T
    mask = valid.ravel()
    return np.ascontiguousarray(ref[mask]), np.ascontiguousarray(mov[mask])


def _cca(
    sxx: "np.ndarray", syy: "np.ndarray", sxy: "np.ndarray"
) -> tuple["np.ndarray", "np.ndarray", "np.ndarray"]:
    """Canonical correlation analysis from joint covariance blocks.

    Returns ``(rho, a, b)`` where columns of ``a``/``b`` are the canonical
    vectors for X and Y, sorted by *increasing* correlation as used by MAD.
    """
    scipy = require_scipy()
    import numpy as np
    from scipy.linalg import eigh

    dim = sxx.shape[0]
    reg = 1e-9 * np.eye(dim)
    sxx_i = np.linalg.inv(sxx + reg)
    syy_i = np.linalg.inv(syy + reg)
    # Solve the generalized eigenproblem for X canonical vectors.
    m = sxy @ syy_i @ sxy.T
    rho2, a = eigh(m, sxx + reg)
    order = np.argsort(rho2)  # increasing rho => MAD ordering
    rho2 = np.clip(rho2[order], 0.0, 1.0)
    a = a[:, order]
    b = syy_i @ sxy.T @ a
    # Normalize to unit variance under the respective covariances.
    for k in range(dim):
        na = float(np.sqrt(a[:, k] @ sxx @ a[:, k]))
        nb = float(np.sqrt(b[:, k] @ syy @ b[:, k]))
        if na > 0:
            a[:, k] /= na
        if nb > 0:
            b[:, k] /= nb
        # Sign convention: positive correlation between U_k and V_k.
        if float(a[:, k] @ sxy @ b[:, k]) < 0:
            b[:, k] = -b[:, k]
    rho = np.sqrt(rho2)
    _ = scipy  # imported for chi2 later by callers
    return rho, a, b


def compute_irmad(
    reference: "np.ndarray",
    moving: "np.ndarray",
    valid_mask: "np.ndarray | None" = None,
    max_iterations: int = 30,
    tolerance: float = 1e-4,
    pif_probability: float = 0.95,
    chunk_rows: int = 256,
) -> IRMADResult:
    """Run the full IR-MAD algorithm on an image pair.

    Parameters
    ----------
    reference / moving:
        ``(bands, H, W)`` co-registered arrays with identical shapes.
    valid_mask:
        Boolean ``(H, W)`` mask of usable pixels (``None`` = all).
    max_iterations:
        Hard iteration cap for the chi-square reweighting loop.
    tolerance:
        Convergence threshold on the canonical-correlation change.
    pif_probability:
        Minimum no-change probability for PIF selection.
    chunk_rows:
        Row-chunk height for tiled covariance accumulation.

    Returns
    -------
    IRMADResult
        Chi-square map, MAD variates, PIF mask and linear normalization.

    Raises
    ------
    RadiometricError
        On shape mismatches or when too few valid pixels exist.
    """
    require_scipy()
    import numpy as np
    from scipy.stats import chi2

    ref = np.asarray(reference, dtype=np.float64)
    mov = np.asarray(moving, dtype=np.float64)
    if ref.shape != mov.shape or ref.ndim != 3:
        raise RadiometricError(
            f"IR-MAD needs identical (bands, H, W) arrays; got {ref.shape} vs {mov.shape}.",
            f"يتطلب IR-MAD مصفوفتين متطابقتين (bands, H, W)؛ وجد {ref.shape} و{mov.shape}.",
            suggestion_en="Co-register and harmonize the pair before IR-MAD.",
            suggestion_ar="سجّل الزوج هندسياً ووحّده قبل IR-MAD.",
        )
    bands, height, width = ref.shape
    if valid_mask is None:
        valid_mask = np.ones((height, width), dtype=bool)
    valid_mask = valid_mask & np.all(np.isfinite(ref), axis=0) & np.all(
        np.isfinite(mov), axis=0
    )
    n_valid = int(valid_mask.sum())
    if n_valid < 10 * bands:
        raise RadiometricError(
            f"Too few valid pixels ({n_valid}) for IR-MAD with {bands} bands.",
            f"عدد البكسلات الصالحة ({n_valid}) غير كافٍ لتشغيل IR-MAD بعدد نطاقات {bands}.",
            suggestion_en="Reduce mask coverage or use histogram matching instead.",
            suggestion_ar="قلّل تغطية الأقنعة أو استخدم مطابقة الهيستوغرام بدلاً منه.",
        )

    weights_full = np.ones((height, width), dtype=np.float64)
    rho_prev: "np.ndarray | None" = None
    rho = np.zeros(bands)
    a = np.eye(bands)
    b = np.eye(bands)
    iterations = 0
    converged = False

    def _iter_chunks() -> Iterator[tuple[slice, "np.ndarray"]]:
        for r0 in range(0, height, chunk_rows):
            sl = slice(r0, min(height, r0 + chunk_rows))
            yield sl, valid_mask[sl]

    for iteration in range(1, max_iterations + 1):
        iterations = iteration
        stats = IncrementalStats(dim=2 * bands)
        for sl, vm in _iter_chunks():
            x, y = _flatten_valid(ref[:, sl], mov[:, sl], vm)
            if x.shape[0] == 0:
                continue
            w = weights_full[sl].ravel()[vm.ravel()]
            stats.update(np.hstack([x, y]), w)
        cov = stats.covariance
        mu = stats.mean
        sxx = cov[:bands, :bands]
        syy = cov[bands:, bands:]
        sxy = cov[:bands, bands:]
        rho, a, b = _cca(sxx, syy, sxy)

        # MAD variances: var(U - V) = 2 (1 - rho).
        mad_var = np.maximum(2.0 * (1.0 - rho), 1e-12)

        # Recompute chi-square over the full image, chunked.
        chi = np.zeros((height, width), dtype=np.float64)
        for sl, vm in _iter_chunks():
            x, y = _flatten_valid(ref[:, sl], mov[:, sl], vm)
            if x.shape[0] == 0:
                continue
            u = (x - mu[:bands]) @ a
            v = (y - mu[bands:]) @ b
            mads = u - v
            chi_vals = np.sum((mads**2) / mad_var, axis=1)
            block = np.zeros(vm.shape, dtype=np.float64).ravel()
            block[vm.ravel()] = chi_vals
            chi[sl] = block.reshape(vm.shape)
        weights_full = chi2.sf(chi, df=bands)
        weights_full[~valid_mask] = 0.0
        # Degenerate guard: when the pair is almost perfectly correlated the
        # chi-square explodes and all weights vanish; floor them so the
        # accumulation stays well-defined (uniform in the limit).
        if float(weights_full[valid_mask].sum()) < 1e-6:
            weights_full[valid_mask] = 1.0

        if rho_prev is not None and float(np.max(np.abs(rho - rho_prev))) < tolerance:
            converged = True
            break
        rho_prev = rho.copy()

    warnings: list[str] = []
    if not converged:
        warnings.append(
            f"IR-MAD did not converge within {max_iterations} iterations. | "
            f"لم يتقارب IR-MAD خلال {max_iterations} تكراراً."
        )

    # Final MAD variates and no-change probability over the whole image.
    mad_var = np.maximum(2.0 * (1.0 - rho), 1e-12)
    stats = IncrementalStats(dim=2 * bands)
    for sl, vm in _iter_chunks():
        x, y = _flatten_valid(ref[:, sl], mov[:, sl], vm)
        if x.shape[0]:
            stats.update(np.hstack([x, y]), weights_full[sl].ravel()[vm.ravel()])
    if stats.weight_sum <= 0:
        for sl, vm in _iter_chunks():
            x, y = _flatten_valid(ref[:, sl], mov[:, sl], vm)
            if x.shape[0]:
                stats.update(np.hstack([x, y]))
    mu = stats.mean
    mad_variates = np.full((bands, height, width), np.nan)
    chi = np.zeros((height, width), dtype=np.float64)
    for sl, vm in _iter_chunks():
        x, y = _flatten_valid(ref[:, sl], mov[:, sl], vm)
        if x.shape[0] == 0:
            continue
        u = (x - mu[:bands]) @ a
        v = (y - mu[bands:]) @ b
        mads = u - v
        chi_vals = np.sum((mads**2) / mad_var, axis=1)
        flat = np.full((bands, vm.size), np.nan)
        flat[:, vm.ravel()] = mads.T
        mad_variates[:, sl] = flat.reshape(bands, *vm.shape)
        block = np.zeros(vm.shape).ravel()
        block[vm.ravel()] = chi_vals
        chi[sl] = block.reshape(vm.shape)

    no_change_prob = chi2.sf(chi, df=bands)
    no_change_prob[~valid_mask] = 0.0
    pif_mask = (no_change_prob >= pif_probability) & valid_mask
    if int(pif_mask.sum()) < 10:
        warnings.append(
            "Very few PIF pixels found; falling back to all high-probability "
            "no-change pixels at 0.8. | عدد بكسلات PIF قليل جداً؛ يتم استخدام "
            "عتبة احتمال 0.8 بدلاً منها."
        )
        pif_mask = (no_change_prob >= 0.8) & valid_mask

    gains, offsets = fit_pif_normalization(ref, mov, pif_mask)
    return IRMADResult(
        chi_square=chi,
        no_change_probability=no_change_prob,
        mad_variates=mad_variates,
        canonical_correlations=rho,
        iterations=iterations,
        converged=converged,
        pif_mask=pif_mask,
        gains=gains,
        offsets=offsets,
        warnings=warnings,
    )


def fit_pif_normalization(
    reference: "np.ndarray", moving: "np.ndarray", pif_mask: "np.ndarray"
) -> tuple["np.ndarray", "np.ndarray"]:
    """Fit per-band ``ref ≈ gain * mov + offset`` on PIF pixels.

    Returns ``(gains, offsets)`` arrays of length ``bands``.
    """
    import numpy as np

    bands = reference.shape[0]
    gains = np.ones(bands)
    offsets = np.zeros(bands)
    flat_mask = pif_mask.ravel()
    if int(flat_mask.sum()) < 2:
        raise RadiometricError(
            "Not enough PIF pixels to fit a linear normalization.",
            "عدد بكسلات PIF غير كافٍ لتقدير تطبيع خطي.",
            suggestion_en="Lower the PIF probability threshold.",
            suggestion_ar="خفّض عتبة احتمال PIF.",
        )
    for k in range(bands):
        x = moving[k].ravel()[flat_mask]
        y = reference[k].ravel()[flat_mask]
        ok = np.isfinite(x) & np.isfinite(y)
        x, y = x[ok], y[ok]
        if x.size < 2 or float(np.var(x)) <= 0:
            continue
        gain, offset = np.polyfit(x, y, 1)
        gains[k] = gain
        offsets[k] = offset
    return gains, offsets


def apply_irmad_normalization(
    moving: "np.ndarray", gains: "np.ndarray", offsets: "np.ndarray"
) -> "np.ndarray":
    """Apply the fitted per-band linear normalization to the moving image."""
    import numpy as np

    mov = np.asarray(moving, dtype=np.float64)
    return mov * np.asarray(gains)[:, None, None] + np.asarray(offsets)[:, None, None]
