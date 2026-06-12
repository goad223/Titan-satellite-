"""PreprocessingPipeline: orchestration, checkpoints and the final report.

Optical order: quality -> harmonize -> coregistration -> masking -> radiometric.
SAR order:     quality -> calibration -> speckle -> harmonize -> coregistration.

Every step writes a checkpoint (intermediate GeoTIFF when rasterio is
available, otherwise ``.npz``, plus a JSON state file) so a run can resume
from any completed step after an interruption. A final
:class:`PreprocessingReport` collects all metrics and is saved as JSON.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from changemaster.core.exceptions import PipelineError
from changemaster.core.hardware import HardwareInfo, detect_hardware
from changemaster.core.logging_setup import get_logger
from changemaster.io_engine.base_reader import open_image
from changemaster.io_engine.metadata import GeoReference, ImageMetadata
from changemaster.preprocessing.coregistration.evaluator import evaluate_registration
from changemaster.preprocessing.coregistration.pyramid import register_pyramid
from changemaster.preprocessing.coregistration.feature_based import pick_matching_band
from changemaster.preprocessing.harmonize import HarmonizedPair, harmonize_arrays
from changemaster.preprocessing.masking.cloud import detect_clouds
from changemaster.preprocessing.masking.combiner import ValidityMask, combine_masks
from changemaster.preprocessing.masking.nodata import detect_nodata, detect_saturation
from changemaster.preprocessing.quality import RECOMMEND_REJECT, assess_quality
from changemaster.preprocessing.radiometric.selector import normalize_pair
from changemaster.preprocessing.sar.convert import percentile_clip, to_db
from changemaster.preprocessing.sar.speckle import apply_speckle_filter

if TYPE_CHECKING:
    import numpy as np

logger = get_logger(__name__)

#: Optical pipeline step order.
OPTICAL_STEPS = ("quality", "harmonize", "coregistration", "masking", "radiometric")
#: SAR pipeline step order.
SAR_STEPS = ("quality", "calibration", "speckle", "harmonize", "coregistration")


@dataclass
class StepRecord:
    """Execution record of one pipeline step.

    Attributes
    ----------
    name:
        Step identifier.
    status:
        ``"completed"``, ``"skipped"`` or ``"resumed"``.
    duration_s:
        Wall-clock duration in seconds.
    metrics:
        Step-specific metric dictionary.
    warnings:
        Bilingual warnings raised inside the step.
    """

    name: str
    status: str
    duration_s: float = 0.0
    metrics: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


@dataclass
class PreprocessingReport:
    """Final report of a full preprocessing run.

    Attributes
    ----------
    mode:
        ``"optical"`` or ``"sar"``.
    steps:
        Ordered :class:`StepRecord` list.
    rmse_px:
        Final registration RMSE on independent check points (when known).
    mask_fractions:
        Coverage fraction per mask class label.
    total_duration_s:
        Total wall-clock duration.
    warnings:
        All warnings aggregated across steps.
    """

    mode: str
    steps: list[StepRecord] = field(default_factory=list)
    rmse_px: float | None = None
    mask_fractions: dict[str, float] = field(default_factory=dict)
    total_duration_s: float = 0.0
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary representation."""
        return asdict(self)

    def save_json(self, path: Path | str) -> Path:
        """Persist the report as a JSON file and return its path."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return out

    def summary(self) -> str:
        """Human-readable bilingual run summary."""
        lines = [
            "Preprocessing report | تقرير المعالجة المسبقة",
            "=" * 50,
            f"Mode: {self.mode}",
            f"Total duration: {self.total_duration_s:.1f}s",
        ]
        for step in self.steps:
            lines.append(f"  [{step.status:>9}] {step.name:<15} {step.duration_s:.1f}s")
        if self.rmse_px is not None:
            lines.append(f"Registration RMSE: {self.rmse_px:.3f} px")
        for label, fraction in self.mask_fractions.items():
            if fraction > 0:
                lines.append(f"  mask {label}: {fraction:.1%}")
        if self.warnings:
            lines.append(f"Warnings ({len(self.warnings)}):")
            lines.extend(f"  - {w}" for w in self.warnings)
        return "\n".join(lines)


class PreprocessingPipeline:
    """Checkpointed preprocessing pipeline for an image pair.

    Parameters
    ----------
    workdir:
        Directory for checkpoints, state JSON and the final report.
    mode:
        ``"optical"`` (default) or ``"sar"``.
    hardware:
        Hardware snapshot; detected automatically when omitted.
    speckle_method:
        SAR speckle filter name (``"refined_lee"`` default).
    target_rmse_px:
        Registration acceptance threshold.
    """

    def __init__(
        self,
        workdir: Path | str,
        mode: str = "optical",
        hardware: HardwareInfo | None = None,
        speckle_method: str = "refined_lee",
        target_rmse_px: float = 1.0,
    ) -> None:
        if mode not in ("optical", "sar"):
            raise PipelineError(
                f"Unknown pipeline mode '{mode}'.",
                f"وضع أنبوب معالجة غير معروف '{mode}'.",
                suggestion_en="Use 'optical' or 'sar'.",
                suggestion_ar="استخدم 'optical' أو 'sar'.",
            )
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.mode = mode
        self.hardware = hardware if hardware is not None else detect_hardware()
        self.speckle_method = speckle_method
        self.target_rmse_px = target_rmse_px
        self.steps: tuple[str, ...] = OPTICAL_STEPS if mode == "optical" else SAR_STEPS

    # -- checkpointing ---------------------------------------------------------

    def _state_path(self) -> Path:
        return self.workdir / "pipeline_state.json"

    def _checkpoint_path(self, step: str) -> Path:
        return self.workdir / f"checkpoint_{step}.npz"

    def _load_state(self) -> dict[str, Any]:
        path = self._state_path()
        if not path.exists():
            return {"completed": [], "mode": self.mode}
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise PipelineError(
                f"Pipeline state file is corrupt: {exc}",
                f"ملف حالة الأنبوب تالف: {exc}",
                suggestion_en="Delete pipeline_state.json to restart from scratch.",
                suggestion_ar="احذف pipeline_state.json لإعادة التشغيل من البداية.",
            ) from exc
        if state.get("mode") != self.mode:
            raise PipelineError(
                f"Existing state is for mode '{state.get('mode')}', not '{self.mode}'.",
                f"الحالة الموجودة لوضع '{state.get('mode')}' وليس '{self.mode}'.",
                suggestion_en="Use a fresh working directory for a different mode.",
                suggestion_ar="استخدم مجلد عمل جديداً لوضع مختلف.",
            )
        return state

    def _save_state(self, state: dict[str, Any]) -> None:
        self._state_path().write_text(
            json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def _write_checkpoint(
        self,
        step: str,
        arrays: dict[str, "np.ndarray"],
        georef: GeoReference | None = None,
    ) -> None:
        """Write the step checkpoint: ``.npz`` arrays plus optional GeoTIFF."""
        import numpy as np

        np.savez_compressed(self._checkpoint_path(step), **arrays)
        if georef is not None and georef.is_georeferenced:
            try:
                from changemaster.io_engine.writer import write_geotiff

                for key, arr in arrays.items():
                    if arr.ndim in (2, 3) and np.issubdtype(arr.dtype, np.number):
                        write_geotiff(
                            self.workdir / f"checkpoint_{step}_{key}.tif",
                            np.asarray(arr, dtype=np.float32),
                            georef=georef,
                        )
            except Exception:  # noqa: BLE001 - GeoTIFF checkpoint is best-effort
                logger.warning("GeoTIFF checkpoint for step '%s' skipped.", step)

    def _read_checkpoint(self, step: str) -> dict[str, "np.ndarray"]:
        import numpy as np

        path = self._checkpoint_path(step)
        if not path.exists():
            raise PipelineError(
                f"Checkpoint for step '{step}' is missing; cannot resume.",
                f"نقطة الاستئناف للخطوة '{step}' مفقودة؛ تعذر الاستئناف.",
                suggestion_en="Re-run the pipeline without resume.",
                suggestion_ar="أعد تشغيل الأنبوب دون استئناف.",
            )
        with np.load(path, allow_pickle=False) as data:
            return {k: data[k] for k in data.files}

    # -- execution -------------------------------------------------------------

    def run(
        self,
        reference_path: Path | str,
        moving_path: Path | str,
        resume: bool = False,
        sun_elevation_deg: float | None = None,
        sun_azimuth_deg: float | None = None,
        band_roles: dict[str, int] | None = None,
        looks: float = 1.0,
        reflectance_scale: float = 1.0,
    ) -> PreprocessingReport:
        """Run (or resume) the full preprocessing pipeline on a pair.

        Parameters
        ----------
        reference_path / moving_path:
            Input rasters (any supported format).
        resume:
            Continue from the last completed checkpoint when ``True``.
        sun_elevation_deg / sun_azimuth_deg:
            Solar geometry for shadow projection (optical mode).
        band_roles:
            Mapping role -> 1-based band index (e.g. ``{"blue": 1,
            "green": 2, "red": 3, "nir": 4}``) for spectral masking.
        looks:
            Equivalent number of looks for SAR speckle filtering.
        reflectance_scale:
            DN-to-reflectance divisor for spectral masking (e.g. 10000 for
            Sentinel-2 L2A).

        Returns
        -------
        PreprocessingReport
            Full metrics report (also saved to ``report.json``).
        """
        import numpy as np

        start_total = time.monotonic()
        report = PreprocessingReport(mode=self.mode)
        state = self._load_state() if resume else {"completed": [], "mode": self.mode}
        completed: list[str] = list(state.get("completed", []))

        with open_image(reference_path) as ref_reader, open_image(moving_path) as mov_reader:
            ref_meta = ref_reader.metadata
            mov_meta = mov_reader.metadata

            # ---- quality ----------------------------------------------------
            record = self._begin("quality", completed, report)
            if record is not None:
                for label, reader in (("reference", ref_reader), ("moving", mov_reader)):
                    q = assess_quality(reader, hardware=self.hardware)
                    record.metrics[label] = q.to_dict()
                    record.warnings.extend(q.warnings)
                    if q.recommendation == RECOMMEND_REJECT:
                        raise PipelineError(
                            f"Quality gate rejected the {label} image "
                            f"(score {q.score:.0f}/100).",
                            f"رفضت بوابة الجودة صورة {label} (الدرجة {q.score:.0f}/100).",
                            suggestion_en="Inspect the quality warnings and supply a better scene.",
                            suggestion_ar="راجع تحذيرات الجودة ووفر مشهداً أفضل.",
                        )
                self._finish("quality", record, {}, None, completed, state, report)

            ref_data = np.asarray(ref_reader.read(), dtype=np.float64)
            mov_data = np.asarray(mov_reader.read(), dtype=np.float64)

            if self.mode == "sar":
                ref_data, mov_data = self._run_sar_steps(
                    ref_data, mov_data, completed, state, report, looks
                )

            # ---- harmonize --------------------------------------------------
            record = self._begin("harmonize", completed, report)
            if record is not None:
                pair = harmonize_arrays(ref_data, mov_data, ref_meta, mov_meta)
                record.warnings.extend(pair.warnings)
                record.metrics["harmonized"] = pair.to_dict()
                self._finish(
                    "harmonize",
                    record,
                    {"reference": pair.reference, "moving": pair.moving},
                    pair.georef,
                    completed,
                    state,
                    report,
                )
                georef = pair.georef
            else:
                arrays = self._read_checkpoint("harmonize")
                pair = HarmonizedPair(
                    reference=arrays["reference"],
                    moving=arrays["moving"],
                    georef=ref_meta.georef,
                )
                georef = pair.georef

            # ---- coregistration ---------------------------------------------
            record = self._begin("coregistration", completed, report)
            if record is not None:
                aligned, rmse, reg_warnings = self._coregister(pair, ref_meta)
                record.warnings.extend(reg_warnings)
                record.metrics["rmse_px"] = rmse
                report.rmse_px = rmse
                pair.moving = aligned
                self._finish(
                    "coregistration",
                    record,
                    {"reference": pair.reference, "moving": pair.moving},
                    georef,
                    completed,
                    state,
                    report,
                )
            else:
                arrays = self._read_checkpoint("coregistration")
                pair.reference = arrays["reference"]
                pair.moving = arrays["moving"]
                report.rmse_px = state.get("rmse_px")

            if self.mode == "optical":
                # ---- masking ------------------------------------------------
                record = self._begin("masking", completed, report)
                if record is not None:
                    validity = self._build_masks(
                        pair,
                        ref_meta,
                        band_roles,
                        sun_elevation_deg,
                        sun_azimuth_deg,
                        reflectance_scale,
                    )
                    record.warnings.extend(validity.warnings)
                    record.metrics["fractions"] = validity.fractions
                    record.metrics["invalid_fraction"] = validity.invalid_fraction
                    report.mask_fractions = validity.fractions
                    self._finish(
                        "masking",
                        record,
                        {"validity_mask": validity.mask},
                        georef,
                        completed,
                        state,
                        report,
                    )
                else:
                    arrays = self._read_checkpoint("masking")
                    validity = combine_masks(
                        arrays["validity_mask"].shape,
                        nodata=arrays["validity_mask"] == 5,
                    )
                    validity.mask = arrays["validity_mask"]

                # ---- radiometric ---------------------------------------------
                record = self._begin("radiometric", completed, report)
                if record is not None:
                    valid = validity.mask == 0
                    selection = normalize_pair(
                        pair.reference,
                        pair.moving,
                        valid_mask=valid,
                        reference_sensor=ref_meta.sensor_id,
                        moving_sensor=mov_meta.sensor_id,
                        reference_datetime=ref_meta.acquisition_datetime,
                        moving_datetime=mov_meta.acquisition_datetime,
                    )
                    record.warnings.extend(selection.warnings)
                    record.metrics["method"] = selection.method
                    record.metrics["reason"] = selection.reason
                    record.metrics.update(selection.details)
                    pair.moving = selection.normalized
                    self._finish(
                        "radiometric",
                        record,
                        {"reference": pair.reference, "moving": pair.moving},
                        georef,
                        completed,
                        state,
                        report,
                    )

        report.total_duration_s = time.monotonic() - start_total
        report.warnings = [w for step in report.steps for w in step.warnings]
        state["rmse_px"] = report.rmse_px
        self._save_state(state)
        report.save_json(self.workdir / "report.json")
        logger.info("\n%s", report.summary())
        return report

    # -- step helpers ------------------------------------------------------------

    def _begin(
        self, step: str, completed: list[str], report: PreprocessingReport
    ) -> StepRecord | None:
        """Start a step; returns ``None`` when it is resumed from checkpoint."""
        if step in completed:
            report.steps.append(StepRecord(name=step, status="resumed"))
            logger.info("Step '%s' resumed from checkpoint.", step)
            return None
        record = StepRecord(name=step, status="completed")
        record.metrics["_start"] = time.monotonic()
        return record

    def _finish(
        self,
        step: str,
        record: StepRecord,
        arrays: dict[str, "np.ndarray"],
        georef: GeoReference | None,
        completed: list[str],
        state: dict[str, Any],
        report: PreprocessingReport,
    ) -> None:
        """Close a step: write its checkpoint, update state and the report."""
        start = record.metrics.pop("_start", time.monotonic())
        record.duration_s = time.monotonic() - start
        if arrays:
            self._write_checkpoint(step, arrays, georef)
        completed.append(step)
        state["completed"] = completed
        self._save_state(state)
        report.steps.append(record)
        logger.info("Step '%s' completed in %.1fs.", step, record.duration_s)

    def _run_sar_steps(
        self,
        ref_data: "np.ndarray",
        mov_data: "np.ndarray",
        completed: list[str],
        state: dict[str, Any],
        report: PreprocessingReport,
        looks: float,
    ) -> tuple["np.ndarray", "np.ndarray"]:
        """Run SAR calibration (dB conversion + clip) and speckle filtering."""
        import numpy as np

        record = self._begin("calibration", completed, report)
        if record is not None:
            # Inputs are assumed sigma0/amplitude; convert to clipped dB so
            # both dates share one radiometric scale. (Full LUT calibration
            # is available via sar.calibration for SAFE annotation files.)
            ref_data = np.stack([percentile_clip(to_db(b)) for b in ref_data])
            mov_data = np.stack([percentile_clip(to_db(b)) for b in mov_data])
            record.metrics["unit"] = "dB"
            self._finish(
                "calibration",
                record,
                {"reference": ref_data, "moving": mov_data},
                None,
                completed,
                state,
                report,
            )
        else:
            arrays = self._read_checkpoint("calibration")
            ref_data, mov_data = arrays["reference"], arrays["moving"]

        record = self._begin("speckle", completed, report)
        if record is not None:
            ref_data = np.stack(
                [
                    apply_speckle_filter(b, self.speckle_method, looks=looks)
                    for b in ref_data
                ]
            )
            mov_data = np.stack(
                [
                    apply_speckle_filter(b, self.speckle_method, looks=looks)
                    for b in mov_data
                ]
            )
            record.metrics["filter"] = self.speckle_method
            self._finish(
                "speckle",
                record,
                {"reference": ref_data, "moving": mov_data},
                None,
                completed,
                state,
                report,
            )
        else:
            arrays = self._read_checkpoint("speckle")
            ref_data, mov_data = arrays["reference"], arrays["moving"]
        return ref_data, mov_data

    def _coregister(
        self, pair: HarmonizedPair, ref_meta: ImageMetadata
    ) -> tuple["np.ndarray", float | None, list[str]]:
        """Pyramid-register the pair; returns (aligned moving, rmse, warnings)."""
        from changemaster.core.exceptions import CoregistrationError

        band = pick_matching_band(pair.band_names, pair.reference.shape[0])
        ref_band = pair.reference[band - 1]
        mov_band = pair.moving[band - 1]
        warnings: list[str] = []
        try:
            result = register_pyramid(
                ref_band, mov_band, hardware=self.hardware,
                target_rmse_px=self.target_rmse_px,
            )
        except CoregistrationError as exc:
            warnings.append(
                f"Co-registration failed; the pair is used unregistered: "
                f"{exc.message_en} | فشل التسجيل الهندسي؛ يُستخدم الزوج دون "
                f"تسجيل: {exc.message_ar}"
            )
            return pair.moving, None, warnings
        warnings.extend(result.warnings)
        rmse: float | None = None
        if result.matches.src_points.shape[0] >= 8:
            _, evaluation = evaluate_registration(
                result.matches.src_points,
                result.matches.dst_points,
                target_rmse_px=self.target_rmse_px,
            )
            rmse = evaluation.rmse_px
            warnings.extend(evaluation.warnings)
        aligned = result.transform.warp_image(
            pair.moving, (pair.reference.shape[1], pair.reference.shape[2])
        )
        _ = ref_meta
        return aligned, rmse, warnings

    def _build_masks(
        self,
        pair: HarmonizedPair,
        ref_meta: ImageMetadata,
        band_roles: dict[str, int] | None,
        sun_elevation_deg: float | None,
        sun_azimuth_deg: float | None,
        reflectance_scale: float = 1.0,
    ) -> ValidityMask:
        """Build the unified validity mask for the harmonized pair."""
        from changemaster.core.exceptions import MaskingError
        from changemaster.preprocessing.masking.shadow import detect_shadows
        from changemaster.preprocessing.masking.snow_water import (
            detect_snow,
            detect_water,
        )

        shape = (pair.reference.shape[1], pair.reference.shape[2])
        nodata_mask = detect_nodata(pair.reference, pair.nodata) | detect_nodata(
            pair.moving, pair.nodata
        )
        nodata_mask |= detect_saturation(pair.reference, ref_meta.dtype)

        cloud = shadow = snow = water = None
        if band_roles:
            roles = {
                role: pair.reference[idx - 1]
                for role, idx in band_roles.items()
                if 1 <= idx <= pair.reference.shape[0]
            }
            try:
                detection = detect_clouds(roles, reflectance_scale=reflectance_scale)
                cloud = detection.cloud
                shadow = detection.shadow if detection.shadow.any() else None
            except MaskingError:
                cloud = None
            if (
                cloud is not None
                and shadow is None
                and "nir" in roles
                and sun_elevation_deg is not None
                and sun_azimuth_deg is not None
            ):
                pixel_size = 10.0
                if ref_meta.georef.transform is not None:
                    pixel_size = abs(ref_meta.georef.transform[0]) or 10.0
                shadow_result = detect_shadows(
                    cloud,
                    roles["nir"],
                    sun_elevation_deg,
                    sun_azimuth_deg,
                    pixel_size,
                    valid_mask=~nodata_mask,
                )
                shadow = shadow_result.shadow
            if "green" in roles and "swir" in roles:
                snow = detect_snow(
                    roles["green"],
                    roles["swir"],
                    roles.get("nir"),
                    reflectance_scale=reflectance_scale,
                )
            if "green" in roles and ("nir" in roles or "swir" in roles):
                water = detect_water(
                    roles["green"],
                    nir=roles.get("nir"),
                    swir=roles.get("swir"),
                    reflectance_scale=reflectance_scale,
                )
        return combine_masks(
            shape, cloud=cloud, shadow=shadow, snow=snow, water=water, nodata=nodata_mask
        )
