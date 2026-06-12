"""Tests for harmonization, the preprocessing pipeline and the CLI."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from changemaster.core.exceptions import PipelineError, PreprocessingError
from changemaster.io_engine.metadata import GeoReference, ImageMetadata
from changemaster.preprocessing.harmonize import common_bands, harmonize_arrays
from changemaster.preprocessing.pipeline import (
    OPTICAL_STEPS,
    SAR_STEPS,
    PreprocessingPipeline,
    PreprocessingReport,
    StepRecord,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))


def _meta(
    width: int,
    height: int,
    transform: tuple | None = None,
    crs: str | None = None,
    band_names: list[str] | None = None,
) -> ImageMetadata:
    return ImageMetadata(
        path=Path("synthetic.tif"),
        driver="GTiff",
        width=width,
        height=height,
        band_count=len(band_names) if band_names else 1,
        dtype="float64",
        georef=GeoReference(crs=crs, transform=transform),
        band_names=band_names or [],
    )


class TestCommonBands:
    def test_matches_by_name(self) -> None:
        ref_idx, mov_idx, names = common_bands(
            ["Blue", "Green", "Red"], ["Red", "Blue"]
        )
        assert ref_idx == [1, 3]
        assert mov_idx == [2, 1]
        assert names == ["Blue", "Red"]

    def test_positional_fallback(self) -> None:
        ref_idx, mov_idx, names = common_bands(["A", "B", "C"], ["X", "Y"])
        assert ref_idx == [1, 2]
        assert mov_idx == [1, 2]
        assert names == ["A", "B"]


class TestHarmonizeArrays:
    def test_pixel_space_resize(self) -> None:
        pytest.importorskip("cv2")
        ref = np.random.default_rng(0).random((2, 50, 60))
        mov = np.random.default_rng(1).random((2, 40, 40))
        pair = harmonize_arrays(ref, mov)
        assert pair.moving.shape == pair.reference.shape
        assert pair.warnings  # resize warning emitted

    def test_band_count_trim(self) -> None:
        ref = np.zeros((3, 10, 10))
        mov = np.zeros((2, 10, 10))
        pair = harmonize_arrays(ref, mov)
        assert pair.reference.shape[0] == 2

    def test_georeferenced_overlap_crop(self) -> None:
        # Moving image shifted 10 px east and 5 px south of the reference.
        ref_meta = _meta(
            40, 40, (10.0, 0.0, 0.0, 0.0, -10.0, 0.0), "EPSG:32636", ["Gray"]
        )
        mov_meta = _meta(
            40, 40, (10.0, 0.0, 100.0, 0.0, -10.0, -50.0), "EPSG:32636", ["Gray"]
        )
        ref = np.arange(1600, dtype=np.float64).reshape(1, 40, 40)
        mov = np.arange(1600, dtype=np.float64).reshape(1, 40, 40) + 1000
        pair = harmonize_arrays(ref, mov, ref_meta, mov_meta)
        assert pair.reference.shape == (1, 35, 30)
        assert pair.moving.shape == (1, 35, 30)
        assert pair.georef.crs == "EPSG:32636"

    def test_no_overlap_raises(self) -> None:
        ref_meta = _meta(10, 10, (10.0, 0.0, 0.0, 0.0, -10.0, 0.0), "EPSG:32636", ["G"])
        mov_meta = _meta(
            10, 10, (10.0, 0.0, 100000.0, 0.0, -10.0, 0.0), "EPSG:32636", ["G"]
        )
        with pytest.raises(PreprocessingError):
            harmonize_arrays(np.zeros((1, 10, 10)), np.zeros((1, 10, 10)), ref_meta, mov_meta)

    def test_to_dict(self) -> None:
        pair = harmonize_arrays(np.zeros((1, 5, 5)), np.zeros((1, 5, 5)))
        data = pair.to_dict()
        assert data["shape"] == [1, 5, 5]


class TestReport:
    def test_summary_and_json(self, tmp_path: Path) -> None:
        report = PreprocessingReport(mode="optical")
        report.steps.append(StepRecord(name="quality", status="completed", duration_s=0.5))
        report.rmse_px = 0.4
        report.warnings = ["w1"]
        path = report.save_json(tmp_path / "report.json")
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["mode"] == "optical"
        text = report.summary()
        assert "quality" in text and "0.400" in text


class TestPipeline:
    def test_invalid_mode_raises(self, tmp_path: Path) -> None:
        with pytest.raises(PipelineError):
            PreprocessingPipeline(tmp_path, mode="thermal")

    def test_step_orders(self) -> None:
        assert OPTICAL_STEPS[0] == "quality" and OPTICAL_STEPS[-1] == "radiometric"
        assert SAR_STEPS[1] == "calibration" and SAR_STEPS[2] == "speckle"

    def test_optical_run_full(
        self, tmp_path: Path, optical_pair_png: tuple[Path, Path]
    ) -> None:
        pytest.importorskip("cv2")
        pytest.importorskip("scipy")
        ref_path, mov_path = optical_pair_png
        pipeline = PreprocessingPipeline(tmp_path / "work", mode="optical")
        report = pipeline.run(
            ref_path,
            mov_path,
            band_roles={"blue": 3, "green": 2, "red": 1},
            reflectance_scale=255.0,
        )
        assert [s.name for s in report.steps] == list(OPTICAL_STEPS)
        assert all(s.status == "completed" for s in report.steps)
        assert report.rmse_px is not None and report.rmse_px < 1.0
        assert (tmp_path / "work" / "report.json").exists()
        assert (tmp_path / "work" / "pipeline_state.json").exists()

    def test_optical_resume_uses_checkpoints(
        self, tmp_path: Path, optical_pair_png: tuple[Path, Path]
    ) -> None:
        pytest.importorskip("cv2")
        pytest.importorskip("scipy")
        ref_path, mov_path = optical_pair_png
        workdir = tmp_path / "work"
        pipeline = PreprocessingPipeline(workdir, mode="optical")
        pipeline.run(ref_path, mov_path, reflectance_scale=255.0)
        report = pipeline.run(ref_path, mov_path, resume=True, reflectance_scale=255.0)
        assert all(s.status == "resumed" for s in report.steps)

    def test_sar_run_full(
        self, tmp_path: Path, optical_pair_png: tuple[Path, Path]
    ) -> None:
        pytest.importorskip("cv2")
        ref_path, mov_path = optical_pair_png
        pipeline = PreprocessingPipeline(tmp_path / "sar", mode="sar")
        report = pipeline.run(ref_path, mov_path)
        assert [s.name for s in report.steps] == list(SAR_STEPS)
        assert all(s.status == "completed" for s in report.steps)
        speckle_step = next(s for s in report.steps if s.name == "speckle")
        assert speckle_step.metrics["filter"] == "refined_lee"

    def test_mode_mismatch_on_resume_raises(
        self, tmp_path: Path, optical_pair_png: tuple[Path, Path]
    ) -> None:
        pytest.importorskip("cv2")
        pytest.importorskip("scipy")
        ref_path, mov_path = optical_pair_png
        workdir = tmp_path / "work"
        PreprocessingPipeline(workdir, mode="optical").run(
            ref_path, mov_path, reflectance_scale=255.0
        )
        sar = PreprocessingPipeline(workdir, mode="sar")
        with pytest.raises(PipelineError):
            sar.run(ref_path, mov_path, resume=True)

    def test_corrupt_state_raises(self, tmp_path: Path) -> None:
        workdir = tmp_path / "broken"
        workdir.mkdir()
        (workdir / "pipeline_state.json").write_text("{not json", encoding="utf-8")
        pipeline = PreprocessingPipeline(workdir)
        with pytest.raises(PipelineError):
            pipeline._load_state()

    def test_missing_checkpoint_raises(self, tmp_path: Path) -> None:
        pipeline = PreprocessingPipeline(tmp_path / "w")
        with pytest.raises(PipelineError):
            pipeline._read_checkpoint("harmonize")


class TestCLI:
    def test_parse_band_roles(self) -> None:
        from titan_preprocess import parse_band_roles

        assert parse_band_roles("blue=1, green=2") == {"blue": 1, "green": 2}
        assert parse_band_roles(None) is None
        with pytest.raises(ValueError):
            parse_band_roles("oops")

    def test_cli_optical_run(
        self, tmp_path: Path, optical_pair_png: tuple[Path, Path], capsys: pytest.CaptureFixture
    ) -> None:
        pytest.importorskip("cv2")
        pytest.importorskip("scipy")
        from titan_preprocess import main

        ref_path, mov_path = optical_pair_png
        code = main(
            [
                str(ref_path),
                str(mov_path),
                "--workdir",
                str(tmp_path / "cli"),
                "--reflectance-scale",
                "255",
                "--json",
            ]
        )
        assert code == 0
        data = json.loads(capsys.readouterr().out)
        assert data["mode"] == "optical"

    def test_cli_error_exit_code(self, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
        from titan_preprocess import main

        code = main(
            [
                str(tmp_path / "missing1.png"),
                str(tmp_path / "missing2.png"),
                "--workdir",
                str(tmp_path / "w"),
            ]
        )
        assert code == 1
        assert "خطأ" in capsys.readouterr().err
