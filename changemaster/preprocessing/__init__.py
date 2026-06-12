"""Preprocessing package (Phase 2) — the technical heart of ChangeMaster.

حزمة المعالجة المسبقة (المرحلة الثانية): بوابة الجودة، التسجيل الهندسي،
التطبيع الإشعاعي، الأقنعة، معالجة SAR، التوحيد، وأنبوب المعالجة الكامل.

Public surface: quality gate, geometric co-registration, radiometric
normalization (histogram / IR-MAD / PIF), validity masking, SAR
calibration & speckle filtering, pair harmonization and the
:class:`PreprocessingPipeline` orchestrator.
"""

from changemaster.preprocessing.harmonize import (
    HarmonizedPair,
    common_bands,
    harmonize_arrays,
    reproject_to_reference,
)
from changemaster.preprocessing.pipeline import (
    OPTICAL_STEPS,
    SAR_STEPS,
    PreprocessingPipeline,
    PreprocessingReport,
    StepRecord,
)
from changemaster.preprocessing.quality import (
    RECOMMEND_PROCEED,
    RECOMMEND_REJECT,
    RECOMMEND_WARN,
    QualityReport,
    assess_quality,
)

__all__ = [
    "OPTICAL_STEPS",
    "RECOMMEND_PROCEED",
    "RECOMMEND_REJECT",
    "RECOMMEND_WARN",
    "SAR_STEPS",
    "HarmonizedPair",
    "PreprocessingPipeline",
    "PreprocessingReport",
    "QualityReport",
    "StepRecord",
    "assess_quality",
    "common_bands",
    "harmonize_arrays",
    "reproject_to_reference",
]
