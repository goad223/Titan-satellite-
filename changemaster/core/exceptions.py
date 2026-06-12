"""Exception hierarchy for ChangeMaster with bilingual (English/Arabic) messages.

Every exception carries both an English and an Arabic message so the GUI and
CLI can present errors in the user's preferred language.
"""

from __future__ import annotations


class ChangeMasterError(Exception):
    """Base class for all ChangeMaster errors.

    Parameters
    ----------
    message_en:
        Human readable English error message.
    message_ar:
        Human readable Arabic error message. Falls back to the English
        message when not provided.
    """

    def __init__(self, message_en: str, message_ar: str | None = None) -> None:
        self.message_en: str = message_en
        self.message_ar: str = message_ar if message_ar is not None else message_en
        super().__init__(message_en)

    def bilingual(self) -> str:
        """Return the combined bilingual message ``"<en> | <ar>"``."""
        if self.message_ar == self.message_en:
            return self.message_en
        return f"{self.message_en} | {self.message_ar}"

    def __str__(self) -> str:
        return self.bilingual()


class HardwareDetectionError(ChangeMasterError):
    """Raised when hardware probing fails irrecoverably."""


class ConfigError(ChangeMasterError):
    """Raised for invalid, unreadable or unwritable configuration."""


class DependencyMissingError(ChangeMasterError):
    """Raised when an optional heavy dependency is required but missing."""

    def __init__(self, package: str, feature_en: str, feature_ar: str) -> None:
        self.package: str = package
        super().__init__(
            f"Optional dependency '{package}' is not installed; "
            f"{feature_en} is unavailable. Install it with: pip install {package}",
            f"الاعتمادية الاختيارية '{package}' غير مثبتة؛ "
            f"{feature_ar} غير متاحة. ثبّتها بالأمر: pip install {package}",
        )


class FormatNotSupportedError(ChangeMasterError):
    """Raised when no registered reader supports a given file."""

    def __init__(self, path: str) -> None:
        self.path: str = path
        super().__init__(
            f"No available reader supports the file: {path}",
            f"لا يوجد قارئ متاح يدعم الملف: {path}",
        )


class ImageReadError(ChangeMasterError):
    """Raised when an image file cannot be opened or read."""


class ImageWriteError(ChangeMasterError):
    """Raised when an image file cannot be written."""


class MetadataError(ChangeMasterError):
    """Raised when image metadata is missing, corrupt or inconsistent."""


class SensorProfileError(ChangeMasterError):
    """Raised for unknown sensors or invalid sensor profile definitions."""
