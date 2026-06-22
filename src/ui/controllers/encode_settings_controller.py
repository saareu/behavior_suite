"""Typed Encode Settings updates and crop/run invalidation policy."""

from __future__ import annotations

from pydantic import ValidationError

from preprocess.config import (
    CanonicalResolutionConfig,
    DebugConfig,
    EncodingConfig,
    FFmpegEncodingConfig,
    OpenCVEncodingConfig,
    PrepareConfig,
    PreprocessConfig,
)
from ui.state import PreprocessSetupState


class EncodeSettingsValidationError(ValueError):
    """Expected user-facing encode configuration failure."""


def _geometry_signature(config: PreprocessConfig) -> tuple[bool, int, int]:
    canonical = config.prepare.canonical_resolution
    return canonical.enabled, canonical.width, canonical.height


class EncodeSettingsController:
    """Update only existing validated encoding and canonical config fields."""

    def __init__(self, state: PreprocessSetupState) -> None:
        self.state = state
        if state.original_preprocess_config is None and state.preprocess_config is not None:
            state.original_preprocess_config = PreprocessConfig.model_validate(
                state.preprocess_config.model_dump(mode="python")
            )
            state.config_dirty = False

    def update_settings(
        self,
        *,
        canonical_enabled: bool | None = None,
        canonical_width: int | None = None,
        canonical_height: int | None = None,
        ffmpeg_container: str | None = None,
        ffmpeg_codec: str | None = None,
        ffmpeg_pixel_format: str | None = None,
        ffmpeg_preset: str | None = None,
        ffmpeg_crf: int | None = None,
        ffmpeg_bframes: int | None = None,
        ffmpeg_faststart: bool | None = None,
        opencv_container: str | None = None,
        opencv_fourcc: str | None = None,
        debug_enabled: bool | None = None,
    ) -> PreprocessConfig:
        """Create, validate, and store a typed config copy from GUI edits."""

        current = self.state.preprocess_config
        if current is None:
            raise self._validation_error("Load a preprocess configuration first.")

        current_geometry = _geometry_signature(current)
        canonical_updates = {
            name: value
            for name, value in {
                "enabled": canonical_enabled,
                "width": canonical_width,
                "height": canonical_height,
            }.items()
            if value is not None
        }
        ffmpeg_updates = {
            name: value
            for name, value in {
                "container": ffmpeg_container,
                "codec": ffmpeg_codec,
                "pixel_format": ffmpeg_pixel_format,
                "preset": ffmpeg_preset,
                "crf": ffmpeg_crf,
                "bframes": ffmpeg_bframes,
                "faststart": ffmpeg_faststart,
            }.items()
            if value is not None
        }
        opencv_updates = {
            name: value
            for name, value in {
                "container": opencv_container,
                "fourcc": opencv_fourcc,
            }.items()
            if value is not None
        }
        debug_updates = (
            {} if debug_enabled is None else {"enabled": debug_enabled}
        )

        try:
            canonical = CanonicalResolutionConfig.model_validate(
                {
                    **current.prepare.canonical_resolution.model_dump(),
                    **canonical_updates,
                }
            )
            prepare = PrepareConfig.model_validate(
                {
                    **current.prepare.model_dump(mode="python"),
                    "canonical_resolution": canonical,
                }
            )
            ffmpeg = FFmpegEncodingConfig.model_validate(
                {
                    **current.encoding.ffmpeg.model_dump(mode="python"),
                    **ffmpeg_updates,
                }
            )
            opencv = OpenCVEncodingConfig.model_validate(
                {
                    **current.encoding.opencv.model_dump(mode="python"),
                    **opencv_updates,
                }
            )
            encoding = EncodingConfig(ffmpeg=ffmpeg, opencv=opencv)
            debug = DebugConfig.model_validate(
                {**current.debug.model_dump(), **debug_updates}
            )
            updated = PreprocessConfig.model_validate(
                {
                    **current.model_dump(mode="python"),
                    "prepare": prepare,
                    "encoding": encoding,
                    "debug": debug,
                }
            )
        except (ValidationError, ValueError) as exc:
            self.state.config_dirty = True
            raise self._validation_error(f"Invalid encode setting: {exc}") from exc

        config_changed = current.model_dump(mode="json") != updated.model_dump(mode="json")
        geometry_changed = _geometry_signature(updated) != current_geometry
        if geometry_changed:
            self.state.invalidate_crop_review(
                "Canonical geometry changed; crop review is required."
            )
        elif config_changed:
            self.state.invalidate_run_result("Encode settings changed")
        self.state.store_preprocess_config(updated)
        self.state.encode_settings_valid = True
        self.state.last_validation_error = None
        return self.state.preprocess_config

    def reset_to_loaded_config(self) -> PreprocessConfig:
        """Restore exposed fields from the session's loaded config without writing YAML."""

        current = self.state.preprocess_config
        original = self.state.original_preprocess_config
        if current is None or original is None:
            raise self._validation_error(
                "No originally loaded preprocess configuration is available."
            )
        geometry_changed = _geometry_signature(current) != _geometry_signature(original)
        try:
            prepare = PrepareConfig.model_validate(
                {
                    **current.prepare.model_dump(mode="python"),
                    "canonical_resolution": original.prepare.canonical_resolution,
                }
            )
            restored = PreprocessConfig.model_validate(
                {
                    **current.model_dump(mode="python"),
                    "prepare": prepare,
                    "encoding": original.encoding,
                    "debug": original.debug,
                }
            )
        except (ValidationError, ValueError) as exc:
            raise self._validation_error(
                f"Loaded encode configuration is invalid: {exc}"
            ) from exc
        config_changed = current.model_dump(mode="json") != restored.model_dump(mode="json")
        if geometry_changed:
            self.state.invalidate_crop_review(
                "Canonical geometry reset; crop review is required."
            )
        elif config_changed:
            self.state.invalidate_run_result("Encode settings reset")
        self.state.store_preprocess_config(restored)
        self.state.encode_settings_valid = True
        self.state.last_validation_error = None
        return self.state.preprocess_config

    def validate_navigation(self) -> None:
        """Require a valid config and an accepted crop before Run and Validate."""

        config = self.state.preprocess_config
        if config is None or not self.state.encode_settings_valid:
            raise self._validation_error(
                "Encode settings must contain a valid typed configuration."
            )
        try:
            PreprocessConfig.model_validate(config.model_dump(mode="python"))
        except ValidationError as exc:
            raise self._validation_error(f"Invalid encode setting: {exc}") from exc
        accepted = self.state.accepted_crop_plan
        if accepted is None or not accepted.accepted_by_user:
            raise self._validation_error(
                "Canonical geometry changed; return to Crop Review and accept a new crop.",
                mark_config_invalid=False,
            )
        self.state.last_validation_error = None

    def can_advance(self) -> bool:
        """Return whether Encode Settings can advance to Run and Validate."""

        config = self.state.preprocess_config
        accepted = self.state.accepted_crop_plan
        return bool(
            config is not None
            and self.state.encode_settings_valid
            and accepted is not None
            and accepted.accepted_by_user
        )

    def _validation_error(
        self,
        message: str,
        *,
        mark_config_invalid: bool = True,
    ) -> EncodeSettingsValidationError:
        if mark_config_invalid:
            self.state.encode_settings_valid = False
        self.state.last_validation_error = message
        return EncodeSettingsValidationError(message)
