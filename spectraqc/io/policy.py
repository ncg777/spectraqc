"""Input policy evaluation for audio metadata."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from spectraqc.types import AudioBuffer, Status


def _to_lower_list(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    return [str(v).strip().lower() for v in values if str(v).strip()]


def _append_check(checks: list[dict], *, rule: str, status: Status, expected: Any, actual: Any, notes: str = "") -> None:
    checks.append(
        {
            "rule": rule,
            "status": status.value,
            "expected": expected,
            "actual": actual,
            "notes": notes,
        }
    )


def evaluate_input_policy(policy: dict | None, *, audio: AudioBuffer, audio_path: str) -> dict:
    """Evaluate input policy against audio metadata."""
    policy_cfg = policy or {}
    checks: list[dict] = []
    ext = Path(audio_path).suffix.lower()
    fs_hz = float(audio.fs)
    channels = int(audio.channels)
    duration = float(audio.duration)

    accepted_formats = _to_lower_list(policy_cfg.get("accepted_formats"))
    if accepted_formats:
        status = Status.PASS if ext in accepted_formats else Status.FAIL
        _append_check(
            checks,
            rule="accepted_formats",
            status=status,
            expected=accepted_formats,
            actual=ext,
        )

    sample_rate_cfg = policy_cfg.get("sample_rate_hz", {}) if isinstance(policy_cfg.get("sample_rate_hz", {}), dict) else {}
    allowed_rates = sample_rate_cfg.get("allowed")
    if isinstance(allowed_rates, list) and allowed_rates:
        rounded = int(round(fs_hz))
        status = Status.PASS if rounded in [int(round(v)) for v in allowed_rates] else Status.FAIL
        _append_check(
            checks,
            rule="sample_rate_hz.allowed",
            status=status,
            expected=allowed_rates,
            actual=rounded,
        )
    min_rate = sample_rate_cfg.get("min")
    max_rate = sample_rate_cfg.get("max")
    if min_rate is not None or max_rate is not None:
        status = Status.PASS
        if min_rate is not None and fs_hz < float(min_rate):
            status = Status.FAIL
        if max_rate is not None and fs_hz > float(max_rate):
            status = Status.FAIL
        _append_check(
            checks,
            rule="sample_rate_hz.range",
            status=status,
            expected={"min": min_rate, "max": max_rate},
            actual=fs_hz,
        )

    channels_cfg = policy_cfg.get("channels", {}) if isinstance(policy_cfg.get("channels", {}), dict) else {}
    allowed_channels = channels_cfg.get("allowed")
    if isinstance(allowed_channels, list) and allowed_channels:
        status = Status.PASS if channels in [int(v) for v in allowed_channels] else Status.FAIL
        _append_check(
            checks,
            rule="channels.allowed",
            status=status,
            expected=allowed_channels,
            actual=channels,
        )
    min_channels = channels_cfg.get("min")
    max_channels = channels_cfg.get("max")
    if min_channels is not None or max_channels is not None:
        status = Status.PASS
        if min_channels is not None and channels < int(min_channels):
            status = Status.FAIL
        if max_channels is not None and channels > int(max_channels):
            status = Status.FAIL
        _append_check(
            checks,
            rule="channels.range",
            status=status,
            expected={"min": min_channels, "max": max_channels},
            actual=channels,
        )

    duration_cfg = policy_cfg.get("duration_s", {}) if isinstance(policy_cfg.get("duration_s", {}), dict) else {}
    min_duration = duration_cfg.get("min")
    max_duration = duration_cfg.get("max")
    if min_duration is not None or max_duration is not None:
        status = Status.PASS
        if min_duration is not None and duration < float(min_duration):
            status = Status.FAIL
        if max_duration is not None and duration > float(max_duration):
            status = Status.FAIL
        _append_check(
            checks,
            rule="duration_s.range",
            status=status,
            expected={"min": min_duration, "max": max_duration},
            actual=duration,
        )

    allowed_backends = _to_lower_list(policy_cfg.get("decode_backends"))
    if allowed_backends:
        status = Status.PASS if audio.backend.lower() in allowed_backends else Status.FAIL
        _append_check(
            checks,
            rule="decode_backends",
            status=status,
            expected=allowed_backends,
            actual=audio.backend,
        )

    warn_on_decode_warnings = bool(policy_cfg.get("warn_on_decode_warnings", False))
    if warn_on_decode_warnings:
        status = Status.WARN if audio.warnings else Status.PASS
        _append_check(
            checks,
            rule="decode_warnings",
            status=status,
            expected="no warnings",
            actual=len(audio.warnings),
            notes="decode_warnings_present" if audio.warnings else "",
        )

    overall = Status.PASS
    if any(c["status"] == Status.FAIL.value for c in checks):
        overall = Status.FAIL
    elif any(c["status"] == Status.WARN.value for c in checks):
        overall = Status.WARN

    return {
        "status": overall.value,
        "checks": checks,
        "applied": bool(checks),
    }
