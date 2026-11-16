"""Report JSON schema, validation and text rendering (DESIGN 5.10). Owned by WP6."""

from __future__ import annotations

import pathlib
from typing import Any

from ..types import REPORT_SCHEMA_VERSION, Report

__all__ = ["REPORT_SCHEMA_VERSION", "validate", "render_text", "write_report"]


def validate(report: dict[str, Any]) -> None:
    """validate(report) raises ValueError naming the first field that breaks the schema of version
    ``REPORT_SCHEMA_VERSION``."""
    raise NotImplementedError("validate(report): WP6, DESIGN 5.10")


def render_text(report: Report) -> str:
    """render_text(report) -> report.txt: headline line, then one section per block with message lines
    first and numbers after, flags marked ``FLAG:`` and failures ``FAIL:``. Nothing absent from the JSON."""
    raise NotImplementedError("render_text(report) -> str: WP6, DESIGN 5.10")


def write_report(report: Report, run_dir: str | pathlib.Path) -> None:
    """write_report(report, run_dir): validate, then write report.json and report.txt atomically."""
    raise NotImplementedError("write_report(report, run_dir): WP6, DESIGN 5.10")
