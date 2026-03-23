from __future__ import annotations

from pathlib import Path
from typing import Optional


DEFAULT_SUBTITLE_FONT_CANDIDATES = (
    "Source Han Sans CN",
    "Noto Sans CJK SC",
    "WenQuanYi Zen Hei",
)


def escape_ffmpeg_filter_value(value: str) -> str:
    """Escape a value embedded inside an ffmpeg filter argument."""

    replacements = {
        "\\": "\\\\",
        "'": "\\'",
        ":": "\\:",
        ",": "\\,",
        "[": "\\[",
        "]": "\\]",
    }
    for old, new in replacements.items():
        value = value.replace(old, new)
    return value


def build_subtitle_force_style(font_name: Optional[str] = None) -> str:
    style_parts = []
    if font_name:
        style_parts.append(f"FontName={font_name}")
    style_parts.extend(
        [
            "FontSize=24",
            "PrimaryColour=&Hffffff",
            "OutlineColour=&H000000",
            "Bold=1",
        ]
    )
    return ",".join(style_parts)


def build_subtitle_filter(
    subtitle_path: str | Path,
    *,
    fonts_dir: str | Path | None = None,
    font_name: Optional[str] = None,
) -> str:
    """Build a libass subtitles filter with explicit font lookup."""

    filter_parts = [f"subtitles='{escape_ffmpeg_filter_value(str(Path(subtitle_path)))}'"]
    if fonts_dir:
        filter_parts.append(
            f"fontsdir='{escape_ffmpeg_filter_value(str(Path(fonts_dir)))}'"
        )
    filter_parts.append(f"force_style='{build_subtitle_force_style(font_name)}'")
    return ":".join(filter_parts)
