import __init_path__

import unittest

from src.output_render import (
    DEFAULT_SUBTITLE_FONT_CANDIDATES,
    build_subtitle_filter,
)


class TestOutputRender(unittest.TestCase):
    def test_build_subtitle_filter_uses_fonts_dir_and_open_source_font(self):
        subtitle_filter = build_subtitle_filter(
            "/tmp/vidove/test'subtitle.srt",
            fonts_dir="/app/ViDove/fonts",
            font_name="Source Han Sans CN",
        )

        self.assertIn("subtitles='/tmp/vidove/test\\'subtitle.srt'", subtitle_filter)
        self.assertIn("fontsdir='/app/ViDove/fonts'", subtitle_filter)
        self.assertIn("FontName=Source Han Sans CN", subtitle_filter)

    def test_font_fallback_candidates_prioritize_open_source_cjk_fonts(self):
        self.assertEqual(
            DEFAULT_SUBTITLE_FONT_CANDIDATES,
            (
                "Source Han Sans CN",
                "Noto Sans CJK SC",
                "WenQuanYi Zen Hei",
            ),
        )


if __name__ == "__main__":
    unittest.main()
