import __init_path__
import unittest

from entries.config_schema import TaskConfig
from src.SRT.srt import SrtScript


class TestTranslation(unittest.TestCase):
    def test_parse_source_only_srt(self):
        srt = SrtScript.parse_from_srt_file(
            src_lang="EN",
            tgt_lang="ZH",
            domain="General",
            srt_str=(
                "1\n"
                "00:00:00,000 --> 00:00:01,000\n"
                "Hello there\n\n"
                "2\n"
                "00:00:01,000 --> 00:00:02,500\n"
                "General Kenobi\n"
            ),
        )

        self.assertEqual([seg.src_text for seg in srt.segments], ["Hello there", "General Kenobi"])
        self.assertEqual([seg.translation for seg in srt.segments], ["", ""])

    def test_parse_bilingual_srt(self):
        srt = SrtScript.parse_from_srt_file(
            src_lang="EN",
            tgt_lang="ZH",
            domain="General",
            srt_str=(
                "1\n"
                "00:00:00,000 --> 00:00:01,000\n"
                "Hello there\n"
                "你好\n"
            ),
        )

        self.assertEqual(len(srt.segments), 1)
        self.assertEqual(srt.segments[0].src_text, "Hello there")
        self.assertEqual(srt.segments[0].translation, "你好")

    def test_task_config_normalizes_language_codes(self):
        cfg = TaskConfig(
            source_lang="en",
            target_lang="zh",
            audio={"src_lang": "en", "tgt_lang": "zh"},
        )

        self.assertEqual(cfg.source_lang, "EN")
        self.assertEqual(cfg.target_lang, "ZH")
        self.assertEqual(cfg.audio.src_lang, "EN")
        self.assertEqual(cfg.audio.tgt_lang, "ZH")

if __name__ == "__main__":
    unittest.main()
