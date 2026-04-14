import __init_path__
import sys
import types
import unittest

openai_stub = types.ModuleType("openai")


class _AzureOpenAI:
    pass


openai_stub.AzureOpenAI = _AzureOpenAI
sys.modules.setdefault("openai", openai_stub)

from src.SRT.srt import SrtScript, split_script


class ChunkAlignmentTests(unittest.TestCase):
    def test_split_script_preserves_last_chunk_range(self):
        script = "s0\n\ns1\n\ns2\n\ns3\n\ns4"

        chunks, ranges = split_script(script, chunk_size=10)

        self.assertEqual(chunks, ["s0\n\ns1\n\ns2", "s3\n\ns4"])
        self.assertEqual(ranges, [(1, 3), (4, 5)])

    def test_set_translation_writes_full_tail_range(self):
        srt = SrtScript("EN", "ZH", segments=["s0", "s1", "s2", "s3", "s4"], task_id="test")

        srt.set_translation("t0\n\nt1\n\nt2", (1, 3), "test-model", "video")
        srt.set_translation("t3\n\nt4", (4, 5), "test-model", "video")

        self.assertEqual(
            [seg.translation for seg in srt.segments],
            ["t0", "t1", "t2", "t3", "t4"],
        )

    def test_get_source_only_uses_consistent_segment_separator(self):
        srt = SrtScript("EN", "ZH", segments=["alpha", "beta", "gamma"], task_id="test")

        self.assertEqual(srt.get_source_only(), "alpha\n\nbeta\n\ngamma")

    def test_set_translation_strips_label_wrappers(self):
        srt = SrtScript("EN", "ZH", segments=["s0", "s1", "s2"], task_id="test")

        translate = "Your translation:\n\n翻译0\n\n翻译1\n\n翻译2"
        srt.set_translation(translate, (1, 3), "test-model", "video")

        self.assertEqual(
            [seg.translation for seg in srt.segments],
            ["翻译0", "翻译1", "翻译2"],
        )

    def test_set_translation_strips_inline_and_chinese_labels(self):
        srt = SrtScript("EN", "ZH", segments=["s0", "s1"], task_id="test")

        translate = "Your translation: 翻译0\n\n你的翻译：翻译1"
        srt.set_translation(translate, (1, 2), "test-model", "video")

        self.assertEqual(
            [seg.translation for seg in srt.segments],
            ["翻译0", "翻译1"],
        )

    def test_split_seg_returns_two_valid_segments(self):
        srt = SrtScript("EN", "ZH", segments=["hello world, nice to meet you"], task_id="test")
        seg = srt.segments[0]
        seg.start_time = 0.0
        seg.end_time = 4.0
        seg.duration = 4.0
        seg.translation = "你好世界，很高兴认识你们大家"

        result = srt.split_seg(seg, text_threshold=5, time_threshold=1.0)

        self.assertGreaterEqual(len(result), 2)
        for piece in result:
            self.assertIsInstance(piece.src_text, str)
            self.assertTrue(piece.src_text)
            self.assertGreaterEqual(piece.end_time, piece.start_time)


if __name__ == "__main__":
    unittest.main()
