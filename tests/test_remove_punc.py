import unittest
import __init_path__
from src.SRT.srt import SrtScript, SrtSegment

class TestRemovePunc(unittest.TestCase):
    def setUp(self):
        self.zh_test1 = "再次，如果你对一些福利感兴趣，你也可以。"
        self.zh_en_test1 = "GG。Classic在我今年解说的最奇葩的系列赛中获得了胜利。"

    def form_srt_class(self, src_lang, tgt_lang, source_text="", translation="", duration="00:00:00,740 --> 00:00:08,779"):
        segment = [0, duration, source_text, translation, ""]
        return SrtScript(src_lang, tgt_lang, [segment])

    def test_zh(self):
        srt = self.form_srt_class(src_lang="EN", tgt_lang="ZH", translation=self.zh_test1)
        srt.remove_trans_punctuation()
        self.assertEqual(srt.segments[0].translation, "再次 如果你对一些福利感兴趣 你也可以 ")

    def test_zh_en(self):
        srt = self.form_srt_class(src_lang="EN", tgt_lang="ZH", translation=self.zh_en_test1)
        srt.remove_trans_punctuation()
        self.assertEqual(srt.segments[0].translation, "GG Classic在我今年解说的最奇葩的系列赛中获得了胜利 ")

if __name__ == '__main__':
    unittest.main()