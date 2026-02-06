import __init_path__

import unittest

import src.dict_util as dict_util
from src.SRT.srt import SrtScript

class TestPreprocess(unittest.TestCase):

    def setUp(self):
        self.test1 = "the zerg player send hydra to the forge and engin bay"
        self.test2 = "and succeeded."
        self.en_dict_test = [["engineering bay", "engin bay"],["forge"],["hydralisk", "hydra"],["zerg"]]
        self.zh_dict_test = [["工程站", "BE"],["BF", "锻炉"],["刺蛇"],["虫族"]]


    def form_srt_class(self, src_lang="EN", tgt_lang="ZH", source_text="", translation="", source_text2 = "", duration="00:00:00,000 --> 00:00:02,220"):
        segments = [[0, duration, source_text, translation, ""]]
        if source_text2:
            segments.append([1, "00:00:02,600 --> 00:00:03,620", source_text2, translation, ""])
        return SrtScript(src_lang, tgt_lang, segments)

    def form_dict(self, src_dict, tgt_dict) -> dict:
        test_dict = dict_util.form_dict(src_dict, tgt_dict)
        return test_dict

    def test_sentence_form(self): #wait for comments about function
        srt = self.form_srt_class(source_text=self.test1, source_text2=self.test2)
        srt.form_whole_sentence()
        assert srt.segments[0].translation == ""

    def test_en_spell_check(self):
        srt = self.form_srt_class(source_text=self.test1)
        srt.dict = self.form_dict(self.en_dict_test, self.zh_dict_test)
        srt.spell_check_term()
        assert srt.segments[0].translation == ""

    def test_term_correct(self):
        srt = self.form_srt_class(src_lang="EN", tgt_lang="ZH", source_text=self.test1)
        srt.dict = self.form_dict(self.en_dict_test, self.zh_dict_test)
        srt.correct_with_force_term()
        trans = srt.segments[0].source_text
        assert trans[:-10] == "the 虫族 player send 刺蛇 to the" and trans[-9:-7] in {"锻炉", "BF"} and trans[-2:] in {"工程站", "BE"}

if __name__ == "__main__":
    unittest.main()
