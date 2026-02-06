import __init_path__
# refactor it by using unittest and pytest
import unittest
from src.translators import LLM, translation
from src.SRT.srt import SrtScript

class TestTranslation(unittest.TestCase):
    def test_LLM_task(self):
        task = "如果输入中含有‘测试’则返回true，若没有则返回false"
        assert "true" in LLM.LLM_task("gpt-4", "测试", task, temp = 0.15),"LLM prompt 1 failed"
        assert "false" in LLM.LLM_task("gpt-4", "错误样例", task, temp = 0.15),"LLM prompt 2 failed"

    def test_pmptsel(self):
        prompt = translation.prompt_selector("test1", "test2", "test3")
        assert "test1" in prompt,"prompt selector failed"
        assert "test2" in prompt,"prompt selector failed"
        assert "test3" in prompt,"prompt selector failed"

    def test_translation_def(self): 
        test1 = SrtScript.parse_from_srt_file(src_lang="EN", tgt_lang="ZH", domain="General", path="tests/translation_test/test1.srt")
        translation.get_translation(test1, "gpt-4", video_name = "v-name")
        assert test1.segments[1].translation != "", "translation write in & empty prompt failed"

    def test_translation_pmpt(self):
        test1 = SrtScript.parse_from_srt_file(src_lang="EN", tgt_lang="ZH", domain="General", path="tests/translation_test/test1.srt")
        pmpt = translation.prompt_selector("EN", "ZH", "StarCraft2")
        translation.get_translation(test1, "gpt-4", "v-name", pmpt)
        task = "输入如果为中文，返回true,反之返回false"
        assert "true" in LLM.LLM_task("gpt-4", test1.segments[1].translation, task, temp = 0.15),"EN to ZH failed"

    def test_translation_pmpt2(self):
        test1 = SrtScript.parse_from_srt_file(src_lang="ZH", tgt_lang="EN", domain="General", path="tests/translation_test/test1.srt")
        pmpt = translation.prompt_selector("ZH", "EN", "StarCraft2")
        translation.get_translation(test1, "gpt-4", "v-name", pmpt)
        task = "输入如果为英文，返回true,反之返回false"
        assert "true" in LLM.LLM_task("gpt-4", test1.segments[1].translation, task, temp = 0.15),"ZH to EN failed"

    def test_translation_pmpt3(self):
        test1 = SrtScript.parse_from_srt_file(src_lang="ZH", tgt_lang="ZH", domain="General", path="tests/translation_test/test1.srt")
        pmpt = translation.prompt_selector("ZH", "ZH", "StarCraft2")
        translation.get_translation(test1, "gpt-4", "v-name", pmpt)
        task = "输入如果为中文，返回true,反之返回false"
        assert "true" in LLM.LLM_task("gpt-4", test1.segments[1].translation, task, temp = 0.15), "ZH to ZH failed"

if __name__ == "__main__":
    unittest.main()

# test_pmptsel()
# test_LLM_task()
# test_translation_def()
# test_translation_pmpt()
# test_translation_pmpt2()
# test_translation_pmpt3()