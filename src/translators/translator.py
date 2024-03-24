from os import getenv
import traceback
import logging
from openai import OpenAI
from time import sleep
from tqdm import tqdm
from .LLM import LLM
from .assistant import Assistant
from src.srt_util.srt import split_script

SUPPORT_LANG_MAP = {
    "EN": "English",
    "ZH": "Chinese",
    "ES": "Spanish",
    "FR": "France",
    "DE": "Germany",
    "RU": "Russian",
    "JA": "Japanese",
    "AR": "Arabic",
    "KR": "Korean",
}

class Translator:
    def __init__(self, model_name, src_lang, tgt_lang, domain, task_id, chunk_size = 1000):
        logging.info("initializing translator")
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.domain = domain
        self.task_id = task_id
        self.system_prompt = self.prompt_selector()
        self.srt = None

        if self.model_name == "Assistant":
            self.translator = Assistant(system_prompt = self.system_prompt, domain = domain)
        elif self.model_name in ["gpt-3.5-turbo", "gpt-4", "gpt-4-1106-preview"]:
            self.translator = LLM(self.model_name, system_prompt = self.system_prompt)
        else:
            print(f"Unsupported model name: {self.model_name}")
            raise NotImplementedError
    
        logging.info(f"Using {self.model_name} as translation model")

    def set_srt(self, srt):
        self.srt = srt
        self.script_arr, self.range_arr = split_script(srt.get_source_only(), self.chunk_size)
        logging.info("SRT file set")
    
    def prompt_selector(self):
        try:
            src_lang = SUPPORT_LANG_MAP[self.src_lang]
            tgt_lang = SUPPORT_LANG_MAP[self.tgt_lang]
            assert src_lang != tgt_lang
        except:
            print("Unsupported language, is your abbreviation correct?")
            print(f"supported language map: {SUPPORT_LANG_MAP}")
            logging.info(f"Unsupported language detected: {src_lang} to {tgt_lang}")
            
        prompt = f"""
            you are a translation assistant, your job is to translate a video in domain of {self.domain} from {src_lang} to {tgt_lang}, 
            you will be provided with a segement in {src_lang} parsed by line, where your translation text should keep the original 
            meaning and the number of lines.
            """
        logging.info(f"System Prompt: {prompt}")
        return prompt
    
    def translate(self):
        """
        Translates the given script array into another language using the chatgpt and writes to the SRT file.

        This function takes a script array, a range array, a model name, a video name, and a video link as input. It iterates
        through sentences and range in the script and range arrays. If the translation check fails for five times, the function
        will attempt to resolve merge sentence issues and split the sentence into smaller tokens for a better translation.
        """

        if self.srt is None:
            raise ValueError("SRT file not set")
        
        if self.system_prompt is None:
            self.system_prompt = "你是一个翻译助理，你的任务是翻译视频，你会被提供一个按行分割的英文段落，你需要在保证句意和行数的情况下输出翻译后的文本。"
            logging.info(f"translation prompt: {self.system_prompt}")
        previous_length = 0
        for sentence, range_ in tqdm(zip(self.script_arr, self.range_arr)):
            # update the range based on previous length
            range_ = (range_[0] + previous_length, range_[1] + previous_length)

            print(f"now translating sentences {range_}")
            logging.info(f"now translating sentences {range_}")
            flag = True
            while flag:
                flag = False
                try:
                    translation = self.translator.send_request(sentence)
                except Exception as e:
                    print("An error has occurred during translation:", e)
                    print(traceback.format_exc())
                    logging.debug("An error has occurred during translation:", e)
                    logging.info("Retrying... the script will continue after 30 seconds.")
                    sleep(30)
                    flag = True

            logging.info(f"source text: {sentence}")
            logging.info(f"translate text: {translation}")
            self.srt.set_translation(translation, range_, self.model_name, self.task_id)


        
