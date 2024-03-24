import __init_lib_path
import gradio as gr
from src.task import Task
import logging
from yaml import Loader, Dumper, load, dump
import os
from pathlib import Path
from datetime import datetime
import shutil
from uuid import uuid4
import torch
import stable_whisper

launch_config = "./configs/local_launch.yaml"
task_config = './configs/task_config.yaml'


model_dict = {"stable_large": None, "stable_medium": None, "stable_base": None}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def init(opt_post, opt_pre, output_type, src_lang, tgt_lang, domain, opt_asr_method, chunk_size, translation_model):
    launch_cfg = load(open(launch_config), Loader=Loader)
    task_cfg = load(open(task_config), Loader=Loader)

    # overwrite config file
    task_cfg["source_lang"] = src_lang
    if src_lang == "ZH":
        task_cfg["translation"]["chunk_size"] = 100
        # auto set the chunk size for ZH input
        
    task_cfg["target_lang"] = tgt_lang
    task_cfg["field"] = domain
    task_cfg["ASR"]["ASR_model"] = opt_asr_method

    # ASR model pre-load
    pre_load_asr_model = None
    if opt_asr_method == "stable-whisper-base":
        if model_dict["stable_base"] is None:
            model_dict["stable_base"] = stable_whisper.load_model("base", device)
        pre_load_asr_model = model_dict["stable_base"]
    elif opt_asr_method == "stable-whisper-medium":
        if model_dict["stable_medium"] is None:
            model_dict["stable_medium"] = stable_whisper.load_model("medium", device)
        pre_load_asr_model = model_dict["stable_medium"]
    elif opt_asr_method == "stable-whisper-large":
        if model_dict["stable_large"] is None:
            model_dict["stable_large"] = stable_whisper.load_model("large", device)
        pre_load_asr_model = model_dict["stable_large"]

    if translation_model == "SC2 Domain Expert(beta test)": 
        task_cfg["translation"]["model"] = "Assistant"
    else:
        task_cfg["translation"]["model"] = translation_model

    if "Video File" in output_type:
        task_cfg["output_type"]["video"] = True
    else:
        task_cfg["output_type"]["video"] = False
    
    if "Bilingual" in output_type:
        task_cfg["output_type"]["bilingual"] = True
    else:
        task_cfg["output_type"]["bilingual"] = False
    
    if ".ass output" in output_type:
        task_cfg["output_type"]["subtitle"] = "ass"
    else:
        task_cfg["output_type"]["subtitle"] = "srt"

    task_cfg["pre_process"]["sentence_form"] = True if "Sentence form" in opt_pre else False
    task_cfg["pre_process"]["spell_check"] = True if "Spell Check" in opt_pre else False

    if task_cfg["translation"]["model"] == "Assistant":
        task_cfg["pre_process"]["term_correct"] = False
        task_cfg["field"] = "SC2"

    task_cfg["post_process"]["check_len_and_split"] = True if "Split Sentence" in opt_post else False
    task_cfg["post_process"]["remove_trans_punctuation"] = True if "Remove Punc" in opt_post else False

    task_cfg["translation"]["chunk_size"] = chunk_size
    # initialize dir
    local_dir = Path(launch_cfg['local_dump'])
    if not local_dir.exists():
        local_dir.mkdir(parents=False, exist_ok=False)

    # get task id
    task_id = str(uuid4())

    # create locak dir for the task
    task_dir = local_dir.joinpath(f"task_{task_id}")
    task_dir.mkdir(parents=False, exist_ok=False)
    task_dir.joinpath("results").mkdir(parents=False, exist_ok=False)

    return task_id, task_dir, task_cfg, pre_load_asr_model

def process_input(video_file, audio_file, srt_file, youtube_link, src_lang, tgt_lang, domain, opt_asr_method, opt_post, opt_pre, output_type, chunk_size, translation_model):
    task_id, task_dir, task_cfg, pre_load_asr_model = init(opt_post, opt_pre, output_type, src_lang, tgt_lang, domain, opt_asr_method, chunk_size, translation_model)
    if youtube_link:
        task = Task.fromYoutubeLink(youtube_link, task_id, task_dir, task_cfg)
        task.run(pre_load_asr_model)
        return task.result
    elif audio_file is not None:
        task = Task.fromAudioFile(audio_file.name, task_id, task_dir, task_cfg)
        task.run(pre_load_asr_model)
        return task.result
    elif srt_file is not None:
        task = Task.fromSRTFile(srt_file.name, task_id, task_dir, task_cfg)
        task.run()
        return task.result
    elif video_file is not None:
        task = Task.fromVideoFile(video_file, task_id, task_dir, task_cfg)
        task.run(pre_load_asr_model)
        return task.result
    else:
        return None



with gr.Blocks() as demo:
    gr.Markdown("# ViDove V0.1.0: Pigeon AI Video Translation Toolkit Demo")
    gr.Markdown("Our Website: https://pigeonai.club/")
    gr.Markdown("ViDove Discussion Group: 749825364")
    gr.Markdown("ViDove Discord: coming soon")
    gr.Markdown("Github: https://github.com/pigeonai-org/ViDove")
    gr.Markdown("Please give us a star on GitHub!")
    gr.Markdown("### Input")
    with gr.Tab("Youtube Link"):
        link = gr.components.Textbox(label="Enter a YouTube URL")
    with gr.Tab("Video File"):
        video = gr.components.Video(label="Upload a video")
    with gr.Tab("Audio File"):
        audio = gr.File(label="Upload an Audio File")
    with gr.Tab("SRT File"):
        srt = gr.File(label="Upload a SRT file")

    gr.Markdown("### Settings")
    with gr.Row():
        opt_src = gr.components.Dropdown(choices=["EN", "ZH", "KR"], label="Select Source Language", value="EN")
        opt_tgt = gr.components.Dropdown(choices=["ZH", "EN", "KR"], label="Select Target Language", value="ZH")
        if opt_src.value == opt_tgt.value:
            gr.Error("Source and Target Language should be different")
        opt_domain = gr.components.Dropdown(choices=["General", "SC2"], label="Select Domain", value="General")
    with gr.Tab("ASR"):
        if device.type == "cuda":
            opt_asr_method = gr.components.Dropdown(choices=["whisper-api", "stable-whisper-base", "stable-whisper-medium", "stable-whisper-large"], label="Select ASR Module Inference Method", value="stable-whisper-large", info="use api if you don't have GPU")
        else:
            opt_asr_method = gr.components.Dropdown(choices=["whisper-api", "stable-whisper-base", "stable-whisper-medium"], label="Select ASR Module Inference Method", value="whisper-api", info="use api if you don't have GPU")
    with gr.Tab("Pre-process"):
        default_pre = ["Sentence form", 'Term Correct'] if opt_src.value == "EN" else []
        opt_pre = gr.CheckboxGroup(["Sentence form", "Spell Check", "Term Correct"], label="Pre-process Module", info="Pre-process module settings", value=default_pre)
    with gr.Tab("Post-process"):
        opt_post = gr.CheckboxGroup(["Split Sentence", "Remove Punc"], label="Post-process Module", info="Post-process module settings", value=["Split Sentence", "Remove Punc"])
    with gr.Tab("Translation"):
        translation_model = gr.Dropdown(choices=["gpt-3.5-turbo", "gpt-4", "gpt-4-1106-preview", "SC2 Domain Expert(beta test)"], label="Select Translation Model", value="gpt-4-1106-preview")
        default_chunksize = 2000 if opt_src.value == "EN" else 100
        chunk_size = gr.Number(value=default_chunksize, info="100 for ZH as source language")
    
    opt_out = gr.CheckboxGroup(["Bilingual"], label="Output Settings", info="What do you want?")

    submit_button = gr.Button("Submit")

    gr.Markdown("### Output")
    file_output = gr.components.File(label="Output")

    submit_button.click(process_input, inputs=[video, audio, srt, link, opt_src, opt_tgt, opt_domain, opt_asr_method, opt_post, opt_pre, opt_out, chunk_size, translation_model], outputs=file_output)
    # def clear():
    #     file_output.clear()

    # clear_btn = gr.Button(value="Clear")
    # clear_btn.click(clear, [], [])
if __name__ == "__main__":
    demo.queue(max_size=5)
    demo.launch()
