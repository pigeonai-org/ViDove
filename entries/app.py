import __init_lib_path
from pathlib import Path
from uuid import uuid4
from cryptography.fernet import Fernet
import os 
import streamlit as st
import stable_whisper
import torch
from yaml import Loader, load

from src.task import Task

launch_config = "./configs/local_launch.yaml"
task_config = './configs/task_config.yaml'
launch_cfg = load(open(launch_config), Loader=Loader)
LAUNCH_MODE = launch_cfg["environ"]
API_SOURCE = launch_cfg["api_source"]

model_dict = {"stable_large": None, "stable_medium": None, "stable_base": None}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def init(apikey, opt_resolution, opt_post, opt_pre, output_type, src_lang, tgt_lang, domain, opt_asr_method, chunk_size, translation_model):
    task_cfg = load(open(task_config), Loader=Loader)

    # overwrite config file
    resolution = "best" if opt_resolution == "best" else int(opt_resolution[:-1])

    if LAUNCH_MODE == "demo":
        VIDOVE_DECODE_KEY = os.getenv("VIDOVE_DECODE_KEY")
        # overwrite api key
        if apikey != "":
            try:
                fernet = Fernet(VIDOVE_DECODE_KEY.encode())
                apikey = fernet.decrypt(apikey.encode()).decode()
            except:
                raise st.error("Invalid API key")
            if API_SOURCE == "openai":
                task_cfg["OPENAI_API_KEY"] = apikey
            elif API_SOURCE == "azure":
                task_cfg["AZURE_OPENAI_API_KEY"] = apikey
        else:
            if API_SOURCE == "openai":
                task_cfg["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
            elif API_SOURCE == "azure":
                task_cfg["AZURE_OPENAI_API_KEY"] = os.getenv("AZURE_OPENAI_API_KEY")
            translation_model = "gpt-4o-mini"
            resolution = 480
            st.warning("Free Mode: API key is not provided, you can only use gpt-4o-mini model for translation. And the video resolution is set to <=480p.")
    elif LAUNCH_MODE == "local":
        if apikey != "":
            task_cfg["AZURE_OPENAI_API_KEY"] = apikey
        else:
            task_cfg["AZURE_OPENAI_API_KEY"] = os.getenv("AZURE_OPENAI_API_KEY")
    else:
        st.error("Invalid Launch Mode")
        return None, None, None, None

    print(f"API Key: {task_cfg['AZURE_OPENAI_API_KEY']}")
    task_cfg["api_source"] = API_SOURCE
    
    task_cfg["video_download"]["resolution"] = resolution
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

def process_input(apikey, video_file, audio_file, srt_file, youtube_link, opt_resolution, src_lang, tgt_lang, domain, opt_asr_method, opt_post, opt_pre, output_type, chunk_size, translation_model):
    task_id, task_dir, task_cfg, pre_load_asr_model = init(apikey, opt_resolution, opt_post, opt_pre, output_type, src_lang, tgt_lang, domain, opt_asr_method, chunk_size, translation_model)
    
    
    # save video file to the local_dump directory, and then use the path to create a Task instance
        
    if youtube_link:
        
        task = Task.fromYoutubeLink(youtube_link, task_id, task_dir, task_cfg)
        task.run(pre_load_asr_model)
        return task.result, task.log_dir
    elif audio_file is not None:
        audio_file_path = os.path.join(task_dir, audio_file.name)
        with open(audio_file_path, "wb") as f:
            f.write(audio_file.getbuffer())
        
        task = Task.fromAudioFile(audio_file_path, task_id, task_dir, task_cfg)
        task.run(pre_load_asr_model)
        return task.result, task.log_dir
    elif srt_file is not None:
        srt_file_path = os.path.join(task_dir, srt_file.name)
        with open(srt_file_path, "wb") as f:
            f.write(srt_file.getbuffer())
        
        task = Task.fromSRTFile(srt_file_path, task_id, task_dir, task_cfg)
        task.run()
        return task.result, task.log_dir
    elif video_file is not None:
        video_file_path = os.path.join(task_dir, video_file.name)
        with open(video_file_path, "wb") as f:
            f.write(video_file.getbuffer())
        
        # task = Task.fromVideoFile(video_file, task_id, task_dir, task_cfg)
        task = Task.fromVideoFile(video_file_path, task_id, task_dir, task_cfg)
        task.run(pre_load_asr_model)
        return task.result, task.log_dir
    else:
        return None

def main():
    st.set_page_config(page_title="ViDove: Pigeon AI Video Translation Toolkit", layout="wide")
    
    st.title("ViDove V0.1.1: Pigeon AI Video Translation Toolkit Demo")
    
    st.markdown("### General Information")
    st.markdown("[Official Website](https://pigeonai.club/) / [Github](https://github.com/pigeonai-org/ViDove) / [Discord](https://discord.gg/9EcCBvAN87) / [Feedback Form](https://wj.qq.com/s2/14361192/a182/) / [Bilibili](https://space.bilibili.com/195670539)")
    st.markdown("Discussion Group(QQ): 749825364 / Email: gggzmz@163.com")
    st.markdown("**Please give us a star on GitHub!**")

    with st.expander("Update Log"):
        st.markdown("- 2024-04-05: ViDove V0.1.1 is released! Now we support SC2 domain expert translation model.")
        st.markdown("- 2024-04-05: ViDove V0.1.1 已发布! 现在可以使用针对星际2领域的翻译模型.")

    with st.expander("Purchase"):
        st.markdown("Note that you can use our demo without purchasing an API key, but you can only use the **gpt-3.5-turbo** model for translation. If you want to use other models, please purchase an API key.")
        st.markdown("**注意** ：你可以不填写API key即可使用我们的demo,但是你只能使用 **gpt-3.5-turbo** 模型.如果你需要使用更多模型,除了自己本地部署外,可以点击下方链接购买专用API key.")

        st.markdown("[Purchase API Key Here](https://afdian.com/a/gggzmz)")
    
    with st.sidebar:
        st.header("Settings")
        apikey = st.text_input("Insert Your API Key Here", type="password")
        
        st.subheader("Language and Domain Settings")
        col1, col2, col3 = st.columns(3)
        with col1:
            opt_src = st.selectbox("Select Source Language", ["EN", "ZH", "KR"], index=0)
        with col2:
            opt_tgt = st.selectbox("Select Target Language", ["ZH", "EN", "KR"], index=0)
        with col3:
            opt_domain = st.selectbox("Select Dictionary", ["General", "SC2", "CS:GO"], index=0)
            
        if opt_src == opt_tgt:
            st.error("Source and Target Language should be different")
        
        st.subheader("ASR Settings")
        if device.type == "cuda":
            opt_asr_method = st.selectbox(
                "Select ASR Module Inference Method",
                ["whisper-api", "stable-whisper-base", "stable-whisper-medium", "stable-whisper-large"],
                index=3,
                help="use api if you don't have GPU"
            )
        else:
            opt_asr_method = st.selectbox(
                "Select ASR Module Inference Method",
                ["whisper-api", "stable-whisper-base", "stable-whisper-medium"],
                index=0,
                help="use api if you don't have GPU"
            )
            
        st.subheader("Pre-process Settings")
        default_pre = ["Sentence form", 'Term Correct'] if opt_src == "EN" else []
        opt_pre = st.multiselect(
            "Pre-process Module",
            ["Sentence form", "Spell Check", "Term Correct"],
            default=default_pre,
            help="Pre-process module settings"
        )
        
        st.subheader("Post-process Settings")
        opt_post = st.multiselect(
            "Post-process Module",
            ["Split Sentence", "Remove Punc"],
            default=["Split Sentence", "Remove Punc"],
            help="Post-process module settings"
        )
        
        st.subheader("Translation Settings")
        translation_model = st.selectbox(
            "Select Translation Model",
            ["gpt-3.5-turbo", "gpt-4", "gpt-4o", "SC2 Domain Expert(beta test)"],
            index=2
        )
        
        default_chunksize = 2000 if opt_src == "EN" else 100
        chunk_size = st.number_input("Chunk Size", value=default_chunksize, help="100 for ZH as source language")
        
        st.subheader("Output Settings")
        opt_out = st.multiselect("Output Settings", ["Bilingual", "Video File", ".ass output"], default=[], help="What do you want?")
        
        resolution_choices = ["best", "720p", "480p", "360p"]
        if LAUNCH_MODE == "demo":
            resolution_choices = ["720p", "480p", "360p"]
        opt_resolution = st.selectbox("Select Resolution", resolution_choices, index=1)

    # Main content area
    st.header("Input")
    
    tab1, tab2, tab3, tab4 = st.tabs(["YouTube Link", "Video File", "Audio File", "SRT File"])
    
    with tab1:
        youtube_link = st.text_input("Enter a YouTube URL")
    with tab2:
        video_file = st.file_uploader("Upload a video")
        # video_file = st.file_uploader("Upload a video", type=['mp4', 'avi', 'mov', 'mkv'])
    with tab3:
        audio_file = st.file_uploader("Upload an Audio File")
        # audio_file = st.file_uploader("Upload an Audio File", type=['mp3', 'wav'])
    with tab4:
        srt_file = st.file_uploader("Upload a SRT file", type=['srt'])
    
    if st.button("Submit"):
        # Show progress and processing info
        with st.spinner("Processing your request..."):
            result, log_dir = process_input(
                apikey, video_file, audio_file, srt_file, youtube_link, 
                opt_resolution, opt_src, opt_tgt, opt_domain, opt_asr_method, 
                opt_post, opt_pre, opt_out, chunk_size, translation_model
            )
            
            if result:
                st.success("Processing complete!")
                st.header("Output")
                
                with open(result, "rb") as file:
                    st.download_button(
                        label="Download SRT Output",
                        data=file,
                        file_name=os.path.basename(result),
                        mime="text/plain"
                    )
                
                if log_dir:
                    with open(log_dir, "rb") as file:
                        st.download_button(
                            label="Download Log File",
                            data=file,
                            file_name=os.path.basename(log_dir),
                            mime="text/plain"
                        )
                    st.markdown("If you have any issue, please download the log file and send it to us via email or discord.")
            else:
                st.error("Processing failed, please check your input and settings.")

if __name__ == "__main__":
    print(f"Launch Mode: {LAUNCH_MODE}")
    main()
