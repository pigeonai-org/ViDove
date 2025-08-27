import os
import argparse
import torch
import soundfile as sf
from pathlib import Path

from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info

# if you want to run this file, torch is required to be at least 2.6.0.


def parse_args():
    parser = argparse.ArgumentParser(description="Qwen2.5-Omni Video Translation")
    parser.add_argument("--video_path", type=str, default="./evaluation/test_data/videos/test.mp4",required=True, help="Path to the video file")
    parser.add_argument("--output_path", type=str, default="./evaluation/test_data/qwen_result.txt", help="Path to save the translation text")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-Omni-7B", help="Path to the model")
    parser.add_argument("--source_lang", type=str, default="auto", help="Source language (auto for auto-detection)")
    parser.add_argument("--target_lang", type=str, default="en", help="Target language (en, zh, etc.)")
    parser.add_argument("--use_audio", action="store_true", help="Use audio in video for better translation")
    parser.add_argument("--save_audio", action="store_true", help="Save audio output")
    parser.add_argument("--use_flash_attn", action="store_true", help="Use Flash Attention 2 for faster inference")
    return parser.parse_args()

def translate_video(video_path, model_path, source_lang="auto", target_lang="en", 
                   use_audio=True, save_audio=False, use_flash_attn=False):
    """
    Translate content from a video using Qwen2.5-Omni model
    
    Args:
        video_path: Path to the video file
        model_path: Path to the Qwen2.5-Omni model
        source_lang: Source language (auto for auto-detection)
        target_lang: Target language for translation
        use_audio: Whether to use audio in video
        save_audio: Whether to save audio output
        use_flash_attn: Whether to use Flash Attention 2
        
    Returns:
        Translation text
    """
    # Load model and processor
    print(f"Loading model from {model_path}...")
    model_kwargs = {
        "torch_dtype": "auto",
        "device_map": "auto",
    }
    
    if use_flash_attn:
        model_kwargs["attn_implementation"] = "flash_attention_2"
    
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(model_path, **model_kwargs)
    processor = Qwen2_5OmniProcessor.from_pretrained(model_path)
    
    if not save_audio:
        model.disable_talker()
    
    # Prepare conversation with translation instruction
    lang_instruction = ""
    if source_lang != "auto":
        lang_instruction = f"from {source_lang} "
    
    # conversation = [
    #     {
    #         "role": "system",
    #         "content": [
    #             {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
    #         ],
    #     },
    #     {
    #         "role": "user",
    #         "content": [
    #             {"type": "video", "video": video_path},
    #             {"type": "text", "text": f"Please translate all spoken content {lang_instruction}to {target_lang}. Provide the translation only without any explanations."}
    #         ],
    #     },
    # ]
    conversation = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "video", "video": video_path},
                {"type": "text", "text": f"翻译提供的视频到中文。只需要输出翻译内容原文，不要输出任何解释。"}
            ],
        },
    ]

    # Process the input
    print("Processing video...")
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=use_audio)
    inputs = processor(text=text, audio=audios, images=images, videos=videos, 
                      return_tensors="pt", padding=True, use_audio_in_video=use_audio)
    inputs = inputs.to(model.device).to(model.dtype)

    # Generate translation
    print("Generating translation...")
    if save_audio:
        print("saving audio")
        text_ids, audio = model.generate(**inputs, use_audio_in_video=use_audio)
    else:
        text_ids = model.generate(**inputs, use_audio_in_video=use_audio, return_audio=False)
    
    # Decode translation
    print("decoding")
    translation = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    
    # Save audio if requested
    if save_audio:
        print("saving audio")
        audio_output_path = os.path.splitext(video_path)[0] + "_translation.wav"
        sf.write(
            audio_output_path,
            audio.reshape(-1).detach().cpu().numpy(),
            samplerate=24000,
        )
        print(f"Audio saved to {audio_output_path}")
    
    return translation

def main():
    args = parse_args()
    
    # Check if video exists
    if not os.path.exists(args.video_path):
        raise FileNotFoundError(f"Video file not found: {args.video_path}")
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    
    # Translate video
    translation = translate_video(
        args.video_path, 
        args.model_path, 
        args.source_lang, 
        args.target_lang,
        args.use_audio,
        args.save_audio,
        args.use_flash_attn
    )
    
    # Save translation to file
    with open(args.output_path, "w", encoding="utf-8") as f:
        f.write(translation)
    
    print(f"Translation saved to {args.output_path}")
    print("\nTranslation result:")
    print("-" * 50)
    print(translation)
    print("-" * 50)

if __name__ == "__main__":
    main()