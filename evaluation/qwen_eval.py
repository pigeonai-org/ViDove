import os
import argparse
import torch
import soundfile as sf
from pathlib import Path
import time
import threading
import signal
import sys

from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info

# if you want to run this file, torch is required to be at least 2.6.0.

class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("Operation timed out")

def progress_monitor(stop_event):
    """Monitor progress and show that the process is still running"""
    dots = 0
    while not stop_event.is_set():
        print(f"\rGenerating{'.' * (dots % 4)}{' ' * (3 - (dots % 4))}", end='', flush=True)
        dots += 1
        time.sleep(0.5)
    print()  # New line when done

def parse_args():
    parser = argparse.ArgumentParser(description="Qwen2.5-Omni Video Translation")
    parser.add_argument("--video_path", type=str, default="./evaluation/test_data/videos/0k6b5W_fb4A_21.mp4",required=True, help="Path to the video file")
    parser.add_argument("--output_path", type=str, default="./evaluation/test_data/qwen_result.txt", help="Path to save the translation text")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-Omni-7B", help="Path to the model")
    parser.add_argument("--source_lang", type=str, default="auto", help="Source language (auto for auto-detection)")
    parser.add_argument("--target_lang", type=str, default="en", help="Target language (en, zh, etc.)")
    parser.add_argument("--use_audio", action="store_true", help="Use audio in video for better translation")
    parser.add_argument("--save_audio", action="store_true", help="Save audio output")
    parser.add_argument("--use_flash_attn", action="store_true", help="Use Flash Attention 2 for faster inference")
    parser.add_argument("--timeout", type=int, default=300, help="Timeout in seconds for generation (default: 300)")
    return parser.parse_args()

def translate_video(video_path, model_path, source_lang="auto", target_lang="en", 
                   use_audio=True, save_audio=False, use_flash_attn=False, timeout=300):
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
        timeout: Timeout in seconds for generation (default: 300)
        
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
                {"type": "text", "text": f"Please translate all spoken content {lang_instruction}to {target_lang}. Provide the translation only without any explanations."}
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

    # Print debug info
    print(f"Input shapes - Video: {videos[0].shape if videos else 'None'}")
    print(f"Input shapes - Audio: {audios[0].shape if audios else 'None'}")
    print(f"Input shapes - Images: {len(images) if images else 0} images")
    print(f"Model device: {model.device}")
    print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB" if torch.cuda.is_available() else "CPU mode")
    if torch.cuda.is_available():
        print(f"Used GPU memory: {torch.cuda.memory_allocated() / 1024**3:.1f}GB")

    # Generate translation with timeout and progress monitoring
    print("Generating translation...")
    print(f"Timeout set to {timeout} seconds. Press Ctrl+C to interrupt manually.")
    
    # Start progress monitor
    stop_event = threading.Event()
    progress_thread = threading.Thread(target=progress_monitor, args=(stop_event,))
    progress_thread.daemon = True
    progress_thread.start()
    
    try:
        # Set up timeout
        if os.name != 'nt':  # Unix systems
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout)
        
        start_time = time.time()
        
        # Generate with optimized parameters
        generation_kwargs = {
            **inputs,
            "use_audio_in_video": use_audio,
            "max_new_tokens": 512,  # Limit output length
            "do_sample": False,     # Use greedy decoding for faster generation
            "num_beams": 1,         # No beam search for speed
        }
        
        if save_audio:
            print("Generating with audio output...")
            text_ids, audio = model.generate(**generation_kwargs)
        else:
            generation_kwargs["return_audio"] = False
            text_ids = model.generate(**generation_kwargs)
        
        generation_time = time.time() - start_time
        print(f"\nGeneration completed in {generation_time:.2f} seconds")
        
        # Cancel timeout
        if os.name != 'nt':
            signal.alarm(0)
            
    except KeyboardInterrupt:
        print("\nGeneration interrupted by user")
        sys.exit(1)
    except TimeoutError:
        print(f"\nGeneration timed out after {timeout} seconds")
        sys.exit(1)
    except Exception as e:
        print(f"\nGeneration failed with error: {e}")
        print(f"Error type: {type(e).__name__}")
        if torch.cuda.is_available():
            print(f"GPU memory after error: {torch.cuda.memory_allocated() / 1024**3:.1f}GB")
        raise
    finally:
        # Stop progress monitor
        stop_event.set()
        progress_thread.join(timeout=1)
    
    # Decode translation
    print("Decoding translation...")
    translation = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    
    # Save audio if requested
    if save_audio:
        print("Saving audio output...")
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
        args.use_flash_attn,
        args.timeout
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
