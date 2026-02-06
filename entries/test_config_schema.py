#!/usr/bin/env python3
"""
Test configuration schema functionality
"""

import sys
from pathlib import Path

# Add project root directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

from entries.config_schema import (
    TaskConfig, 
    load_task_config, 
    validate_task_config,
    VideoDownloadConfig,
    AudioConfig,
    VisionConfig
)


def test_basic_config():
    """Test basic configuration creation"""
    print("🧪 Testing basic configuration creation...")
    
    config = TaskConfig()
    print(f"✅ Default configuration created successfully")
    print(f"   - Source language: {config.source_lang}")
    print(f"   - Target language: {config.target_lang}")
    print(f"   - Domain: {config.domain}")
    
    return config


def test_custom_config():
    """Test custom configuration"""
    print("\n🧪 Testing custom configuration...")
    
    custom_config = TaskConfig(
        source_lang="EN",
        target_lang="ZH",
        domain="Gaming",
        video_download=VideoDownloadConfig(resolution=720),
        audio=AudioConfig(enable_audio=True, src_lang="en", tgt_lang="zh"),
        vision=VisionConfig(enable_vision=True, frame_per_seg=6)
    )
    
    print(f"✅ Custom configuration created successfully")
    print(f"   - Video resolution: {custom_config.video_download.resolution}")
    print(f"   - Audio processing: {'Enabled' if custom_config.audio.enable_audio else 'Disabled'}")
    print(f"   - Vision processing: {'Enabled' if custom_config.vision.enable_vision else 'Disabled'}")
    print(f"   - Frames per segment: {custom_config.vision.frame_per_seg}")
    
    return custom_config


def test_yaml_loading():
    """Test loading configuration from YAML file"""
    print("\n🧪 Testing YAML file loading...")
    
    config_path = Path(__file__).parent.parent / "configs" / "task_config.yaml"
    
    if config_path.exists():
        try:
            config = load_task_config(config_path)
            print(f"✅ Successfully loaded configuration from YAML file: {config_path}")
            print(f"   - Source language: {config.source_lang}")
            print(f"   - Target language: {config.target_lang}")
            print(f"   - Domain: {config.domain}")
            print(f"   - Audio processing: {'Enabled' if config.audio.enable_audio else 'Disabled'}")
            print(f"   - Vision processing: {'Enabled' if config.vision.enable_vision else 'Disabled'}")
            return config
        except Exception as e:
            print(f"❌ YAML file loading failed: {e}")
            return None
    else:
        print(f"⚠️  YAML configuration file does not exist: {config_path}")
        return None


def test_validation():
    """Test configuration validation"""
    print("\n🧪 Testing configuration validation...")
    
    # Test invalid language code
    try:
        invalid_config = TaskConfig(source_lang="INVALID")
        print("❌ Should have thrown validation error")
    except Exception as e:
        print(f"✅ Language code validation working: {e}")
    
    # Test invalid resolution
    try:
        invalid_video_config = VideoDownloadConfig(resolution=999)
        print("❌ Should have thrown resolution validation error")
    except Exception as e:
        print(f"✅ Resolution validation working: {e}")
    
    # Test valid configuration
    try:
        valid_config = TaskConfig(
            source_lang="EN",
            target_lang="ZH",
            video_download=VideoDownloadConfig(resolution=720)
        )
        print("✅ Valid configuration validation passed")
    except Exception as e:
        print(f"❌ Valid configuration validation failed: {e}")


def test_dict_conversion():
    """Test dictionary conversion"""
    print("\n🧪 Testing dictionary conversion...")
    
    config = TaskConfig()
    config_dict = config.to_dict()
    
    print(f"✅ Configuration converted to dictionary successfully")
    print(f"   - Dictionary type: {type(config_dict)}")
    print(f"   - Contains keys: {list(config_dict.keys())}")
    
    # Test recreating configuration from dictionary
    try:
        new_config = validate_task_config(config_dict)
        print("✅ Recreated configuration from dictionary successfully")
    except Exception as e:
        print(f"❌ Failed to recreate configuration from dictionary: {e}")


def test_yaml_conversion():
    """Test YAML conversion"""
    print("\n🧪 Testing YAML conversion...")
    
    config = TaskConfig()
    yaml_str = config.to_yaml()
    
    print(f"✅ Configuration converted to YAML successfully")
    print(f"YAML content preview:")
    print("=" * 50)
    print(yaml_str[:500] + "..." if len(yaml_str) > 500 else yaml_str)
    print("=" * 50)


def main():
    """Main test function"""
    print("🚀 Starting configuration schema tests...")
    print("=" * 60)
    
    # Run all tests
    test_basic_config()
    test_custom_config()
    test_yaml_loading()
    test_validation()
    test_dict_conversion()
    test_yaml_conversion()
    
    print("\n" + "=" * 60)
    print("🎉 All tests completed!")


if __name__ == "__main__":
    main() 