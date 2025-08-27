import os
import sys
import argparse
from pathlib import Path
from typing import List, Optional, Union, Dict, Tuple
from datetime import datetime
import glob
import re


def format_timestamp(seconds: float) -> str:
    """
    将秒数转换为SRT时间戳格式 (HH:MM:SS,mmm)
    
    Args:
        seconds: 时间戳（秒）
        
    Returns:
        格式化的时间戳字符串
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    milliseconds = int((seconds % 1) * 1000)
    
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"


def create_srt_from_text_lines(
    text_lines: List[str], 
    output_path: str, 
    interval_seconds: float = 10.0,
    start_time: float = 0.0,
    verbose: bool = True
) -> str:
    """
    从文本行列表创建SRT文件
    
    Args:
        text_lines: 文本行列表，每行对应一个字幕条目
        output_path: 输出SRT文件路径
        interval_seconds: 每个字幕条目的时间间隔（秒）
        start_time: 起始时间（秒）
        verbose: 是否显示详细信息
        
    Returns:
        创建的SRT文件路径
        
    Raises:
        ValueError: 当文本行为空时
        OSError: 当无法写入文件时
    """
    lines = [line.strip() for line in text_lines if line.strip()]
    
    if not lines:
        raise ValueError("文本内容为空，无法创建SRT文件")
    
    # 确保输出目录存在
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, line in enumerate(lines, 1):
                # 计算时间戳
                start_seconds = start_time + (i - 1) * interval_seconds
                end_seconds = start_time + i * interval_seconds
                
                start_time_str = format_timestamp(start_seconds)
                end_time_str = format_timestamp(end_seconds)
                
                # 写入SRT格式
                f.write(f"{i}\n")
                f.write(f"{start_time_str} --> {end_time_str}\n")
                f.write(f"{line}\n\n")
        
        if verbose:
            print(f"✅ SRT文件创建成功: {output_path}")
            print(f"   包含 {len(lines)} 个字幕条目")
            print(f"   总时长: {format_timestamp(start_time + len(lines) * interval_seconds)}")
        
        return str(output_path)
        
    except OSError as e:
        raise OSError(f"无法写入SRT文件 {output_path}: {e}")


def extract_qwen_assistant_responses(content: str) -> List[str]:
    """
    从Qwen对话内容中提取assistant响应（专门针对Qwen格式优化）
    
    Args:
        content: Qwen对话内容
        
    Returns:
        提取的assistant响应列表
    """
    responses = []
    
    # 按assistant标记分割，更精确的处理
    sections = re.split(r'\nassistant\s*\n', content)
    
    # 跳过第一个部分（assistant之前的内容）
    for section in sections[1:]:
        # 找到下一个system或user标记之前的文本
        # 使用更严格的匹配模式
        next_section_match = re.search(r'\n(system|user)\s*\n', section)
        
        if next_section_match:
            # 提取assistant响应到下一个section之前的内容
            assistant_text = section[:next_section_match.start()].strip()
        else:
            # 如果没有找到下一个section，使用整个剩余部分
            assistant_text = section.strip()
        
        # 清理和验证文本
        if assistant_text:
            # 移除多余的空白字符，但保留必要的换行
            cleaned_text = re.sub(r'\n+', '\n', assistant_text).strip()
            # 替换换行为空格，创建单行字幕
            single_line_text = cleaned_text.replace('\n', ' ')
            
            # 过滤掉过短或无意义的响应
            if len(single_line_text) > 3:  # 至少3个字符
                responses.append(single_line_text)
    
    return responses


def create_srt_from_text_file(
    input_file: str, 
    output_file: Optional[str] = None,
    interval_seconds: float = 10.0,
    start_time: float = 0.0,
    extract_qwen_responses: bool = True,  # 默认开启Qwen提取
    verbose: bool = True
) -> str:
    """
    从文本文件创建SRT文件（专门针对Qwen对话文件优化）
    
    Args:
        input_file: 输入文本文件路径
        output_file: 输出SRT文件路径（可选，默认基于输入文件名）
        interval_seconds: 每个字幕条目的时间间隔（秒）
        start_time: 起始时间（秒）
        extract_qwen_responses: 是否提取Qwen对话中的assistant响应
        verbose: 是否显示详细信息
        
    Returns:
        创建的SRT文件路径
    """
    input_path = Path(input_file)
    
    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_file}")
    
    # 生成输出文件名
    if output_file is None:
        output_file = input_path.with_suffix('.srt')
    
    # 读取文本内容
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    if extract_qwen_responses:
        # 提取Qwen对话中的assistant响应
        text_lines = extract_qwen_assistant_responses(content)
        if verbose and text_lines:
            print(f"🤖 从Qwen对话中提取到 {len(text_lines)} 个assistant响应")
    else:
        # 按行分割
        text_lines = content.split('\n')
    
    return create_srt_from_text_lines(
        text_lines, 
        str(output_file), 
        interval_seconds, 
        start_time,
        verbose
    )


def batch_create_srt_from_folder(
    input_folder: str,
    output_folder: Optional[str] = None,
    interval_seconds: float = 10.0,
    start_time: float = 0.0,
    extract_qwen_responses: bool = True,  # 默认开启Qwen提取
    file_pattern: str = "*.txt",
    overwrite: bool = False
) -> Dict[str, Union[str, Exception]]:
    """
    批量从文件夹中的Qwen对话文件创建SRT文件
    
    Args:
        input_folder: 输入文件夹路径
        output_folder: 输出文件夹路径（可选，默认为输入文件夹）
        interval_seconds: 每个字幕条目的时间间隔（秒）
        start_time: 起始时间（秒）
        extract_qwen_responses: 是否提取Qwen对话中的assistant响应
        file_pattern: 文件匹配模式（默认: "*.txt"）
        overwrite: 是否覆盖已存在的SRT文件
        
    Returns:
        Dict[str, Union[str, Exception]]: 处理结果字典，键为输入文件名，值为输出路径或异常
    """
    input_path = Path(input_folder)
    
    if not input_path.exists():
        raise FileNotFoundError(f"输入文件夹不存在: {input_folder}")
    
    if not input_path.is_dir():
        raise NotADirectoryError(f"输入路径不是文件夹: {input_folder}")
    
    # 设置输出文件夹
    if output_folder is None:
        output_path = input_path
    else:
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)
    
    # 查找匹配的文件
    input_files = list(input_path.glob(file_pattern))
    
    if not input_files:
        print(f"⚠️ 在文件夹 {input_folder} 中没有找到匹配 {file_pattern} 的文件")
        return {}
    
    print(f"🔍 找到 {len(input_files)} 个文件待处理")
    print(f"📁 输入文件夹: {input_path}")
    print(f"📁 输出文件夹: {output_path}")
    print(f"⏱️ 时间间隔: {interval_seconds} 秒")
    print(f"🎬 起始时间: {start_time} 秒")
    if extract_qwen_responses:
        print("🤖 将提取Qwen对话中的assistant响应")
    print("-" * 60)
    
    results = {}
    success_count = 0
    error_count = 0
    
    for i, input_file in enumerate(input_files, 1):
        filename = input_file.name
        print(f"[{i}/{len(input_files)}] 处理文件: {filename}")
        
        try:
            # 生成输出文件路径
            output_file = output_path / input_file.with_suffix('.srt').name
            
            # 检查是否已存在
            if output_file.exists() and not overwrite:
                print(f"  ⚠️ 跳过，文件已存在: {output_file.name}")
                results[filename] = f"跳过（文件已存在）: {output_file}"
                continue
            
            # 创建SRT文件
            result_path = create_srt_from_text_file(
                str(input_file),
                str(output_file),
                interval_seconds,
                start_time,
                extract_qwen_responses,
                verbose=False  # 批量模式下关闭详细输出
            )
            
            results[filename] = result_path
            success_count += 1
            print(f"  ✅ 成功创建: {Path(result_path).name}")
            
        except Exception as e:
            results[filename] = e
            error_count += 1
            print(f"  ❌ 失败: {e}")
    
    # 输出总结
    print("\n" + "=" * 60)
    print("📊 批量处理完成")
    print("=" * 60)
    print(f"✅ 成功: {success_count} 个文件")
    print(f"❌ 失败: {error_count} 个文件")
    print(f"📁 输出目录: {output_path}")
    
    if error_count > 0:
        print("\n❌ 失败的文件:")
        for filename, result in results.items():
            if isinstance(result, Exception):
                print(f"  - {filename}: {result}")
    
    return results


# 保留旧的函数名以兼容性
def extract_ai_assistant_responses(content: str) -> List[str]:
    """兼容性函数，调用新的Qwen专用函数"""
    return extract_qwen_assistant_responses(content)


def create_srt_with_custom_timing(
    text_timing_pairs: List[tuple], 
    output_path: str
) -> str:
    """
    使用自定义时间创建SRT文件
    
    Args:
        text_timing_pairs: (文本, 开始时间, 结束时间) 元组列表
        output_path: 输出SRT文件路径
        
    Returns:
        创建的SRT文件路径
    """
    if not text_timing_pairs:
        raise ValueError("时间对列表为空")
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, (text, start_time, end_time) in enumerate(text_timing_pairs, 1):
            start_time_str = format_timestamp(start_time)
            end_time_str = format_timestamp(end_time)
            
            f.write(f"{i}\n")
            f.write(f"{start_time_str} --> {end_time_str}\n")
            f.write(f"{text.strip()}\n\n")
    
    print(f"✅ 自定义时间SRT文件创建成功: {output_path}")
    return str(output_path)


def interactive_srt_creator() -> str:
    """
    交互式SRT文件创建器
    
    Returns:
        创建的SRT文件路径
    """
    print("🎬 交互式SRT文件创建器")
    print("=" * 50)
    
    # 获取输出文件名
    while True:
        output_file = input("请输入输出SRT文件名（例：output.srt）: ").strip()
        if output_file:
            if not output_file.endswith('.srt'):
                output_file += '.srt'
            break
        print("❌ 请输入有效的文件名")
    
    # 获取时间间隔
    while True:
        try:
            interval = float(input("请输入每个字幕的时间间隔（秒，默认10）: ") or "10")
            if interval > 0:
                break
            print("❌ 时间间隔必须大于0")
        except ValueError:
            print("❌ 请输入有效的数字")
    
    # 获取起始时间
    while True:
        try:
            start_time = float(input("请输入起始时间（秒，默认0）: ") or "0")
            if start_time >= 0:
                break
            print("❌ 起始时间不能为负数")
        except ValueError:
            print("❌ 请输入有效的数字")
    
    # 收集文本行
    print("\n请输入字幕文本（每行一个字幕，输入空行结束）:")
    text_lines = []
    line_number = 1
    
    while True:
        line = input(f"第{line_number}行: ")
        if not line.strip():
            break
        text_lines.append(line)
        line_number += 1
    
    if not text_lines:
        print("❌ 没有输入任何字幕文本")
        return ""
    
    # 创建SRT文件
    try:
        return create_srt_from_text_lines(text_lines, output_file, interval, start_time)
    except Exception as e:
        print(f"❌ 创建SRT文件失败: {e}")
        return ""


def main():
    """主函数 - 命令行接口"""
    parser = argparse.ArgumentParser(
        description="Qwen对话SRT文件创建工具 - 从Qwen对话文件创建SRT字幕文件",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 单文件：从Qwen对话文件创建SRT
  python qwen_srt_extractor.py -i qwen_dialogue.txt -o output.srt
  
  # 批量处理：处理文件夹中的所有Qwen对话文件
  python qwen_srt_extractor.py --batch-folder /path/to/qwen/dialogues
  
  # 批量处理到指定输出文件夹
  python qwen_srt_extractor.py --batch-folder input_folder --output-folder output_folder
  
  # 设置自定义时间间隔
  python qwen_srt_extractor.py -i qwen_dialogue.txt --interval 5
  
  # 禁用Qwen响应提取（按行处理）
  python qwen_srt_extractor.py -i input.txt --no-extract-qwen
  
  # 交互式创建
  python qwen_srt_extractor.py --interactive
        """
    )
    
    # 输入选项
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('-i', '--input', 
                       help='输入Qwen对话文件路径（单文件模式）')
    input_group.add_argument('--batch-folder', 
                       help='输入文件夹路径（批量处理模式）')
    input_group.add_argument('--interactive', action='store_true',
                       help='启动交互式创建模式')
    
    # 输出选项
    parser.add_argument('-o', '--output', 
                       help='输出SRT文件路径（单文件模式）')
    parser.add_argument('--output-folder', 
                       help='输出文件夹路径（批量模式，可选）')
    
    # 处理选项
    parser.add_argument('--interval', type=float, default=8.0,
                       help='每个字幕条目的时间间隔（秒，默认10）')
    parser.add_argument('--start-time', type=float, default=0.0,
                       help='起始时间（秒，默认0）')
    parser.add_argument('--no-extract-qwen', action='store_true',
                       help='禁用Qwen响应提取，按行处理文本')
    
    # 批量处理选项
    parser.add_argument('--pattern', default='*.txt',
                       help='文件匹配模式（批量模式，默认: "*.txt"）')
    parser.add_argument('--overwrite', action='store_true',
                       help='覆盖已存在的SRT文件（批量模式）')
    
    args = parser.parse_args()
    
    # 确定是否提取Qwen响应
    extract_qwen = not args.no_extract_qwen
    
    try:
        if args.interactive:
            # 交互式模式
            result = interactive_srt_creator()
            if result:
                print(f"\n🎉 SRT文件创建完成: {result}")
                
        elif args.batch_folder:
            # 批量处理模式
            results = batch_create_srt_from_folder(
                input_folder=args.batch_folder,
                output_folder=args.output_folder,
                interval_seconds=args.interval,
                start_time=args.start_time,
                extract_qwen_responses=extract_qwen,
                file_pattern=args.pattern,
                overwrite=args.overwrite
            )
            
            success_files = [f for f, r in results.items() if isinstance(r, str) and not r.startswith("跳过")]
            if success_files:
                print(f"\n🎉 批量处理完成，成功创建 {len(success_files)} 个SRT文件")
            
        elif args.input:
            # 单文件模式
            result = create_srt_from_text_file(
                args.input,
                args.output,
                args.interval,
                args.start_time,
                extract_qwen
            )
            print(f"\n🎉 SRT文件创建完成: {result}")
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 