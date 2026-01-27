#!/usr/bin/env python3
"""
将简化格式的PKL（convert_pkl_to_csv.py读取的格式）转换为完整格式的PKL（convert_npz_to_pkl.py生成的格式）

简化格式PKL → CSV → 通过IsaacSim生成完整PKL

使用方法:
    python convert_simple_pkl_to_full_pkl.py --input_file simple.pkl --output_file full.pkl [--fps 30] [--headless]
"""

import argparse
import os
import sys
import subprocess
import tempfile
import shutil

def convert_simple_pkl_to_full_pkl(input_pkl_file, output_pkl_file, fps=None, headless=True):
    """
    将简化格式的PKL转换为完整格式的PKL
    
    步骤：
    1. 简化PKL → CSV
    2. CSV → 完整PKL (通过IsaacSim)
    
    Args:
        input_pkl_file: 输入的简化格式PKL文件
        output_pkl_file: 输出的完整格式PKL文件
        fps: 输出FPS（如果为None，则从输入PKL读取）
        headless: 是否使用headless模式运行IsaacSim
    """
    print(f"\n{'='*80}")
    print(f"将简化格式PKL转换为完整格式PKL")
    print(f"{'='*80}")
    print(f"输入文件: {input_pkl_file}")
    print(f"输出文件: {output_pkl_file}")
    
    # 检查输入文件是否存在
    if not os.path.exists(input_pkl_file):
        print(f"❌ 错误: 输入文件 {input_pkl_file} 不存在")
        sys.exit(1)
    
    # 创建临时目录
    temp_dir = tempfile.mkdtemp(prefix="pkl_convert_")
    temp_csv = os.path.join(temp_dir, "temp_motion.csv")
    
    try:
        # 步骤1: 简化PKL → CSV
        print(f"\n步骤1: 将PKL转换为CSV...")
        print(f"临时CSV文件: {temp_csv}")
        
        # 调用 convert_pkl_to_csv.py
        cmd1 = [
            sys.executable,
            "convert_pkl_to_csv.py",
            "--input_file", input_pkl_file,
            "--output_file", temp_csv
        ]
        
        print(f"执行命令: {' '.join(cmd1)}")
        result1 = subprocess.run(cmd1, capture_output=True, text=True, cwd=os.path.dirname(__file__))
        
        if result1.returncode != 0:
            print(f"❌ 错误: PKL转CSV失败")
            print(f"stdout: {result1.stdout}")
            print(f"stderr: {result1.stderr}")
            sys.exit(1)
        
        print(f"✓ PKL已转换为CSV")
        
        # 从CSV或输入PKL获取FPS
        if fps is None:
            # 尝试从输入PKL读取FPS
            import pickle
            with open(input_pkl_file, 'rb') as f:
                pkl_data = pickle.load(f)
                fps = float(pkl_data['fps'])
            print(f"从输入PKL读取FPS: {fps}")
        else:
            print(f"使用指定的FPS: {fps}")
        
        # 步骤2: CSV → 完整PKL (通过IsaacSim)
        print(f"\n步骤2: 通过IsaacSim生成完整PKL...")
        
        cmd2 = [
            sys.executable,
            "scripts/csv_to_pkl.py",
            "--input_file", temp_csv,
            "--input_fps", str(int(fps)),
            "--output_fps", str(int(fps)),
            "--output_file", output_pkl_file
        ]
        
        if headless:
            cmd2.append("--headless")
        
        print(f"执行命令: {' '.join(cmd2)}")
        print(f"\n注意: 这将启动IsaacSim，可能需要一些时间...")
        
        result2 = subprocess.run(cmd2, capture_output=False, cwd=os.path.dirname(__file__))
        
        if result2.returncode != 0:
            print(f"❌ 错误: CSV转PKL失败")
            sys.exit(1)
        
        print(f"\n✓ 完整格式PKL已生成")
        print(f"输出文件: {output_pkl_file}")
        
        # 验证输出文件
        if os.path.exists(output_pkl_file):
            import pickle
            with open(output_pkl_file, 'rb') as f:
                full_pkl_data = pickle.load(f)
            
            print(f"\n=== 输出文件信息 ===")
            print(f"FPS: {full_pkl_data['fps']}")
            print(f"帧数: {full_pkl_data['root_pos'].shape[0]}")
            print(f"root_pos shape: {full_pkl_data['root_pos'].shape}")
            print(f"root_rot shape: {full_pkl_data['root_rot'].shape}")
            print(f"dof_pos shape: {full_pkl_data['dof_pos'].shape}")
            
            if 'local_body_pos' in full_pkl_data:
                print(f"local_body_pos shape: {full_pkl_data['local_body_pos'].shape}")
            else:
                print(f"⚠️  警告: 输出文件没有 local_body_pos 字段")
            
            if 'link_body_list' in full_pkl_data:
                print(f"link_body_list 长度: {len(full_pkl_data['link_body_list'])}")
            else:
                print(f"⚠️  警告: 输出文件没有 link_body_list 字段")
            
            if 'joint_names' in full_pkl_data:
                print(f"joint_names 长度: {len(full_pkl_data['joint_names'])}")
                print(f"✓ 输出文件包含 joint_names 字段（完整格式）")
            else:
                print(f"⚠️  警告: 输出文件没有 joint_names 字段")
        else:
            print(f"⚠️  警告: 无法验证输出文件")
    
    finally:
        # 清理临时文件
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print(f"\n已清理临时文件: {temp_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="将简化格式的PKL转换为完整格式的PKL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本用法
  python convert_simple_pkl_to_full_pkl.py --input_file simple.pkl --output_file full.pkl
  
  # 指定FPS
  python convert_simple_pkl_to_full_pkl.py --input_file simple.pkl --output_file full.pkl --fps 30
  
  # 不使用headless模式（显示IsaacSim窗口）
  python convert_simple_pkl_to_full_pkl.py --input_file simple.pkl --output_file full.pkl --no-headless
        """
    )
    
    parser.add_argument("--input_file", type=str, required=True,
                       help="输入的简化格式PKL文件路径")
    parser.add_argument("--output_file", type=str, required=True,
                       help="输出的完整格式PKL文件路径")
    parser.add_argument("--fps", type=float, default=None,
                       help="输出FPS（如果未指定，则从输入PKL读取）")
    parser.add_argument("--headless", action="store_true", default=True,
                       help="使用headless模式运行IsaacSim（默认）")
    parser.add_argument("--no-headless", dest="headless", action="store_false",
                       help="不使用headless模式（显示IsaacSim窗口）")
    
    args = parser.parse_args()
    
    convert_simple_pkl_to_full_pkl(
        args.input_file,
        args.output_file,
        fps=args.fps,
        headless=args.headless
    )
