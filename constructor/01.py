import os
from pathlib import Path

def combine_rsp_txt(base_dir, combined_path, encoding='utf-8'):
    """访问各子文件夹，整合内容，并保存为txt文件
    
    参数:
        base_dir: str, 要遍历的基础目录路径。
        combined_path: str, 结合后文件的保存路径。
        encoding: str, 文件编码，默认为 'utf-8'。
        
    返回:
        combined_path: 如果成功，则返回结合后文件的路径；如果失败，则返回None。
    """
    content = []
    target_folder_name = 'cur'
    target_file_name = 'rsp.txt'

    try:
        for root, dirs, files in os.walk(base_dir):
            # Check if the current folder is named 'cur'    
            current_path = Path(root)
            if current_path.name == target_folder_name and target_file_name in files:
                file_path = current_path / target_file_name
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        content.append(f.read())
                except Exception as e:
                    print(f"无法读取文件 {file_path}: {e}")
        
        # Write combined content to the new file
        if content:  # Only write if there's something to write
            with open(combined_path, 'w', encoding=encoding) as f:
                f.write('\n'.join(content))
            print(f"{combined_path} 已保存")
        else:
            print("没有找到任何符合条件的 rsp.txt 文件")
        
        return combined_path if content else None
    
    except Exception as e:
        print(f"发生错误: {e}")
        return None


if __name__ == '__main__':
    rela_path = "./result/1202result_glm/test/americas-small-businesses-time-to-think-big/"
    combined_output_path = "./result/1202result_glm/test/output/combined_rsp.txt"
    
    # 确保输出目录存在
    output_dir = Path(combined_output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    result = combine_rsp_txt(rela_path, combined_output_path)
    if result:
        print(f"合并完成，结果保存在: {result}")
    else:
        print("未找到符合条件的文件或发生错误")