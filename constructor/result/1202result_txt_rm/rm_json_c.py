import os
import shutil
import logging
import re
import glob
from typing import List, Optional
from pathlib import Path

class FileProcessor:
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self._setup_logging()
    
    def _setup_logging(self) -> None:
        """配置日志记录"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler("file_processor.log"),
                logging.StreamHandler()
            ]
        )
        
    def process_subfolders(self) -> None:
        """处理所有子文件夹中的 glm_result 目录"""
        try:
            if not self.base_path.is_dir():
                logging.error(f"Invalid directory path: {self.base_path}")
                return
                
            for subfolder in self.base_path.iterdir():
                if not subfolder.is_dir():
                    logging.warning(f"Skipping non-directory: {subfolder}")
                    continue
                    
                glm_result_path = subfolder / "glm_result"
                if not glm_result_path.is_dir():
                    logging.warning(f"glm_result folder not found in {subfolder}")
                    continue
                    
                self._process_glm_folder(glm_result_path)
                    
        except Exception as e:
            logging.error(f"Error processing folders: {e}")
    
    def _process_glm_folder(self, glm_path: Path) -> None:
        """处理单个 glm_result 文件夹"""
        for dir_name in ['cur', 'next', 'prev']:
            response_path = glm_path / dir_name / "response.json"
            
            try:
                if response_path.is_file():
                    self._handle_response_file(response_path)
                elif response_path.is_dir():
                    self._handle_response_directory(response_path)
            except Exception as e:
                logging.error(f"Error processing {response_path}: {e}")
    
    def _handle_response_file(self, file_path: Path) -> None:
        """处理 response.json 文件"""
        new_path = file_path.parent / file_path.name
        shutil.move(str(file_path), str(new_path))
        logging.info(f"Moved file: {file_path} -> {new_path}")
        os.makedirs(str(file_path), exist_ok=True)
    
    def _handle_response_directory(self, dir_path: Path) -> None:
        """处理 response.json 目录"""
        for file_path in dir_path.iterdir():
            if file_path.is_file():
                new_path = dir_path.parent / file_path.name
                shutil.move(str(file_path), str(new_path))
                logging.info(f"Moved file: {file_path} -> {new_path}")
        
        if not any(dir_path.iterdir()):
            dir_path.rmdir()
            logging.info(f"Removed empty directory: {dir_path}")

    def rename_files(self, pattern: str, file_extension: str) -> None:
        """重命名符合特定模式的文件"""
        pattern_regex = re.compile(rf"(\[.*\]_0\.jpg)\.{file_extension}$")
        search_pattern = f'fortune-or-fiction-final-v3_*/glm_result/sub/*_0.jpg.{file_extension}'
        
        file_paths = list(self.base_path.glob(search_pattern))
        if not file_paths:
            print(f"No files found matching pattern *.{file_extension}")
            return
            
        for file_path in file_paths:
            match = pattern_regex.match(file_path.name)
            if match:
                new_name = match.group(1)
                new_path = file_path.parent / new_name
                file_path.rename(new_path)
                print(f"Renamed: {file_path} -> {new_path}")
            else:
                print(f"File does not match naming pattern: {file_path}")

def main():
    base_path = "./week_in_charts/global-materials-perspective-2024"
    processor = FileProcessor(base_path)
    
    # 处理文件夹结构
    processor.process_subfolders()
    
    # 重命名文件
    processor.rename_files(pattern=r"(\[.*\]_0\.jpg)\.json$", file_extension="json")
    processor.rename_files(pattern=r"(\[.*\]_0\.jpg)\.txt$", file_extension="txt")

if __name__ == "__main__":
    main()