import os
from typing import Callable


def judge_path():
  """判断文件是否在PaddleOCR目录下"""
  current_folder = os.getcwd()
  if not current_folder.endswith("PaddleOCR"):
    print("Please run this script in the PaddleOCR folder")
    exit(1)


def prehook(hooks: list):
  def wrapper(fn: Callable):
    for hook in hooks:
      hook()
    return fn

  return wrapper

def join_cmdlist(cmd_list: list):
  return " ".join(cmd_list)


@prehook(hooks=[judge_path])
def eval_date():
  eval_file = "tools/eval.py"
  config_file = "configs/rec/PP-OCRv3/ch_PP-OCRv3_rec_distillation.yml"
  ckpt_file = "ckpt/ch_PP-OCRv3_rec_train/best_accuracy"
  data_dir = "./data"
  label_file_list = ["./data/val.list"]

  cmd = join_cmdlist([
    f"python {eval_file} -c {config_file}",
    f"-o Global.checkpoints={ckpt_file}",
    f"Eval.dataset.data_dir={data_dir}",
    f"Eval.dataset.label_file_list={label_file_list}",
  ])
  print("running: ", cmd)
  os.system(cmd)


def eval_meter():
  cmd = join_cmdlist([
    "python tools/eval.py",
    "-c configs/rec/PP-OCRv3/ch_PP-OCRv3_rec_distillation.yml",
    "-o Global.checkpoints=ckpt/ch_PP-OCRv3_rec_train/best_accuracy",
    "Eval.dataset.data_dir=./train_data",
    "Eval.dataset.label_file_list=[./train_data/rec/val.list]",
  ])
  print("running: ", cmd)
  os.system(cmd)

if __name__ == "__main__":
  eval_date()
