import os
from typing import Callable


def judge_path():
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


@prehook(hooks=[judge_path])
def eval_date():
  eval_file = "tools/eval.py"
  config_file = "configs/rec/PP-OCRv3/ch_PP-OCRv3_rec_distillation.yml"
  ckpt_file = "ckpt/ch_PP-OCRv3_rec_train/best_accuracy"
  data_dir = "./data"
  label_file_list = ["./data/val.list"]

  cmd_list = [
    f"python {eval_file} -c {config_file}",
    f"-o Global.checkpoints={ckpt_file}",
    f"Eval.dataset.data_dir={data_dir}",
    f"Eval.dataset.label_file_list={label_file_list}",
  ]

  cmd = " ".join(cmd_list)
  print("running: ", cmd)
  os.system(cmd)


if __name__ == "__main__":
  eval_date()
