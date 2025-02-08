import os
import pathlib

cmd_list = [
  # 下载超轻量中文识别模型：
  "wget https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_rec_infer.tar",
  "tar xf ch_ppocr_mobile_v2.0_rec_infer.tar",
  'python3 tools/infer/predict_rec.py --image_dir="./doc/imgs_words/ch/word_4.jpg"'
  + ' --rec_image_shape="3,32,320"'  # critical!
  + ' --rec_model_dir="ch_ppocr_mobile_v2.0_rec_infer"',
]

if os.path.exists("ch_ppocr_mobile_v2.0_rec_infer"):
  print("模型已存在, 直接推理")
  os.system(cmd_list[-1])
else:
  print("模型不存在，开始下载")
  cmd = " && ".join(cmd_list)
  print(cmd)
  os.system(cmd)
