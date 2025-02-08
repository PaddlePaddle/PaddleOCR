"""transform model to deployment model"""

parent = "atrain_script"

# -c 后面设置训练算法的yml配置文件
# -o 配置可选参数
# Global.pretrained_model 参数设置待转换的训练模型地址，不用添加文件后缀 .pdmodel，.pdopt或.pdparams。
# Global.save_inference_dir参数设置转换的模型将保存的地址。
cmd_list = [
  "python3 tools/export_model.py",
  "-c configs/rec/ch_ppocr_v2.0/rec_chinese_lite_train_v2.0.yml",
  f"-o Global.pretrained_model=./{parent}/ch_lite/ch_ppocr_mobile_v2.0_rec_train/best_accuracy",
  f"Global.save_inference_dir=./{parent}/inference/rec_crnn/",
]

cmd = " ".join(cmd_list)
print(cmd)
