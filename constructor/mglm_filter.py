import json
import base64
from zhipuai import ZhipuAI
import os
import re
from pathlib import Path
# 

# 图片路径
img_path = "./result/try/americas-small-businesses-time-to-think-big/americas-small-businesses-time-to-think-big_0/americas-small-businesses-time-to-think-big_0_current_page.png"

# 图片描述路径
img_dsp = "moe/PaddleOCR_m/constructor/result/glm_txt_result/americas-small-businesses-time-to-think-big/americas-small-businesses-time-to-think-big_0/glm_result/sub/[0, 8, 611, 791]_0.jpg.json/rsp.txt"

# 用户提示
prompt0 = f"这是图片，根据图片和描述{img_dsp}判断是否图片是否为统计图表，并输出True或False。"



# 图表类型列表
accepted_chart_type = ['饼图', '柱状图', '折线图']



def read_image(image_path):
    """
    将图片编码为base64字符串
    """
    with open(image_path, 'rb') as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')
  
def find_path(base_dir, target_folder, target_file):
    """继承自find_path类，需要重构"""
    pass


def combine(base_dir, target_file_pattern, combined_path, encoding='utf-8'):
    """"""
    pass
    
    

# 示例调用
rela_path = "./constructor/result/1202result_glm/test/americas-small-businesses-time-to-think-big"
target_file_path = "rsp.txt"
combined_path = "moe/PaddleOCR_m/constructor/result/1202result_glm/test/output"
combine(rela_path, target_file_path, combined_path)

    

def glm(api_key, prompt, img_path, response):
    # 初始化ZhipuAI客户端，并设置API Key
    client = ZhipuAI(api_key=api_key)  # 请替换为您自己的APIKey
    img_base = read_image(img_path)

    # 调用模型生成回复
    response = client.chat.completions.create(
        model="glm-4v-plus",  # 模型名称
        messages=[
          {
            "role": "user",
            "content": [
              {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{img_base}"
                }
              },
              {
                "type": "text",
                "text": prompt
              }
            ]
          }
        ]
    )
    return response.choices[0].message.content

def glm_filter(api_key, prompt, response):
    """使用glm进行图片过滤，获得统计图表并保存"""


    save_path = os.makedirs(save_path, exist_ok= True)
    if response.lower() == "true":
        print("图片为统计图表")
        # 将图片保存到save_path
        save_path = os.path.join(save_path,"chart_img")
    else:
        print("图片不是统计图表, 跳过")

def glm_bck_analysis(api_key, prompt,txt_path, save_path):
    pass


    
def chart_judge(response, prompt, img_path, txt_path, result:bool):
    """使用glm进行图片过滤，判断是否为统计图表，返回bool值"""
    pass


    



# # 打印响应结果
# print(f"Response: {json.dumps(response, ensure_ascii=False, indent=2)}")
# print("-----------------------------------")

# # 打印选择部分的结果
# print(f"response.choices[0]: {json.dumps(response.choices[0], ensure_ascii=False, indent=2)}")
# print("-----------------------------------")

# # 打印消息内容
# print(f"response.choices[0].message.content: {response.choices[0].message.content}")
# print("-----------------------------------")

# # 统计并打印消耗的token数
# completion_tokens = response.usage.completion_tokens if hasattr(response.usage, 'completion_tokens') else 0
# prompt_tokens = response.usage.prompt_tokens if hasattr(response.usage, 'prompt_tokens') else 0
# total_tokens = completion_tokens + prompt_tokens
# print(f"本次调用消耗的总token数为：{total_tokens}")
# print(f"其中，完成部分消耗的token数为：{completion_tokens}")
# print(f"提示部分消耗的token数为：{prompt_tokens}")

