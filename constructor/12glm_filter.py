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
# 模板
qa_pair_template = {
    '柱状图': {
        '简单': '某年销售额是多少？',
        '数学': '2018年与2019年之间的销售增长量是多少？',
        '趋势': '从2018年到2020年，销售额的变化趋势是什么？'
    },
    '折线图': {
        '简单': '2021年某月的用户数是多少？',
        '数学': '从2020年到2021年的用户增长率是多少？',
        '趋势': '折线图显示的总体趋势是什么？'
    },
    '饼图': {
        '简单': '某个类别在总销售额中占多少比例？',
        '数学': '产品D在2021年的份额与2022年的份额差多少？',
        '趋势': '饼图中显示的市场份额变化趋势是什么？'
    }
}


def read_image(image_path):
    """
    将图片编码为base64字符串
    """
    with open(image_path, 'rb') as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')
  
def find_path(base_dir, target_folder, target_file):
    pass


def combine(base_dir, target_file_pattern, combined_path, encoding='utf-8'):
    """访问各子文件夹，整合内容，并保存为txt文件
    
    参数:
        base_dir: str, 要遍历的基础目录路径。
        target_file_pattern: str, 目标文件名模式，可以是简单的文件扩展名或者正则表达式。
        combined_path: str, 结合后文件的保存路径。
        encoding: str, 文件编码，默认为 'utf-8'。
        
    返回:
        combined_path: 如果成功，则返回结合后文件的路径；如果失败，则返回None。
    """
    content = []
    try:
        # Compile the pattern if it's a regex
        pattern = re.compile(target_file_pattern) if '*' in target_file_pattern or '.' in target_file_pattern else None
        
        for root, _, files in os.walk(base_dir):
            for file in files:
                # Check if the file matches the target pattern
                if pattern and pattern.search(file) or file == target_file_pattern:
                    file_path = Path(root) / file
                    try:
                        with open(file_path, 'r', encoding=encoding) as f:
                            content.append(f.read())
                    except Exception as e:
                        print(f"无法读取文件 {file_path}: {e}")
        
        # Write combined content to the new file
        with open(combined_path, 'w', encoding=encoding) as f:
            f.write('\n'.join(content))
        
        print(f"{combined_path} 已保存")
        return combined_path
    
    except Exception as e:
        print(f"发生错误: {e}")
        return None

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

