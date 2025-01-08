import os
import re
from zhipuai import ZhipuAI,ZhipuAIError
import base64

# 图片路径 moe/PaddleOCR_m/constructor/result/1202result_txt/week_in_charts/fortune-or-fiction-final-v3/fortune-or-fiction-final-v3_0/fortune-or-fiction-final-v3_0/[0, 387, 602, 718]_0.jpg
# 图片描述路径  moe/PaddleOCR_m/constructor/result/1202result_txt/week_in_charts/fortune-or-fiction-final-v3/fortune-or-fiction-final-v3_0/glm_result/sub/[0, 387, 602, 718]_0.jpg.json/rsp.txt

# img_path="./result/1202result_txt/week_in_charts/fortune-or-fiction-final-v3/fortune-or-fiction-final-v3_0/fortune-or-fiction-final-v3_0/[0, 387, 602, 718]_0.jpg"
# # img_description_path="./result/1202result_txt/week_in_charts/fortune-or-fiction-final-v3/fortune-or-fiction-final-v3_0/glm_result/sub/[0, 387, 602, 718]_0.jpg.json/rsp.txt"
# img_description_path="./result/1202result_txt/week_in_charts/fortune-or-fiction-final-v3/fortune-or-fiction-final-v3_0/glm_result/sub/[0, 387, 602, 718]_0.jpg.json/rsp.txt"

img_path='./result/1202result_txt/week_in_charts/fortune-or-fiction-final-v3/fortune-or-fiction-final-v3_3/fortune-or-fiction-final-v3_3/[136, 470, 552, 684]_0.jpg'
img_description_path='./result/1202result_txt/week_in_charts/fortune-or-fiction-final-v3/fortune-or-fiction-final-v3_3/glm_result/sub/[136, 470, 552, 684]_0.jpg.json/rsp.txt'

"""预设内容begin"""
# API key配置
# key2 = '7916aaef6c2af99dc9593c64701f8356.YWbNyp2aRGZQ3Z2U'
key2 ='59c47aa0217fb8a72944a2a0c6b9d2eb.B7EG63vlMlr88ejp'


accepted_chart_type = ['饼图', '柱状图', '折线图','雷达图','散点图'] #接受的图表类型


"""预设内容end"""

# 访问图片描述路径，读取txt内容并存储
def count_file_info(file_path):
    """整理文件信息，确保不超过模型的token限制，限制会导致信息的不完整"""
    try: 
        line_count=0
        word_count=0
        char_count=0
        content_lines=[]

        with open(file_path, 'r',encoding='utf-8') as file:
            for line in file:
                line_count +=1
                words = line.split()
                word_count  += len(words)
                char_count += len(line.strip('\n')) #忽略换行
                content_lines.append(line) #将每一行内容添加到列表中


        print(f"文件{file_path}的行数是:{line_count}")
        print(f"文件{file_path}的字数是:{word_count}")
        print(f"文件{file_path}的字符数(去除换行符）是:{char_count}")

        return ''.join(content_lines) # 将列表转换为字符串并返回
    except FileNotFoundError:
        print(f"文件{file_path}不存在")
        return None

def read_txt_file(file_path):
    """读取txt文件内容，"""
    content=count_file_info(file_path)
    if content is not None and content.strip():
        print("文件内容的前50个字符为：",content[:50])
        return content
    else:
        print(f"文件{file_path}不存在或{content}为空")
        return None

# 将图片处理为base64编码
def read_image(image_path):
    """
    将图片编码为base64格式输入给大模型
    """
    with open(image_path, 'rb') as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')


# 拼接完整的prompt
# sub_description=read_txt_file(img_description_path)
# combined_input=f"""这是图片描述{sub_description},请结合图片描述仔细观察这张图片。
# 特别注意任何可能的统计图表元素，例如标题、轴标签、数据系列等。
# 如果是统计图表，返回True，否则返回False。"""
# 拼接完整的prompt
def create_prompt(sub_description):
    """拼接prompt内容"""
    if sub_description:
        prompt = f"""
        这是图片描述{sub_description},请结合图片描述仔细观察这张图片。
        特别注意任何可能的统计图表元素，例如标题、轴标签、数据系列等。
        如果是统计图表，返回True，否则返回False。
        """
        print("生成的prompt内容: ", prompt)  # 打印生成的prompt内容
        return prompt
    else:
        print("图片描述为空，无法生成prompt")
        return None

# 初始化客户端
def glm(img_path, api_key,prompt):
    # 处理图片
    img_base=read_image(img_path)

    # 初始化客户端，设定apikey
    client = ZhipuAI(api_key=api_key)

    # 构建回复
    try:
        response = client.chat.completions.create(
            model="glm-4v-plus",  # 填写需要调用的模型名称
            messages=[
            {
                "role": "user",
                "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": img_base
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
        real_response=response.choices[0].message.content
        return real_response
    except ZhipuAIError as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    #读取图片描述
    sub_description=read_txt_file(img_description_path)

    if sub_description:

        # 生成拼接的prompt
        prompt=create_prompt(sub_description)

                # 检查prompt是否有效
        if prompt:
            # 调用模型并获取结果
            response = glm(img_path, key2, prompt)

            if response:
                print("模型响应: ", response)
            else:
                print("未能成功获取模型响应")
        else:
            print("生成的prompt无效，无法调用模型")
    else:
        print("没有读取到有效的图片描述")