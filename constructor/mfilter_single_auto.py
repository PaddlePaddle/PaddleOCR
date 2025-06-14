
"""自动版筛选"""
import os
import re
from zhipuai import ZhipuAI,ZhipuAIError
import base64
import shutil

# # 图片路径 moe/PaddleOCR_m/constructor/result/1202result_txt/week_in_charts/fortune-or-fiction-final-v3/fortune-or-fiction-final-v3_0/fortune-or-fiction-final-v3_0/[0, 387, 602, 718]_0.jpg
# # 图片描述路径  moe/PaddleOCR_m/construntctor/result/1202result_txt/week_in_charts/fortune-or-fiction-final-v3/fortune-or-fiction-final-v3_0/glm_result/sub/[0, 387, 602, 718]_0.jpg.json/rsp.txt

# img_path='/home/ubuntu/moe/PaddleOCR_m/constructor/mini/source/gbar-2024-attaining-escape-velocity-f/gbar-2024-attaining-escape-velocity-f_7/gbar-2024-attaining-escape-velocity-f_7/[145, 96, 578, 413]_0.jpg'
# img_description_path='/home/ubuntu/moe/PaddleOCR_m/constructor/mini/source/gbar-2024-attaining-escape-velocity-f/gbar-2024-attaining-escape-velocity-f_7/gbar-2024-attaining-escape-velocity-f_7/glm_result/sub/[145, 96, 578, 413]_0.jpg.json/rsp.txt'
txt_path=''
# 读txt文件，img_path为奇数行内容，img_description_path为偶数行内容，输出


# img_path='/home/ubuntu/moe/PaddleOCR_m/constructor/mini/source/deloitte-cn-lshc-yearbook-of-patient-assistance-2021-zh-211029/deloitte-cn-lshc-yearbook-of-patient-assistance-2021-zh-211029_26/deloitte-cn-lshc-yearbook-of-patient-assistance-2021-zh-211029_26/[17, 513, 277, 707]_0.jpg'
# img_description_path='/home/ubuntu/moe/PaddleOCR_m/constructor/mini/source/deloitte-cn-lshc-yearbook-of-patient-assistance-2021-zh-211029/deloitte-cn-lshc-yearbook-of-patient-assistance-2021-zh-211029_26/glm_result/sub/[17, 513, 277, 707]_0.jpg.txt/rsp.txt'
def read_txt_odd(txt_path):
    """
    读取文本文件中的下一个奇数行(img_path)
    
    Args:
        txt_path (str): 文本文件路径
    
    Returns:
        str: 奇数行的img_path值，如果没有更多行则返回None
    """
    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            # 跳过之前读取过的行
            if not hasattr(read_txt_odd, 'current_line'):
                read_txt_odd.current_line = 1
            else:
                for _ in range(read_txt_odd.current_line - 1):
                    next(f, None)
            
            # 读取当前奇数行
            while True:
                line = next(f, None)
                if line is None:  # 文件结束
                    read_txt_odd.current_line = 1  # 重置计数
                    return None
                
                line = line.strip()
                if not line or line.startswith('#'):  # 跳过空行和注释行
                    read_txt_odd.current_line += 1
                    continue
                
                if read_txt_odd.current_line % 2 == 1:  # 奇数行
                    read_txt_odd.current_line += 1
                    if line.startswith('img_path='):
                        return line.split('=')[1].strip().strip("'").strip('"')
                read_txt_odd.current_line += 1
                
    except FileNotFoundError:
        print(f"文件 {txt_path} 不存在")
        return None
    except Exception as e:
        print(f"读取文件时发生错误: {str(e)}")
        return None

def read_txt_even(txt_path):
    """
    读取文本文件中的下一个偶数行(img_description_path)
    
    Args:
        txt_path (str): 文本文件路径
    
    Returns:
        str: 偶数行的img_description_path值，如果没有更多行则返回None
    """
    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            # 跳过之前读取过的行
            if not hasattr(read_txt_even, 'current_line'):
                read_txt_even.current_line = 1
            else:
                for _ in range(read_txt_even.current_line - 1):
                    next(f, None)
            
            # 读取当前偶数行
            while True:
                line = next(f, None)
                if line is None:  # 文件结束
                    read_txt_even.current_line = 1  # 重置计数
                    return None
                
                line = line.strip()
                if not line or line.startswith('#'):  # 跳过空行和注释行
                    read_txt_even.current_line += 1
                    continue
                
                if read_txt_even.current_line % 2 == 0:  # 偶数行
                    read_txt_even.current_line += 1
                    if line.startswith('img_description_path='):
                        return line.split('=')[1].strip().strip("'").strip('"')
                read_txt_even.current_line += 1
                
    except FileNotFoundError:
        print(f"文件 {txt_path} 不存在")
        return None
    except Exception as e:
        print(f"读取文件时发生错误: {str(e)}")
        return None



# img_path="/home/ubuntu/moe/PaddleOCR_m/constructor/mini/source/gbar-2024-attaining-escape-velocity-f/gbar-2024-attaining-escape-velocity-f_0/gbar-2024-attaining-escape-velocity-f_0/[21, 170, 605, 792]_0.jpg"
# img_description_path="/home/ubuntu/moe/PaddleOCR_m/constructor/mini/source/gbar-2024-attaining-escape-velocity-f/gbar-2024-attaining-escape-velocity-f_0/glm_result/sub/[21, 170, 605, 792]_0.jpg.json/rsp.txt"
"""预设内容begin"""
"""涉及一些基础的配置，如api（后期设置到环境变量中）和接受的图表类型列表"""
# API key配置
# key2 = '7916aaef6c2af99dc9593c64701f8356.YWbNyp2aRGZQ3Z2U'
key2 ='7916aaef6c2af99dc9593c64701f8356.YWbNyp2aRGZQ3Z2U' #filter_key

accepted_chart_type = ['饼图', '柱状图', '折线图','雷达图','散点图'] #接受的图表类型
#该列表用于find_sub_by_txt方法，待实现，暂时用不上

"""预设内容end"""

# 读取路径文件，假设txt文件每两行分别是 img_path 和 img_description_path
def read_paths_from_txt(txt_file):
    """
    从txt文件读取路径，假设每两行分别是 img_path 和 img_description_path。
    返回一个列表，其中每个元素是 (img_path, img_description_path) 的元组。
    """
    paths = []
    try:
        with open(txt_file, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            for i in range(0, len(lines), 2):
                img_path = lines[i].strip()  # 获取图片路径
                img_description_path = lines[i + 1].strip()  # 获取图片描述路径
                paths.append((img_path, img_description_path))
        return paths
    except FileNotFoundError:
        print(f"文件 {txt_file} 不存在")
        return []

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
        print("-----------------------------------------------------")

        return ''.join(content_lines) # 将列表转换为字符串并返回
    except FileNotFoundError:
        print(f"文件{file_path}不存在")
        print("--------------------error,非正常退出---------------------------")
        return None

def read_txt_file(file_path):
    """读取txt文件内容，"""
    content=count_file_info(file_path)
    if content is not None and content.strip():
        print("文件内容的前50个字符为：",content[:50])
        print("-----------------------------------------------------")
        return content
    else:
        print(f"文件{file_path}不存在或{content}为空")
        print("--------------------error，非正常退出----------------------------")
        return None

# 将图片处理为base64编码
def read_image(image_path):
    """
    将图片编码为base64格式输入给大模型
    """
    with open(image_path, 'rb') as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

def compress_prompt(info):
    """压缩prompt内容"""

    client=ZhipuAI(api_key=key2)
    try:
        response=client.chat.completions.create(
            model="glm-4-plus",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"请对{info}的内容进行压缩，不改变句子原本的意思，压缩为不超过300个字符。"
                        }
                    ]
                }
            ]
        )
        return response.choices[0].message.content
    except ZhipuAIError as e:
        print(f"Error: {e}")
        return None

# 拼接完整的prompt
# sub_description=read_txt_file(img_description_path)
# combined_input=f"""这是图片描述{sub_description},请结合图片描述仔细观察这张图片。
# 特别注意任何可能的统计图表元素，例如标题、轴标签、数据系列等。
# 如果是统计图表，返回True，否则返回False。"""
# 拼接完整的prompt
def create_prompt(sub_description):
    """拼接prompt内容"""
    sub_description=compress_prompt(sub_description)
    if sub_description:
        prompt = f"""
        这是图片描述{sub_description},请结合图片描述仔细观察这张图片。
        特别注意任何可能的统计图表元素，例如标题、轴标签、数据系列等。
        如果是统计图表，返回True，否则返回False。
        """
        print("生成的prompt内容: ", prompt)  # 打印生成的prompt内容
        print("-----------------------------------------------------")
        return prompt
    else:
        print("图片描述为空，无法生成prompt")
        print("--------------------error，非正常退出-----------------------------")
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



def save_image_based_on_response(img_path, response,target_base_dir):
    """
    根据模型响应保存图片到指定目录
    """
    # 解析img_path获取需要的部分
    img_dir_part = '/'.join(img_path.split('/')[-3:-1])  # 获取fortune-or-fiction-final-v3_3/fortune-or-fiction-final-v3_3/[136, 470, 552, 684]_0.jpg部分
    img_name = os.path.basename(img_path)

    # 保存图片
    # 构造目标路径
    target_base_dir = target_base_dir
    if response.lower() == 'true':
        target_dir = os.path.join(target_base_dir, 'true', img_dir_part)
    else:
        target_dir = os.path.join(target_base_dir, 'false', img_dir_part)

    # 确保目标目录存在
    os.makedirs(target_dir, exist_ok=True)

    # 构造完整的目标文件路径
    target_file_path = os.path.join(target_dir, img_name)

    # 复制图片到目标路径
    shutil.copy2(img_path, target_file_path)  # 使用copy2保留元数据
    print(f"图片已保存至: {target_file_path}")

    # 保存prompt到同样位置
    prompt_file_path = os.path.join(target_dir, f"{os.path.splitext(img_name)[0]}.txt")
    with open(prompt_file_path, 'w', encoding='utf-8') as f:
        f.write(prompt)
    print(f"生成的prompt已保存至: {prompt_file_path}")



if __name__ == "__main__":
    # txt_path = "/home/ubuntu/moe/PaddleOCR_m/constructor/mini/source/20 全球Web3技术产业生态发展报告（2023年）23年）/find_result.txt"
    # txt_path="/home/ubuntu/moe/PaddleOCR_m/constructor/mini/source/20 全球Web3技术产业生态发展报告（2023年）/find_result.txt"
    # txt_path="/home/ubuntu/moe/PaddleOCR_m/constructor/mini/source/deloitte-cn-lshc-2023-global-health-care-outlook-zh-230316/find_result.txt"
    txt_path="/home/ubuntu/moe/PaddleOCR_m/constructor/result/pdf_test/new_week_in_charts/corporate-commitments-to-nature-have-evolved-since-2022/find_result.txt"
    # 尝试一下能不能不改名字
    # txt_path='/home/ubuntu/moe/PaddleOCR_m/constructor/mini/source/*/find_result.txt'
    
    while True:
        # 读取一组路径
        img_path = read_txt_odd(txt_path)
        if img_path is None:
            break
            
        img_description_path = read_txt_even(txt_path)
        if img_description_path is None:
            break
            
        # 读取图片描述
        sub_description = read_txt_file(img_description_path)

        if sub_description:
            # 生成拼接的prompt
            prompt = create_prompt(sub_description)

            # 检查prompt是否有效
            if prompt:
                # 调用模型并获取结果
                response = glm(img_path, key2, prompt)

                if response:
                    print("模型响应: ", response)
                    # 根据模型响应保存图片
                    # mini_path="/home/ubuntu/moe/PaddleOCR_m/constructor/mini/filter_output"
                    filter_path="/home/ubuntu/moe/PaddleOCR_m/constructor/result/pdf_test/refilter"
                    save_image_based_on_response(img_path, response,filter_path)
                else:
                    print("未能成功获取模型响应")
            else:
                print("生成的prompt无效，无法调用模型")
        else:
            print("没有读取到有效的图片描述")