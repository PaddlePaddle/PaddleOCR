# 本脚本用于不同粒度的信息理解，采用模型为glm-4-long
# 粗粒度：整体，细粒度：局部
# 获取总结后再进行图片理解和qa对的生成

import os
import logging
import json
import base64
from zhipuai import ZhipuAI
from pathlib import Path

# glm_result给出的txt结果文件保存在一定深度的目录下，需要进行批量、精准的提取

def read_json_files(directory):
    """读取目录下的JSON文件，
    返回一个包含所有JSON文件的列表。
    """
    json_files = []

    base_path = Path(directory) #使用path库简化路径查找

    # 要查找的文件模式
    # moe/PaddleOCR_m/constructor/result/1202result_glm/test/2021中国医疗AI行业研究报告/2021中国医疗AI行业研究报告_20/glm_result/cur/response.json/rsp.json
    # moe/PaddleOCR_m/constructor/result/1202result_glm/test/2021中国医疗AI行业研究报告/2021中国医疗AI行业研究报告_19/glm_result/sub/[18, 302, 565, 777]_0.jpg.json/rsp.json

    patterns = [
        "**/glm_result/cur/response.json/rsp.json",  # 查找cur目录下的/rsp.json
        "**/glm_result/next/response.json/rsp.json",  # 查找next目录下的/rsp.json
        "**/glm_result/prev/response.json/rsp.json",  # 查找prev目录下的/rsp.json
        "**/glm_result/sub/*_0.jpg.json/rsp.json"  # 查找sub目录下的rsp.json
    ]

    for pattern in patterns:
        # 用glob递归查找
        matches = list(base_path.glob)
        print(f"Matching files for pattern '{pattern}: {matches}")# 调试输出，查看匹配结果

        for path in base_path.glob(pattern):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    json_files.append(data)
            except json.JSONDecodeError:
                print(f"无法解析JSON文件：{path}")
            except Exception as e:
                print(f"出现问题：{path}，错误信息{e}")
        
    return json_files


def get_text(single_txt_path, overall_txt_path):
    """
    将单个文本和整体文本合并，并保存到指定路径。
    该txt文本路径较深，需要进行通用的路径查找。
    """

    single_txt_path = os.path.abspath(single_txt_path)
    print(f"要读取的单个txt路径总目录为:{single_txt_path}")

    for singe_txt in os.listdir(single_txt_path):
        with open(single_txt_path + singe_txt, 'r', encoding='utf-8') as f:
            overall_txt = f.read()
            with open(overall_txt_path, 'a', encoding='utf-8') as f:
                f.write(overall_txt)

    return overall_txt_path


def overall_analysis(text_key, base_json_dir, prompt_template, overall_json_path):
    """
    读取多个JSON文件，合并成一个大的文本，并用llm进行总结。
    """

    # 读取并合并所有json内容
    json_contents = read_json_files(base_json_dir)

    if not json_contents:
        print("没有找到任何JSON文件。")
        return None

    # 将所有JSON合并为一个大的文本格式
    combined_text = '\n'.join([json.dumps(content, ensure_ascii=False) for content in json_contents])

    # 打印一部分合并后的文本用于调试
    try:
        for_test_num = int(input("请输入要测试的json文件数量："))
        print(f"合并后的文本前{for_test_num}个字符：")
        print(combined_text[:{for_test_num}])
    except ValueError:
        print("无效输入！默认显示前100个字符")
        print(combined_text[:100])
    
    # 指定outputpath用于保存合并后的文本
    if overall_json_path:
        with open(overall_json_path, 'w', encoding='utf-8') as f:
            f.write(combined_text)

    # text_content = f"文本路径：{base_json_dir}"
    # 构建prompt时，使用整合的文本内容
    prompt = prompt_template.format(text_content=combined_text)

    # 初始化API客户端
    client = ZhipuAI(api_key=text_key)

    try:
        overall_response = client.chat.completions.create(
            model = "glm-4-plus", #长文本专用模型
            messages= [
                {"role":"system",
                "content":"你是一个专业的文本分析专家，你的任务是分析并总结文本内容，同时给出一个总结。"
                },
                {
                    "role":"user",
                    "content": prompt
                }
            ]
        )

        print("Overall response:", overall_response)

        # 输出分析结果
        response_message = overall_response.choices[0].message.content
        print(f"Model response:{response_message}")

        return response_message
    
    except Exception as e:
        print(f"An error occured while calling API：{e}")
        return None

if __name__ == "__main__":
    import os

    # 从环境变量中获取API密钥，以提高安全性
    API_KEY = os.getenv("ZHIPUAI_API_KEY", "608b81c9b4526f6da90dd9acc5e7c8b5.bkHzsSDQzh16MfUO")
    # a1c9ccc9f10111c1506f4f2975deacd7.FW450DCYI5DbdWik
    
    BASE_JSON_DIR = "./constructor/result/1202result_glm/test/2021中国医疗AI行业研究报告/2021中国医疗AI行业研究报告_1/"
    BASE_JSON_DIR = os.path.abspath(BASE_JSON_DIR)  # 获取绝对路径
    
    PROMPT_TEMPLATE = """
    请根据文本{text_content}，结合你的知识库，分析并总结出该文本的关键信息，并给出一个总结。
    """
    
    OUTPUT_PATH = "./constructor/12overall/combined.json"
    OUTPUT_PATH = os.path.abspath(OUTPUT_PATH)  # 获取绝对路径
    
    # 确保输出目录存在
    output_dir = os.path.dirname(OUTPUT_PATH)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    try:
        result = overall_analysis(API_KEY, BASE_JSON_DIR, PROMPT_TEMPLATE, OUTPUT_PATH)
        print("Summary generated successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")

# # 初步筛选过程
# # 首先仅提取glm_result文件夹下存在sub的文件夹/正则匹配二级目录下的x.x.x.x.jpg,将图片和json描述放到筛选文件夹下（叠加全局的信息标记
# def img_filter(base_path, img_path):
#     pass

# # 然后进行图表筛选，看模板图表的关键词是否与模板匹配
# chart_template = ["柱状图", "折线图", "饼图", "雷达图", "散点图", "箱线图", "热力图", "树图", "地图"]
# def chart_filter(img_path):
#     pass


# # 长文本部分
# from zhipuai import ZhipuAI
# text_key = "4a205612228af0f99567004ce61861a4.WPiYWfp7LB1hC2G1"
# def overall_analysis(text_key, json_path, prompt, text_content, overall_json_path):
    
#     json_path =  os.path.join() 
#     # json_path格式说明：为一系列二级文件夹下子文件中的rsp.json格式的内容，以下为示例：
#     # moe/PaddleOCR_m/constructor/result/1202result/test/2021中国医疗AI行业研究报告/2021中国医疗AI行业研究报告_0/glm_result/cur/response.json/rsp.json
#     # moe/PaddleOCR_m/constructor/result/1202result/test/2021中国医疗AI行业研究报告/2021中国医疗AI行业研究报告_1/glm_result/cur/response.json/rsp.json
#     # 需要整合为一个大文件送入大模型进行总结

#     overall_json_path = ''#所有json内容的集合，并合并保存为一个大的txt格式，每个
#     text_content = f"文本路径：{json_path}" # 先访问一系列文件夹，获取其中的json文件，整合内容然后输出总结（txt格式


#     prompt = f"""
#     请根据文本{text_content}，结合你的知识库，分析并总结出该文本的关键信息，并给出一个总结。
#     """

#     client = ZhipuAI(api_key=text_key) # 填写您自己的APIKey
#     overall_response = client.chat.completions.create(
#         model="glm-4-plus",  # 填写需要调用的模型编码
#         messages=[
#             {"role": "system", "content": "你是一个乐于解答各种问题的助手，你的任务是为用户提供专业、准确、有见地的建议。"},
#             {"role": "user", "content": prompt}
#         ],
#     )
#     print(overall_response.choices[0].message)
#     return overall_response

# # qa对生成部分
# # 采用glm-4v-plus模型

# filter_key = ""
# client = ZhipuAI(api_key = filter_key)

# prompt = "根据{json}的内容，分析图片{sub_img}。这是你可以同步参考的内容{cur_path_json}, {prev_path_json}以及{next_path_json}"

# qa_pair_template = {
#     "柱状图": {
#         "简单": "请根据图表，回答关于该图的问题。",
#         "中等": "请根据图表，回答关于该图的问题。",
#         "困难": "请根据图表，回答关于该图的问题。"
#     },
#     "折线图": {
#         "简单": "请根据图表，回答关于该图的问题。",
#         "中等": "请根据图表，回答关于该图的问题。",
#     }
# }
# def ReadImage(img_path):
#     """
#     读取图像文件并进行Base64编码，返回编码后的字符串。
#     """
#     with open(img_path, 'rb') as img_file:
#         return base64.b64encode(img_file.read()).decode('utf-8')

# # 对图片进行上下文分析并判断是否为统计图表
# def ImageFilter(img_path, re_filter_path, qa_pair_path):
#     qapair = []

#     # 读取图像文件并进行Base64编码
#     img_base = ReadImage(img_path)

#     # 获取图像的背景信息
#     ic_response = client.chat.completions.create(
#         model="glm-4v-plus",
#         messages=[{
#             "role": "user",
#             "content": [
#                 {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base}" }},
#                 {"type": "text", "text": "请详细描述这张图片的内容。特别注意任何可能的统计图表元素，例如标题、轴标签、数据系列等。"}
#             ]
#         }]
#     )
#     background = ic_response.choices[0].message.content
#     print(f"该图的上下文背景信息为: {background}")

#     # 判断图像是否为统计图表
#     filter_response = client.chat.completions.create(
#         model="glm-4v-plus",
#         messages=[{
#             "role": "user",
#             "content": [
#                 {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base}" }},
#                 {"type": "text", "text": f"请根据{background}的描述，判断该图片类型是否为{', '.join(accepted_chart_type)}中的类型，如有，则输出True。如果不是，回答False。"}
#             ]
#         }]
#     )
#     filtered = filter_response.choices[0].message.content.strip().lower() == 'true'

#     if filtered:
#         # 获取文件名（不包括扩展名）
#         base_name = os.path.splitext(os.path.basename(img_path))[0]

#         # 保存统计图表图像
#         new_img_path = os.path.join(re_filter_path, f"{base_name}.jpg")
#         with open(new_img_path, 'wb') as new_img_file:
#             new_img_file.write(base64.b64decode(img_base))
#         print(f"Image saved to {new_img_path}")

#         # 保存上下文背景信息
#         context_info_path = os.path.join(re_filter_path, f"{base_name}_context.json")
#         with open(context_info_path, 'w', encoding='utf-8') as f:
#             json.dump({"background": background}, f, ensure_ascii=False, indent=4)
#         print(f"Context information saved to {context_info_path}")

#         # 生成基于统计图表的QA对
#         for chart_type in chart_template:
#             if chart_type in background:
#                 for difficulty, question in qa_pair_template[chart_type].items():
#                     # 调用API获取答案
#                     response = client.chat.completions.create(
#                         model="glm-4v-plus",
#                         messages=[{
#                             "role": "user",
#                             "content": [
#                                 {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base}" }},
#                                 {"type": "text", "text": f"根据{background}，{question}"}
#                             ]
#                         }]
#                     )
#                     answer = response.choices[0].message.content
#                     qapair.append({"chart_type": chart_type, "difficulty": difficulty, "question": question, "answer": answer})

#         # 将生成的QA对保存到JSON文件
#         qa_json_path = os.path.join(qa_pair_path, f"{base_name}_qa_pairs.json")
#         with open(qa_json_path, 'w', encoding='utf-8') as f:
#             json.dump(qapair, f, ensure_ascii=False, indent=4)
#         print(f"QA pairs saved to {qa_json_path}")
#     else:
#         print("Image is not a statistical chart and was not saved.")

#     return qapair


# if __name__ == "__main__":
#     filter_file_path = ""
#     json_path = ""
#     filtered_path = ""


