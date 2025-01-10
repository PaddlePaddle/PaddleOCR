# 本脚本作废
import os
import json
import base64
from zhipuai import ZhipuAI
import re

def save_rsp_to_json(response, filename):
    """将response保存为json文件"""
    directory = os.path.dirname(filename)
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(response, f, ensure_ascii=False, indent=4)

def save_rsp_to_txt(response, filename):
    """将response保存为txt文件"""
    directory = os.path.dirname(filename)
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(response)

def GetBcgAna(api_key, img_path, img_base, save_path, prompt):
    """获取背景信息"""
    try:
        client = ZhipuAI(api_key=api_key)
        response = client.chat.completions.create(
            model="glm-4v-plus",  # 使用GLM-4V-Plus模型
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": img_base}},
                    {"type": "text", "text": prompt}
                ]
            }]
        )
        response_content = response.choices[0].message.content
        save_rsp_to_json(response_content, os.path.join(save_path, 'rsp.json'))
        return response_content
    except Exception as e:
        print(f"处理文件 {img_path} 时出错：{e}")
        return ""

# 获取当前页、前一页和后一页的图片
def process_images(page_dir, main_dir, new_key, prompt0):
    page_path = os.path.join(main_dir, page_dir)
    # 子图路径正则
    sub_pattern = r"\[(\d+), (\d+), (\d+), (\d+)\]_0\.jpg"
    
    sub_folder = os.path.join(page_path, page_dir)
    sub_files = [f for f in os.listdir(sub_folder) if re.match(sub_pattern, f)]
    
    # 当前页、前页和后页文件路径
    cur_path = os.path.join(page_path, f"{page_dir}_current_page.png")
    pre_path = os.path.join(page_path, f"{page_dir}_prev_page.png")
    next_path = os.path.join(page_path, f"{page_dir}_next_page.png")
    
    # 检查当前页是否存在
    if not os.path.exists(cur_path):
        print(f"缺少当前页文件 {cur_path}")
        return
    
    # 读取图片并转换为base64
    def read_image(image_path):
        with open(image_path, 'rb') as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')
    
    img_base_cur = read_image(cur_path)
    img_base_pre = read_image(pre_path) if os.path.exists(pre_path) else None
    img_base_next = read_image(next_path) if os.path.exists(next_path) else None
    
    # 保存路径
    save_cur = os.path.join(page_path, 'glm_result', 'cur')
    save_pre = os.path.join(page_path, 'glm_result', 'pre')
    save_next = os.path.join(page_path, 'glm_result', 'next')
    save_sub = os.path.join(page_path, 'glm_result', 'sub')
    
    os.makedirs(save_cur, exist_ok=True)
    os.makedirs(save_pre, exist_ok=True)
    os.makedirs(save_next, exist_ok=True)
    os.makedirs(save_sub, exist_ok=True)
    
    # 获取当前页背景分析
    response_cur = GetBcgAna(new_key, cur_path, img_base_cur, save_cur, prompt0)
    
    # 获取前一页背景分析
    response_pre = GetBcgAna(new_key, pre_path, img_base_pre, save_pre, prompt0) if img_base_pre else " "
    
    # 获取后一页背景分析
    response_next = GetBcgAna(new_key, next_path, img_base_next, save_next, prompt0) if img_base_next else " "
    
    # 子图背景分析
    for sub_file in sub_files:
        sub_path = os.path.join(sub_folder, sub_file)
        img_base_sub = read_image(sub_path)
        
        # 生成子图的提示词
        prompt_sub = f"""
        请根据图片{sub_path}及其所在页面{response_cur}、前一页{response_pre}和后一页{response_next}的内容，全面地分析并描述这张图片的上下文信息。具体来说，请注意图片的标题、描述性文字以及它在整个文档中的位置关系。如果图片是统计图表，请明确指出，并进一步分析：
        
        - 如果是折线图，重点描述数据的趋势变化，比如上升、下降、波动等特征；
        - 如果是柱状图，重点关注各组数据的分布情况，比如最高值、最低值、集中区域等；
        - 对于其他类型的图表（如饼图、散点图等），也请根据其特点进行相应的分析。
        
        此外，请结合文档中图表附近的文字内容，推测图表所传达的主要信息或结论。同时，考虑前后页的相关信息，分析这些信息如何相互补充或对比。最终输出应包括：图片的整体描述、图表的具体分析（如果有）、以及前后页内容对理解该图表的帮助。
        """
        
        # 调用背景分析函数进行子图背景分析
        # response_sub = GetBcgAna(new_key, sub_path, img_base_sub, save_sub, prompt_sub)
        GetBcgAna(new_key, sub_path, img_base_sub, save_sub, prompt_sub)
    
    print(f"页面 {page_dir} 的所有回复均已保存到指定位置。")

def main():
    # 设置主目录和API密钥
    # main_dir = r"./constructor/result/1120result/medical1120"
    # main_dir=r"./constructor/result/1120result/medical1120/2024年中国医疗大健康产业发展白皮书1"
    main_dir="./constructor/result/12pp_result/11/americas-small-businesses-time-to-think-big"
    new_key = '50ac1256cb84c79ec648ab975530dbef.zH4dIC7KOQK8yLgU'
    prompt0 = "这是图片,请尽可能多的描述有关该图像的内容。"

    # 获取所有页面文件夹
    page_dirs = sorted([d for d in os.listdir(main_dir) if os.path.isdir(os.path.join(main_dir, d))])

    if not page_dirs:
        print("没有找到任何页面文件夹")
        exit()

    for page_dir in page_dirs:
        process_images(page_dir, main_dir, new_key, prompt0)

    print("所有页面分析结束。")

if __name__ == "__main__":
    main()



# import os
# import json
# import base64
# from zhipuai import ZhipuAI
# import re

# def save_rsp_to_json(response, filename):
#     """将response保存为json文件"""
#     directory = os.path.dirname(filename)
#     if not os.path.exists(directory):
#         os.makedirs(directory)
#     with open(filename, 'w', encoding='utf-8') as f:
#         json.dump(response, f, ensure_ascii=False, indent=4)

# def save_rsp_to_txt(response, filename):
#     """将response保存为txt文件"""
#     directory = os.path.dirname(filename)
#     if not os.path.exists(directory):
#         os.makedirs(directory)
#     with open(filename, 'w', encoding='utf-8') as f:
#         f.write(response)

# def GetBcgAna(api_key, img_path, img_base, save_path, prompt):
#     """获取背景信息"""
#     try:
#         client = ZhipuAI(api_key=api_key)
#         response = client.chat.completions.create(
#             model="glm-4v-plus",  # 使用GLM-4V-Plus模型
#             messages=[{
#                 "role": "user",
#                 "content": [
#                     {"type": "image_url", "image_url": {"url": img_base}},
#                     {"type": "text", "text": prompt}
#                 ]
#             }]
#         )
#         response_content = response.choices[0].message.content
#         # print(f"{img_path} 描述的内容为：", response_content)

#         print(f"页面{page_dir}的所有分析已完成")
#         save_rsp_to_json(response_content, os.path.join(save_path, 'rsp.json'))
#         return response_content
#     except Exception as e:
#         print(f"处理文件 {img_path} 时出错：{e}")
#         return ""

# # 获取当前页、前一页和后一页的图片
# def process_images(page_dir, main_dir, new_key, prompt0):
#     page_path = os.path.join(main_dir, page_dir)
#     # 子图路径正则
#     sub_pattern = r"\[(\d+), (\d+), (\d+), (\d+)\]_0\.jpg"
    
#     sub_folder = os.path.join(page_path, page_dir)
#     sub_files = [f for f in os.listdir(sub_folder) if re.match(sub_pattern, f)]
    
#     # 当前页、前页和后页文件路径
#     cur_path = os.path.join(page_path, f"{page_dir}_current_page.png")
#     pre_path = os.path.join(page_path, f"{page_dir}_prev_page.png")
#     next_path = os.path.join(page_path, f"{page_dir}_next_page.png")
    
#     # 检查当前页是否存在
#     if not os.path.exists(cur_path):
#         print(f"缺少当前页文件 {cur_path}")
#         return
    
#     # 读取图片并转换为base64
#     def read_image(image_path):
#         with open(image_path, 'rb') as img_file:
#             return base64.b64encode(img_file.read()).decode('utf-8')
    
#     img_base_cur = read_image(cur_path)
#     img_base_pre = read_image(pre_path) if os.path.exists(pre_path) else None
#     img_base_next = read_image(next_path) if os.path.exists(next_path) else None
    
#     # 保存路径
#     save_cur = os.path.join(page_path, 'glm_result', 'cur')
#     save_pre = os.path.join(page_path, 'glm_result', 'pre')
#     save_next = os.path.join(page_path, 'glm_result', 'next')
#     save_sub = os.path.join(page_path, 'glm_result', 'sub')
    
#     os.makedirs(save_cur, exist_ok=True)
#     os.makedirs(save_pre, exist_ok=True)
#     os.makedirs(save_next, exist_ok=True)
#     os.makedirs(save_sub, exist_ok=True)
    
#     # 获取当前页背景分析
#     response_cur = GetBcgAna(new_key, cur_path, img_base_cur, save_cur, prompt0)
    
#     # 获取前一页背景分析
#     response_pre = GetBcgAna(new_key, pre_path, img_base_pre, save_pre, prompt0) if img_base_pre else " "
    
#     # 获取后一页背景分析
#     response_next = GetBcgAna(new_key, next_path, img_base_next, save_next, prompt0) if img_base_next else " "
    
#     # 子图背景分析
#     for sub_file in sub_files:
#         sub_path = os.path.join(sub_folder, sub_file)
#         img_base_sub = read_image(sub_path)
        
#         # 生成子图的提示词
#         prompt_sub = f"""
#         请根据图片{sub_path}及其所在页面{response_cur}、前一页{response_pre}和后一页{response_next}的内容，全面地分析并描述这张图片的上下文信息。具体来说，请注意图片的标题、描述性文字以及它在整个文档中的位置关系。如果图片是统计图表，请明确指出，并进一步分析：
        
#         - 如果是折线图，重点描述数据的趋势变化，比如上升、下降、波动等特征；
#         - 如果是柱状图，重点关注各组数据的分布情况，比如最高值、最低值、集中区域等；
#         - 对于其他类型的图表（如饼图、散点图等），也请根据其特点进行相应的分析。
        
#         此外，请结合文档中图表附近的文字内容，推测图表所传达的主要信息或结论。同时，考虑前后页的相关信息，分析这些信息如何相互补充或对比。最终输出应包括：图片的整体描述、图表的具体分析（如果有）、以及前后页内容对理解该图表的帮助。
#         """
        
#         # 调用背景分析函数进行子图背景分析
#         # response_sub = GetBcgAna(new_key, sub_path, img_base_sub, save_sub, prompt_sub)
#         GetBcgAna(new_key, sub_path, img_base_sub, save_sub, prompt_sub)
    
#     print(f"页面 {page_dir} 的所有回复均已保存到指定位置。")

# # 设置主目录和API密钥
# # main_dir = r"./constructor/result/1121result/test/"
# main_dir = r"./constructor/result/1120result/medical1120"
# new_key = '948a77cea8d6a502d0a9b46ee919bd11.iXa2yARr4TyNybao'
# prompt0 = "这是图片,请尽可能多的描述有关该图像的内容。"

# # 获取所有页面文件夹
# page_dirs = sorted([d for d in os.listdir(main_dir) if os.path.isdir(os.path.join(main_dir, d))])

# if not page_dirs:
#     print("没有找到任何页面文件夹")
#     exit()

# for page_dir in page_dirs:
#     process_images(page_dir, main_dir, new_key, prompt0)

# print("所有页面分析结束。")
