import os
import json
import base64
from zhipuai import ZhipuAI
import re
import time

continue_processing = True
# 记录已处理文件的路径
def read_processed_files(processed_file):
    """读取已处理文件路径"""
    if os.path.exists(processed_file):
        with open(processed_file, 'r', encoding='utf-8') as f:
            # return set(f.read().splitlines())
            # 使用strip移除可能存在的空白字符
            return set(os.path.normpath(line.strip()) for line in f.readlines() if line.strip())
    return set()

def write_proifcessed_file(processed_file, file_path):
    """将已处理的文件路径写入记录文件"""
    with open(processed_file, 'a', encoding='utf-8') as f:
        f.write(file_path + '\n')

# 处理图像并保存响应到文件
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
        # 将response转换为字符串并保存为txt文件
        # 若为字典或列表，转换为json字符串再保存
        if isinstance(response, (dict, list)):
            response = json.dumps(response, ensure_ascii=False, indent=4)
        f.write(response)

def GetBcgAna(api_key, img_path, img_base, save_path, prompt):
    """获取背景信息"""
    global continue_processing #使用全局变量控制处理流程

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
        print(f"页面{img_path}的所有分析已完成")
        # save_rsp_to_json(response_content, os.path.join(save_path, 'rsp.json'))
        save_rsp_to_txt(response_content, os.path.join(save_path, 'rsp.txt'))
        return response_content
    except Exception as e:
        error_message = str(e)
        if "1113" in error_message or "欠费" in error_message:
            print(f"API_KEY耗尽， {error_message}")
            continue_processing = False
        else:
            print(f"处理文件 {img_path} 时出错：{e}")
        return ""

# 读取图像并转换为base64
def read_image(image_path):
    with open(image_path, 'rb') as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

def analyze_images(page_dir, main_dir, api_key, prompt0, processed_file):
    """
    分析主目录下所有图片的背景信息
    """
    page_path = os.path.join(main_dir, page_dir)

    # 获取当前处理文件夹的路径
    # folder_path = os.path.dirname(page_path)
    # folder_path = page_path
    # full_page_path = os.path.abspath(page_path)
    rela_page_path = os.path.realpath(page_path)

    # 检查当前页面是否已经处理过
    # if folder_path in processed_file:
    #     print(f"页面 {folder_path} 已处理，跳过")
    #     return
    if rela_page_path in processed_file:
        print(f"页面 {rela_page_path} 已处理，跳过")
        return

    # 子图路径
    sub_folder = os.path.join(page_path, page_dir)
    sub_files = []
    sub_pattern = r"\[(\d+), (\d+), (\d+), (\d+)\]_0\.jpg"  # 子图路径正则

    if os.path.isdir(sub_folder):
        # 通过正则表达式匹配子图文件
        sub_files = [f for f in os.listdir(sub_folder) if re.match(sub_pattern, f)]

    # 如果没有找到符合条件的子图文件，输出调试信息
    if not sub_files:
        print(f"在文件夹 {page_dir} 中没有找到符合条件的子图文件。")

    # 当前页、前页、后页路径
    cur_path = os.path.join(page_path, f"{page_dir}_current_page.png")
    pre_path = os.path.join(page_path, f"{page_dir}_prev_page.png")
    next_path = os.path.join(page_path, f"{page_dir}_next_page.png")

    # 当前页背景分析
    if os.path.exists(cur_path):
        img_base_cur = read_image(cur_path)
        # save_cur = os.path.join(page_path, 'glm_result', 'cur', 'response.json')
        save_cur = os.path.join(page_path, 'glm_result', 'cur', 'response.txt')
        response_cur = GetBcgAna(api_key, cur_path, img_base_cur, save_cur, prompt0)
    else:
        print(f"缺少当前页文件：{cur_path}")
        response_cur = " "

    # 前页背景分析
    response_pre = " "
    if os.path.exists(pre_path):
        img_base_pre = read_image(pre_path)
        # save_pre = os.path.join(page_path, 'glm_result', 'prev', 'response.json')
        save_pre = os.path.join(page_path, 'glm_result', 'prev', 'response.txt')
        response_pre = GetBcgAna(api_key, pre_path, img_base_pre, save_pre, prompt0)

    # 后页背景分析
    response_next = " "
    if os.path.exists(next_path):
        img_base_next = read_image(next_path)
        # save_next = os.path.join(page_path, 'glm_result', 'next', 'response.json')
        save_next = os.path.join(page_path, 'glm_result', 'next', 'response.txt')
        response_next = GetBcgAna(api_key, next_path, img_base_next, save_next, prompt0)

    # 子图分析
    for sub_file in sub_files:
        sub_path = os.path.join(sub_folder, sub_file)
        print(f"正在分析子图：{sub_path}")  # 输出子图路径进行调试

        img_base_sub = read_image(sub_path)
        # save_sub = os.path.join(page_path, 'glm_result', 'sub', f"{sub_file}.json")
        save_sub = os.path.join(page_path, 'glm_result', 'sub', f"{sub_file}.txt")

        prompt_sub = f"""
        请根据图片{sub_path}及其所在页面{response_cur}、前一页{response_pre}和后一页{response_next}的内容，全面分析这张图片的背景信息。
        """
        GetBcgAna(api_key, sub_path, img_base_sub, save_sub, prompt_sub)

    # 标记当前页面为已处理
    # write_processed_file(processed_file, folder_path)
    write_proifcessed_file(processed_file, rela_page_path)

    print(f"页面 {page_dir} 的所有分析已完成。")

def process_multiple_folders(main_dir, api_key, processed_file):
    """
    批量处理多个文件夹中的图片
    """
    global continue_processing

    subfolders = sorted([f for f in os.listdir(main_dir) if os.path.isdir(os.path.join(main_dir, f))])

    if not subfolders:
        print("没有找到任何文件夹")
        return

    prompt0 = "这是图片，请尽可能多地描述有关该图像的内容。"

    total_start_time = time.time() # 总的开始时间
    total_processed_folders = 0 # 记录处理过的文件夹数量
    total_processing_time = 0 # 总的处理时间

    for folder in subfolders:
        if not continue_processing:
            print("由于账户欠费，中断处理操作")
            break


        folder_path = os.path.join(main_dir, folder)
        print(f"开始处理文件夹: {folder_path}")

        folder_start_time = time.time()

        page_dirs = sorted([d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))])

        if not page_dirs:
            print(f"没有找到页面文件夹：{folder_path}")
            continue

        for page_dir in page_dirs:
            if not continue_processing:
                print("由于账户欠费，中断处理操作")
                break
            analyze_images(page_dir, folder_path, api_key, prompt0, processed_file)
            

        folder_end_time = time.time()
        folder_time = folder_end_time - folder_start_time
        total_processed_folders += 1
        total_processing_time += folder_time

        print(f"文件夹 {folder} 处理时间：{folder_time:.2f} 秒")

    total_end_time = time.time() # 总结束时间
    toal_time = total_end_time - total_start_time

    print(f"总共处理了{total_processed_folders}个文件夹。")
    print(f"总处理时间：{toal_time:.2f} 秒")
    avg_process_time = total_processing_time / total_processed_folders
    print(f"平均处理时间：{avg_process_time:.2f} 秒")
    print(f"总计用时：{toal_time/3600:.2f}小时")



        # print(f"文件夹 {folder} 的所有分析已完成。")
    



if __name__ == "__main__":
    main_directory = r"./constructor/result/try"  # 主文件夹路径
    # moe/PaddleOCR_m/constructor/result/12pp_result
    api_key = "3271d58be57cf9e23116b405a4b3f3c2.vyuU317jVmYrCw57"
    # api_key = "0b129ae24c4a37639222d6019d4a934e.NLi7Do892Cx637XA"
    # api_key = "97d68bdba109c8c12dc9e2dee95ed5df.8Hj8fWhGigEgzdZR" 
    # api_key = "2361a0b09e4d921d0fea047ed9dc2f29.gG1cj5POj1IvA0Qj"
    # api_key = "4a205612228af0f99567004ce61861a4.WPiYWfp7LB1hC2G1" # wry
    # api_key = "50ac1256cb84c79ec648ab975530dbef.zH4dIC7KOQK8yLgU" #8539 充值了的
    # api_key = "608b81c9b4526f6da90dd9acc5e7c8b5.bkHzsSDQzh16MfUO" #3968used
    # api_key = "3db90da3f9f64c4531c839b654a9ab5b.cWYwMUQrJiI99gmD" # 5652used
    # api_key = '7916aaef6c2af99dc9593c64701f8356.YWbNyp2aRGZQ3Z2U' #moe

    processed_file = "./constructor/result/try/processed_files.txt"  # 记录已处理文件的路径
    processed_files = read_processed_files(processed_file)  # 读取已处理文件列表

    process_multiple_folders(main_directory, api_key, processed_file)
