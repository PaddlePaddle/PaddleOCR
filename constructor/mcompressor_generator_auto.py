"""自动化"""
"""一波处理掉，手动把glm_result文件夹拉过来，不用修改路径，记得及时处理掉true文件夹里处理过的内容"""

import os
import re
import base64
from zhipuai import ZhipuAI, ZhipuAIError

# 图表问答模板
qa_tempalte={
    "柱状图": {
      "介绍": "柱状图通过不同高度的条形展示各类别或各时间点的数据，适用于比较不同类别的数量或趋势变化。",
      "简单": "某类别的数量是多少？",
      "数学": "从一个时间段到另一个时间段，某类别的变化量是多少？",
      "趋势": "柱状图中，某类别的变化趋势如何？",
      "对比": "哪个类别的数量最大？最小？与其他类别相比，差异有多大？",
      "百分比": "某类别在总数据中占比多少？"
    },
    "折线图": {
      "介绍": "折线图通过点与点之间的连线展示数据的变化趋势，适用于显示时间序列数据的波动与趋势。",
      "简单": "某一时段的数据是多少？",
      "数学": "从一个时间段到另一个时间段，数据的变化量是多少？",
      "趋势": "折线图中的数据变化趋势是上升还是下降？",
      "对比": "哪个时间段的数据变化最大？与其他时段的趋势相比，有什么不同？",
      "预测": "根据当前的变化趋势，预测未来的数据趋势如何？"
    },
    "饼图": {
      "介绍": "饼图通过扇形区域展示各部分数据占总数的比例，适用于显示部分与整体的关系。",
      "简单": "某类别占总数据的比例是多少？",
      "数学": "某类别与另一个类别的比例差距是多少？",
      "趋势": "饼图中，某部分的比例是增加还是减少？",
      "对比": "哪一部分的比例最大/最小？各部分之间的比例差异如何？",
      "细节": "某类别的比例如何影响整体数据？"
    },
    "散点图": {
      "介绍": "散点图通过点在坐标系中的分布展示两个变量之间的关系，适用于分析变量之间的相关性。",
      "简单": "某个点的坐标是多少？",
      "数学": "两个点之间的距离是多少？",
      "趋势": "散点图中，点的分布显示出怎样的趋势？",
      "对比": "哪些点之间的差异最大？是否存在聚集或离散的情况？",
      "相关性": "散点图中的点是否显示出两个变量之间的相关性？"
    },
    "堆积图": {
      "介绍": "堆积图通过将多个数据系列叠加展示各类别的累计数据，适用于显示各部分对整体的贡献。",
      "简单": "某类别的累计值是多少？",
      "数学": "某类别与其他类别的差距是多少？",
      "趋势": "堆积图中，哪个部分的变化最为显著？",
      "对比": "堆积图中的各部分增长趋势如何？某部分的变化是否超出预期？",
      "百分比": "某部分占总体的百分比是多少？"
    }
  }

# 路径提取函数
def extract_segment_from_filename(filename):
    pattern = re.compile(r'\[(\d+(?:,\s*\d+)*)\]_0\.jpg')
    match = pattern.search(filename)
    return match.group(1) if match else None

# 路径生成函数
def generate_paths(base_path):
    for root, _, files in os.walk(base_path):
        for file in files:
            segment = extract_segment_from_filename(file)
            if segment:
                return {
                    "global_info_path": f"{base_path}/glm_result/sub/rsp.txt",#注意一下文件夹的扩展名
                    "img_path": f"{base_path}/[{segment}]_0.jpg",#注意
                    "cur_info_path": f"{base_path}/glm_result/cur/rsp.txt",
                    "next_info_path": f"{base_path}/glm_result/next/rsp.txt",
                    "prev_info_path": f"{base_path}/glm_result/prev/rsp.txt",
                }
    return None

# 获取两级子目录的函数
def get_two_level_dirs(base_path):
    """
    获取指定路径下的两级子目录完整路径
    
    Args:
        base_path (str): 基础路径
        
    Returns:
        list: 包含所有两级子目录完整路径的列表
    """
    result = []
    
    # 确保base_path存在且是目录
    if not os.path.isdir(base_path):
        return result
        
    # 遍历第一级目录
    for first_level in os.listdir(base_path):
        first_path = os.path.join(base_path, first_level)
        
        # 确保是目录而不是文件
        if os.path.isdir(first_path):
            # 遍历第二级目录
            for second_level in os.listdir(first_path):
                second_path = os.path.join(first_path, second_level)
                
                # 如果是目录，添加到结果列表
                if os.path.isdir(second_path):
                    result.append(second_path)
                    
    return result

# 读取文本文件内容
def read_txt(path) -> str:
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as file:
            return file.read()
    return None

# 读取图片并转换为base64
def read_image(image_path: str) -> str:
    with open(image_path, 'rb') as file:
        return base64.b64encode(file.read()).decode('utf-8')

# 调用GLM进行文本处理
def glm_t(api_key, prompt_system, prompt_user):
    client = ZhipuAI(api_key=api_key)
    response = client.chat.completions.create(
        model="glm-4-plus",
        messages=[{"role": "system", "content": prompt_system}, {"role": "user", "content": prompt_user}]
    )
    return response.choices[0].message.content

# 调用GLM处理图片
def glm_v(img_path, api_key, prompt):
    img_base = read_image(img_path)
    client = ZhipuAI(api_key=api_key)
    try:
        response = client.chat.completions.create(
            model="glm-4v-plus",
            messages=[{"role": "user", "content": [{"type": "image_url", "image_url": {"url": img_base}}, {"type": "text", "text": prompt}]}]
        )
        return response.choices[0].message.content
    except ZhipuAIError as e:
        print(f"Error: {e}")
        return None

# 主程序入口
def main():
    # 设置基本路径
    # base_path = "/home/ubuntu/moe/PaddleOCR_m/constructor/mini/filter_output/true"
    # base_path="/home/ubuntu/moe/PaddleOCR_m/constructor/mini/filter_output/true"
    # base_path="/home/ubuntu/moe/PaddleOCR_m/constructor/mini/filter/true"
    # base_path="/home/ubuntu/moe/PaddleOCR_m/constructor/mini/filter_output/true_glm_tomv"
    # base_path="/home/ubuntu/moe/PaddleOCR_m/constructor/mini/filter_output/true_glm_subjson" #sub下面为.jpg,subjson文件夹后面要改成.jpg.json
    # base_path="/home/ubuntu/moe/PaddleOCR_m/constructor/mini/filter_output/add"
    # base_path="/home/ubuntu/moe/PaddleOCR_m/constructor/result/pdf_test/refilter/true-add"
    base_path="/home/ubuntu/moe/PaddleOCR_m/constructor/result/pdf_test/refilter/true-add"
    
    # 获取所有两级子目录
    dirs = get_two_level_dirs(base_path)
    
    for dir_path in dirs:
        print(f"Processing directory: {dir_path}")
        
        # 生成路径
        paths = generate_paths(dir_path)
        if not paths:
            print(f"没有找到符合条件的文件 in {dir_path}")
            continue
        
        # 读取全局信息并处理
        global_info = read_txt(paths["global_info_path"])
        if not global_info:
            print(f"未找到全局信息文件 in {dir_path}")
            continue
        
        global_prompt_system = f"你是一个智能图表分析专家，请按要求输出。这是全局信息：{global_info},并对内容进行压缩。"
        global_prompt_user = """你需要根据图表的描述，输出一个关于图表的描述，并输出一个关于图表的图表类型。"""
        
        api_key = "59c47aa0217fb8a72944a2a0c6b9d2eb.B7EG63vlMlr88ejp"
        global_response = glm_t(api_key, global_prompt_system, global_prompt_user)

        # 保存全局信息响应
        parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(paths["global_info_path"])))
        global_target_path = os.path.join(parent_dir, "global.txt")
        os.makedirs(parent_dir, exist_ok=True)
        with open(global_target_path, "w") as f:
            f.write(global_response)
        print(f"文件已保存至: {global_target_path}")

        # 读取局部信息并处理
        cur_info = read_txt(paths["cur_info_path"])
        next_info = read_txt(paths["next_info_path"])
        prev_info = read_txt(paths["prev_info_path"])

        local_prompt_system = f"你是一个智能图表分析专家，请按要求输出。这是局部信息：当前信息:{cur_info}, 前一页内容{prev_info},后一页内容{next_info}。"
        local_prompt_user = """请根据获取到的信息以如下格式输出，输出简要内容，但请包括各自的细节"""
        
        local_response = glm_t(api_key, local_prompt_system, local_prompt_user)

        # 保存局部信息响应
        local_target_path = os.path.join(parent_dir, "local.txt")
        with open(local_target_path, "w") as f:
            f.write(local_response)
        print(f"文件已保存至: {local_target_path}")

        # 生成QA对
        qa_pair_generator_prompt = f"你是一个智能图表分析专家，请按要求输出。全局信息：{global_response}, 局部信息：{local_response}。\n请结合模板生成QA对"
        qa_pair = glm_v(paths["img_path"], api_key, qa_pair_generator_prompt)

        # 保存QA对
        qapair_target_path = os.path.join(parent_dir, "qapair.txt")
        with open(qapair_target_path, "w") as f:
            f.write(qa_pair)
        print(f"文件已保存至: {qapair_target_path}")

if __name__ == "__main__":
    main()
