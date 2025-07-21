import os
import base64
from zhipuai import ZhipuAI, ZhipuAIError

# Constants
API_KEY = "59c47aa0217fb8a72944a2a0c6b9d2eb.B7EG63vlMlr88ejp"
ACCEPTED_CHART_TYPES = ['饼图', '柱状图', '折线图', '雷达图', '散点图', '堆积图']
BCK_PATH = '/home/ubuntu/moe/PaddleOCR_m/constructor/mini/filter_output/true/fortune-or-fiction-final-v3_3/[136, 470, 552, 684]_0.txt'
CUR_PATH = '/home/ubuntu/moe/PaddleOCR_m/constructor/mini/filter_output/true/fortune-or-fiction-final-v3_3/cur/rsp.txt'
NEXT_PATH = '/home/ubuntu/moe/PaddleOCR_m/constructor/mini/filter_output/true/fortune-or-fiction-final-v3_3/next/rsp.txt'
PREV_PATH = '/home/ubuntu/moe/PaddleOCR_m/constructor/mini/filter_output/true/fortune-or-fiction-final-v3_3/prev/rsp.txt'
OUTPUT_DIR = '/home/ubuntu/moe/PaddleOCR_m/constructor/mini/global&local'

# Template for global prompt generation
GLOBAL_TEMPLATE = {
    "background": "这是一张显示了不同国家或地区在三个不同类别上的得分比较的图表。",
    "context": "图表的内容为各个国家或地区在'乐观'、'悲观'和'焦虑'三个类别上的得分，每个类别用不同的颜色表示，并且每个柱状条上标注了具体的得分数值。",
    "legend": "图例内容分别表示：乐观（蓝色）、悲观（浅蓝色）、焦虑（深蓝色）。",
    "type": "图表类型为水平柱状图。"
}

def generate_prompt(template, **kwargs):
    """Generates a prompt text dynamically based on the given template and keyword arguments."""
    return template.format(**kwargs)

def read_file(file_path, file_type="txt"):
    """Reads a file and returns its content as a string."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read().strip()
            if content:
                print(f"文件{file_path}的前50个字符为：{content[:50]}")
                return content
            else:
                raise ValueError(f"文件{file_path}为空。")
    except FileNotFoundError:
        print(f"错误：文件{file_path}不存在。")
    except Exception as e:
        print(f"读取文件{file_path}时发生错误：{e}")
    return None

def encode_image_to_base64(image_path):
    """Encodes an image file to a base64 string."""
    try:
        with open(image_path, 'rb') as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')
    except Exception as e:
        print(f"图片编码失败：{e}")
        return None

def query_glm(api_key, system_prompt, user_prompt, model="glm-4-plus"):
    """Calls the large language model API and returns the response."""
    try:
        client = ZhipuAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        return response.choices[0].message.content
    except ZhipuAIError as e:
        print(f"模型调用失败：{e}")
        return None

def save_to_file(content, output_path):
    """Saves content to a specified path."""
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as file:
            file.write(content)
            print(f"内容成功保存至 {output_path}")
    except Exception as e:
        print(f"保存内容到 {output_path} 时失败：{e}")

def main():
    # Read file contents
    global_user_prompt = read_file(BCK_PATH)
    cur_content = read_file(CUR_PATH)
    prev_content = read_file(PREV_PATH)
    next_content = read_file(NEXT_PATH)

    # Construct global and local prompts
    global_prompt = generate_prompt(
        "请提取有关图片的描述，格式如下：\n"
        "background: {background},\ncontext: {context},\nlegend: {legend},\ntype: {type}\n",
        **GLOBAL_TEMPLATE
    )

    local_prompt = f"""
    这是可以参考的内容：\n当前信息: {cur_content or '无'}, \n前一页内容: {prev_content or '无'}, \n后一页内容: {next_content or '无'}。\n
    请以以下格式输出：\n"
    context: "这张图表的大致讲述了...",\nprev: "前一页的内容描述了...",\nnext: "后一页的内容描述了..."
    """

    # Call the model to generate content
    global_info = query_glm(API_KEY, global_prompt, global_user_prompt or "")
    local_info = query_glm(API_KEY, local_prompt, "")

    # Save results
    save_to_file(global_info or "", os.path.join(OUTPUT_DIR, 'global_info.txt'))
    save_to_file(local_info or "", os.path.join(OUTPUT_DIR, 'local_info.txt'))

if __name__ == "__main__":
    main()