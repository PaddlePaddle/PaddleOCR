import base64
import os
import json
from zhipuai import ZhipuAI

# API key配置
key2 = '7916aaef6c2af99dc9593c64701f8356.YWbNyp2aRGZQ3Z2U'
client = ZhipuAI(api_key=key2)

# 图表类型列表
accepted_chart_type = ['饼图', '柱状图', '折线图']
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

# 输入文件路径
sub_img_path = r""
# 
sub_json_path = r''
cur_json_path = r''
next_json_path = r''
prev_json_path = r''
chart_position_path = r''

# 输出文件路径
filtered_path = '' # 存放筛选出来的子图
os.makedirs(filtered_path, exist_ok=True)
qa_pair_path = '' # 存放生成的QA对，和子图一一对应
os.makedirs(qa_pair_path, exist_ok=True)

# 首先查询子文件夹目录下是否存在名为sub的文件夹，有则下一步
# 若存在sub文件夹，载入sub_img和sub_json，喂入模型进行分析
# 同时要结合当前页、前页和后页的内容
# 

prompt = "根据{json}的内容，分析图片{sub_img}。这是你可以同步参考的内容{cur_path_json}, {prev_path_json}以及{next_path_json}"

def ReadImage(img_path):
    with open(img_path, 'rb') as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

# 对图片进行上下文分析并判断是否为统计图表
def ImageFilter(img_path, re_filter_path, qa_pair_path):
    qapair = []

    # 读取图像文件并进行Base64编码
    img_base = ReadImage(img_path)

    # 获取图像的背景信息
    ic_response = client.chat.completions.create(
        model="glm-4v-plus",
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base}" }},
                {"type": "text", "text": "请详细描述这张图片的内容。特别注意任何可能的统计图表元素，例如标题、轴标签、数据系列等。"}
            ]
        }]
    )
    background = ic_response.choices[0].message.content
    print(f"该图的上下文背景信息为: {background}")

    # 判断图像是否为统计图表
    filter_response = client.chat.completions.create(
        model="glm-4v-plus",
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base}" }},
                {"type": "text", "text": f"请根据{background}的描述，判断该图片类型是否为{', '.join(accepted_chart_type)}中的类型，如有，则输出True。如果不是，回答False。"}
            ]
        }]
    )
    filtered = filter_response.choices[0].message.content.strip().lower() == 'true'

    if filtered:
        # 获取文件名（不包括扩展名）
        base_name = os.path.splitext(os.path.basename(img_path))[0]

        # 保存统计图表图像
        new_img_path = os.path.join(re_filter_path, f"{base_name}.jpg")
        with open(new_img_path, 'wb') as new_img_file:
            new_img_file.write(base64.b64decode(img_base))
        print(f"Image saved to {new_img_path}")

        # 保存上下文背景信息
        context_info_path = os.path.join(re_filter_path, f"{base_name}_context.json")
        with open(context_info_path, 'w', encoding='utf-8') as f:
            json.dump({"background": background}, f, ensure_ascii=False, indent=4)
        print(f"Context information saved to {context_info_path}")

        # 生成基于统计图表的QA对
        for chart_type in accepted_chart_type:
            if chart_type in background:
                for difficulty, question in qa_pair_template[chart_type].items():
                    # 调用API获取答案
                    response = client.chat.completions.create(
                        model="glm-4v-plus",
                        messages=[{
                            "role": "user",
                            "content": [
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base}" }},
                                {"type": "text", "text": f"根据{background}，{question}"}
                            ]
                        }]
                    )
                    answer = response.choices[0].message.content
                    qapair.append({"chart_type": chart_type, "difficulty": difficulty, "question": question, "answer": answer})

        # 将生成的QA对保存到JSON文件
        qa_json_path = os.path.join(qa_pair_path, f"{base_name}_qa_pairs.json")
        with open(qa_json_path, 'w', encoding='utf-8') as f:
            json.dump(qapair, f, ensure_ascii=False, indent=4)
        print(f"QA pairs saved to {qa_json_path}")
    else:
        print("Image is not a statistical chart and was not saved.")

    return qapair


# 示例调用：处理指定目录中的所有图像文件
for filename in os.listdir(sub_img_path):
    if filename.endswith('.jpg'):
        img_path = os.path.join(sub_img_path, filename)
        ImageFilter(img_path, filtered_path, qa_pair_path)

if __name__ == '__main__':
    pass