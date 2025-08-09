
# 本脚本为img_filter.py的copy部分但略有不同，主体功能是完备的
# 使用glm对已经处理的文件进行背景分析与筛选
# 1105运行版本

import base64
import os
from zhipuai import ZhipuAI
import json

# API key配置
key2 = '7916aaef6c2af99dc9593c64701f8356.YWbNyp2aRGZQ3Z2U'
client = ZhipuAI(api_key=key2)

# 图表类型列表
accepted_chart_type = ['饼图', '柱状图', '折线图']

# QA对模板
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

# 输入输出文件路径
# 包含输入文件路径、筛选后的图表路径、qapair路径
input_path = r'./constructor/result/pp_result/medical/deloitte-cn-lshc-internet-hospitals-in-china-the-new-step-into-digital-healthcare-zh-210315/'
# PaddleOCR/constructor/result/pp_result/medical/2021中国医疗AI行业研究报告
# /home/ubuntu/moe/PaddleOCR/constructor/result/caict/电信业发展蓝皮书（2024年）——智能化发展
# PaddleOCR/constructor/result/pp_result/caict/全球产业创新生态发展报告（2023年）——数字创新高地全球图景与中国位势
# PaddleOCR/constructor/result/pp_result/caict/全球数字经济白皮书（2023年）
re_output_path = r'./constructor/result/re_filtered/medical/deloitte-cn-lshc-internet-hospitals-in-china-the-new-step-into-digital-healthcare-zh-210315/'
qa_pair_path = r'./constructor/result/re_filtered/qa_pairs/deloitte-cn-lshc-internet-hospitals-in-china-the-new-step-into-digital-healthcare-zh-210315/'

# 确保输出目录存在
if not os.path.exists(re_output_path):
    os.makedirs(re_output_path)

if not os.path.exists(qa_pair_path):
    os.makedirs(qa_pair_path)

# 对图片进行上下文分析并判断是否为统计图表
def analyze_and_filter_image(img_path, re_filter_path, qa_pair_path):
    # 初始化 qapair 变量
    qapair = []

    with open(img_path, 'rb') as img_file:
        img_base = base64.b64encode(img_file.read()).decode('utf-8')

    # 调用API获取图片背景信息
    ic_response = client.chat.completions.create(
        model="glm-4v-plus",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base}"}},
                    {"type": "text", "text": "请详细描述这张图片的内容。特别注意任何可能的统计图表元素，例如标题、轴标签、数据系列等。"}
                ]
            }
        ]
    )
    background = ic_response.choices[0].message.content
    print(f"该图的上下文背景信息为: {background}")

    # 判断图片是否为统计图表
    filter_response = client.chat.completions.create(
        model="glm-4v-plus",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base}"}},
                    {"type": "text", "text": f"请根据{background}的描述，判断该图片类型是否为{', '.join(accepted_chart_type)}中的类型，如有，则输出True。如果不是，回答False。"}
                ]
            }
        ]
    )
    filtered = filter_response.choices[0].message.content.strip().lower() == 'true'

    # 识别出统计图表和对应的上下文信息，保存到指定路径
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
                print(f"Found chart type: {chart_type}")
                if chart_type in qa_pair_template:
                    for difficulty, question in qa_pair_template[chart_type].items():
                        # 调用API获取答案
                        response = client.chat.completions.create(
                            model="glm-4v-plus",
                            messages=[
                                {
                                    "role": "user",
                                    "content": [
                                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base}"}},
                                        {"type": "text", "text": f"根据{background}，{question}"}
                                    ]
                                }
                            ]
                        )
                        answer = response.choices[0].message.content
                        qapair.append({"chart_type": chart_type, "difficulty": difficulty, "question": question, "answer": answer})
                else:
                    print(f"Chart type {chart_type} not found in qa_pair_template")

        # 将生成的QA对保存到JSON文件
        qa_json_path = os.path.join(qa_pair_path, f"{base_name}_qa_pairs.json")
        with open(qa_json_path, 'w', encoding='utf-8') as f:
            json.dump(qapair, f, ensure_ascii=False, indent=4)
        print(f"QA pairs saved to {qa_json_path}")

    else:
        print("Image is not a statistical chart and was not saved.")

    return qapair

# 示例调用
for filename in os.listdir(input_path):
    if filename.endswith('.jpg'):
        img_path = os.path.join(input_path, filename)
        analyze_and_filter_image(img_path, re_output_path, qa_pair_path)