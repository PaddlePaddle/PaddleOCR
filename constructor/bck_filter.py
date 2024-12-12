# glm背景分析+filter功能

import base64
import os
import json
from zhipuai import ZhipuAI

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

# API key配置
api_key = '7916aaef6c2af99dc9593c64701f8356.YWbNyp2aRGZQ3Z2U'
client = ZhipuAI(api_key=api_key)

class ChartProcessor:
    def __init__(self, accepted_chart_types, qa_template, client):
        self.accepted_chart_types = accepted_chart_types
        self.qa_template = qa_template
        self.client = client

    def _get_image_base64(self, img_path):
        """读取并返回图片的base64编码"""
        with open(img_path, 'rb') as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')

    def _analyze_image_context(self, img_base64):
        """调用大模型分析图片背景信息"""
        response = self.client.chat.completions.create(
            model="glm-4v-plus",
            messages=[
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}},
                    {"type": "text", "text": "请详细描述这张图片的内容。特别注意任何可能的统计图表元素，例如标题、轴标签、数据系列等。"}
                ]}
            ]
        )
        return response.choices[0].message.content

    def _filter_chart(self, img_base64, background):
        """判断图片是否为统计图表"""
        filter_response = self.client.chat.completions.create(
            model="glm-4v-plus",
            messages=[
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}},
                    {"type": "text", "text": f"请根据{background}的描述，判断该图片类型是否为{', '.join(self.accepted_chart_types)}中的类型，如有，则输出True。如果不是，回答False。"}
                ]}
            ]
        )
        return filter_response.choices[0].message.content.strip().lower() == 'true'

    def _generate_qa_pairs(self, img_base64, background, chart_type):
        """根据图表背景生成问答对"""
        qa_pairs = []
        for difficulty, question in self.qa_template[chart_type].items():
            response = self.client.chat.completions.create(
                model="glm-4v-plus",
                messages=[
                    {"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}},
                        {"type": "text", "text": f"根据{background}，{question}"}
                    ]}
                ]
            )
            answer = response.choices[0].message.content
            qa_pairs.append({"chart_type": chart_type, "difficulty": difficulty, "question": question, "answer": answer})
        return qa_pairs

    def process_image(self, img_path, re_filter_path, qa_pair_path):
        """处理单张图片，判断是否为统计图表，生成QA对并保存"""
        img_base64 = self._get_image_base64(img_path)
        background = self._analyze_image_context(img_base64)

        # 判断图表是否符合要求
        is_filtered = self._filter_chart(img_base64, background)
        if not is_filtered:
            print(f"图像 {img_path} 不是统计图表，跳过处理。")
            return []

        # 保存图表
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        new_img_path = os.path.join(re_filter_path, f"{base_name}.jpg")
        with open(new_img_path, 'wb') as new_img_file:
            new_img_file.write(base64.b64decode(img_base64))
        print(f"图像已保存到 {new_img_path}")

        # 保存图表的背景信息
        context_info_path = os.path.join(re_filter_path, f"{base_name}_context.json")
        with open(context_info_path, 'w', encoding='utf-8') as f:
            json.dump({"background": background}, f, ensure_ascii=False, indent=4)
        print(f"背景信息已保存到 {context_info_path}")

        # 根据图表背景生成问答对
        all_qapairs = []
        for chart_type in self.accepted_chart_types:
            if chart_type in background:
                qa_pairs = self._generate_qa_pairs(img_base64, background, chart_type)
                all_qapairs.extend(qa_pairs)

        # 保存生成的QA对
        qa_json_path = os.path.join(qa_pair_path, f"{base_name}_qa_pairs.json")
        with open(qa_json_path, 'w', encoding='utf-8') as f:
            json.dump(all_qapairs, f, ensure_ascii=False, indent=4)
        print(f"QA对已保存到 {qa_json_path}")
        return all_qapairs


def process_all_images(input_path, re_filter_path, qa_pair_path):
    """处理指定目录下的所有图片文件"""
    processor = ChartProcessor(accepted_chart_type, qa_pair_template, client)

    for filename in os.listdir(input_path):
        if filename.endswith('.jpg'):
            img_path = os.path.join(input_path, filename)
            processor.process_image(img_path, re_filter_path, qa_pair_path)


# 示例调用
input_path = './constructor/result/1121result/test/2021中国医疗AI行业研究报告_glm/2021中国医疗AI行业研究报告_8/2021中国医疗AI行业研究报告_8'
# moe/PaddleOCR_m/constructor/result/1121result/test/2021中国医疗AI行业研究报告_glm/2021中国医疗AI行业研究报告_8/2021中国医疗AI行业研究报告_8
re_output_path = './constructor/result/re_filtered/others/testttt1128/'
qa_pair_path = './constructor/result/re_filtered/qa_pairs/'

# 确保输出目录存在
os.makedirs(re_output_path, exist_ok=True)
os.makedirs(qa_pair_path, exist_ok=True)

# 处理所有图片
process_all_images(input_path, re_output_path, qa_pair_path)
