import os
from zhipuai import ZhipuAI, ZhipuAIError
import re

# QA模板
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

# API Key和文件路径
api_key = "59c47aa0217fb8a72944a2a0c6b9d2eb.B7EG63vlMlr88ejp"
global_info_path = "./result/aggrev_result/_aggregated.txt"
# local_info_path = "./result/aggrev_result/local/fortune-or-fiction-final-v3_3---2--4_aggregated.txt"
local_info_path="./result/aggrev_result/local/fortune-or-fiction-final-v3_3---1--5_aggregated.txt" 

# 读取文件内容并返回
def count_file_info(file_path):
    """整理文件信息，确保不超过模型的token限制，限制会导致信息的不完整"""
    try:
        line_count = 0
        word_count = 0
        char_count = 0
        content_lines = []

        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                line_count += 1
                words = line.split()
                word_count += len(words)
                char_count += len(line.strip('\n'))  # 忽略换行
                content_lines.append(line)  # 将每一行内容添加到列表中

        print(f"文件{file_path}的行数是:{line_count}")
        print(f"文件{file_path}的字数是:{word_count}")
        print(f"文件{file_path}的字符数(去除换行符）是:{char_count}")

        return ''.join(content_lines)  # 将列表转换为字符串并返回
    except FileNotFoundError:
        print(f"文件{file_path}不存在")
        return None

def read_txt_file(file_path):
    """读取txt文件内容，"""
    content = count_file_info(file_path)
    if content is not None and content.strip():
        print("文件内容的前50个字符为：", content[:50])
        return content
    else:
        print(f"文件{file_path}不存在或{content}为空")
        return None

def read_global_info(global_info_path):
    global_info = read_txt_file(global_info_path)
    return global_info

def read_local_info(local_info_path):
    local_info = read_txt_file(local_info_path)
    return local_info


def generate_prompt(domain, global_info, local_info):
    """生成LLM的提示内容"""
    prompt_role_system = f"""
你是一个经验老到的{domain}领域模板生成大师,现在需要你根据模板的样式,结合接受的两种上下文,生成针对图表的问题\n
注意,先进行模板类型的确认,然后生成问题的模板,最后再结合接受到的两种上下文,生成针对图表的问题\n
示例模板的样式如下:\n
1. 柱状图:\n
    1.1 简单:某年销售额是多少? \n
    1.2 数学:2018年与2019年之间的销售增长量是多少? \n
    1.3 趋势:从2018年到2020年,销售额的变化趋势是什么? \n
2. 折线图:\n
    2.1 简单:某年销售额是多少?\n
    2.2 数学:2018年与2019年之间的销售增长量是多少?\n
    2.3 趋势:从2018年到2020年,销售额的变化趋势是什么?\n
3. 饼图:\n
    3.1 简单:某年销售额是多少?\n
    3.2 数学:2018年与2019年之间的销售增长量是多少?\n
    3.3 趋势:从2018年到2020年,销售额的变化趋势是什么?\n
"""
    prompt_role_user = f"""
这是文件内容的全局信息:{global_info}和局部信息:{local_info},
结合图表的内容生成适当的问题模板，并结合实际内容生成具体问题。
"""
    return prompt_role_system, prompt_role_user


def glm(global_info, local_info, api_key, prompt_role_system, prompt_role_user):
    """调用ZhipuAI生成问题模板"""
    client = ZhipuAI(api_key=api_key)

    try:
        response = client.chat.completions.create(
            model="glm-4-plus",  # 填写需要调用的模型名称
            messages=[
                {"role": "system", "content": prompt_role_system},
                {"role": "user", "content": prompt_role_user}
            ]
        )
        real_response = response.choices[0].message.content
        return real_response
    except ZhipuAIError as e:
        print(f"Error: {e}")
        return None


# 主程序
def main():
    domain = input("请输入您要生成的模板的领域: ")

    # 读取全局和局部信息
    global_info = read_global_info(global_info_path)
    local_info = read_local_info(local_info_path)

    if not global_info or not local_info:
        print("无法读取所需的全局或局部信息")
        return

    # 生成LLM的提示
    prompt_role_system, prompt_role_user = generate_prompt(domain, global_info, local_info)

    # 获取LLM的回答
    result = glm(global_info, local_info, api_key, prompt_role_system, prompt_role_user)
    if result:
        print("生成的模板为：")
        print(result)
    else:
        print("生成模板失败。")

if __name__ == "__main__":
    main()
