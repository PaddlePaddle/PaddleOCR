# 该脚本用于构建chat-glm用于文本问答，借助global和local两种粒度的信息生成问题模板
# 流程为：
# 1. 用户指定global和local信息的路径，读取并放入列表or字典中
# 2. 构建prompt,根据两种粒度的信息,结合few-shot提示的模板
# 3. 调用glm-4的文本模型,根据两种粒度的信息,结合few-shot提示的模板
# 4. 输出问题模板
# 5. 保存问题模板到指定位置

import time
import os

from zhipuai import ZhipuAI

domain = input("请输入您要生成的模板的领域:")


global_info="moe/PaddleOCR_m/constructor/result/aggrev_result/_aggregated.txt"
local_info =""

prompt_role_system=f"""
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
print(f"即将输入的prompt模板样式为:{prompt_role_system}")

txt_path=""

# 需要指定输出的格式,便于后续的步骤
prompt_role_user=f"""
这是文件{txt_path}的内容,你需要理解并接收粗粒度信息{global_info}和细粒度信息{local_info},
结合图表的内容进行问题的模板。
"""

def chat_model(api_key, prompt, txt_path,response):
    """调用glm-4的文本模型,根据两种粒度的信息,结合few-shot提示的模板
    进行动态的模板生成
    """
    client = ZhipuAI(api_key=api_key)
    txt=txt_path
    
    response= client.chat.completions.create(
    model="glm-4-plus",  # 填写需要调用的模型编码，也可以采用glm-4-long
    messages=[
        {"role": "system", "content": prompt},
        {"role": "user", "content": txt}
    ],
    )
    real_response=response.choices[0].message.content
    return real_response

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

        return ''.join(content_lines) # 将列表转换为字符串并返回
    except FileNotFoundError:
        print(f"文件{file_path}不存在")
        return None

def read_txt_file(file_path):
    content=count_file_info(file_path)
    if content is not None:
        print(content)

api_key="59c47aa0217fb8a72944a2a0c6b9d2eb.B7EG63vlMlr88ejp"




# 指明身份,系统提示

#注:txt_path后续需要与find_path联动,根据find_path输出的内容进行user_prompt的生成
medical_chat=chat_model(api_key,prompt_role_system,)


    

# txt_file_path='/home/ubuntu/moe/PaddleOCR_m/constructor/result/aggrev_result/_aggregated.txt'
txt_file_path='moe/PaddleOCR_m/constructor/result/aggrev_result/americas-small-businesses-time-to-think-big_aggregated_乱序.txt'
read_txt_file(txt_file_path)




# 模型编码:
# glm-4-plus、glm-4-0520、glm-4-air、glm-4-airx、glm-4-long 、 glm-4-flashx 、 glm-4-flash；

# client = ZhipuAI(api_key=api_key) # 填写您自己的APIKey
# response = client.chat.completions.create(
#     model="glm-4-plus",  # 填写需要调用的模型编码
#     messages=[
#         {"role": "system", "content": prompt_role_system},
#         {"role": "user", "content": prompt_role_user}
#     ],
# )
# answer = response.choices[0].message.content
# print(answer)