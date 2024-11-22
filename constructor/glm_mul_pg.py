#  该脚本用于测试glm是否可以在一次api调用中同时处理多张图片
#  已知glm可以进行多轮对话，但是一轮对话中仅可使用一张图片
# 多模态大模型往往使用自定义数据集进行微调

# 获得子图和三张定位图片

import os
import json


def save_rsp_to_json(response, filename):
    """
    将response保存为json文件
    args:
      response: api调用返回的response对象
      filename: 保存路径
    """
    directory= os.path.dirname(filename)
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    with open(filename,'w',encoding='utf-8') as f:
        json.dump(response.to_dict(),f, ensure_ascii=False, indent=4)
      
  
    
  

key2 = '7916aaef6c2af99dc9593c64701f8356.YWbNyp2aRGZQ3Z2U'

# imgpath
# moe/PaddleOCR_m/constructor/result_batch_copy_有子图无过滤后res/medical/2021中国医疗AI行业研究报告/2021中国医疗AI行业研究报告_0/2021中国医疗AI行业研究报告_0/[7, 9, 596, 836]_0.jpg

import base64
from zhipuai import ZhipuAI

# 传入图片路径，并将结果保存到指定目录下
img_path = r"constructor/result_batch_copy_有子图无过滤后res/medical/2021中国医疗AI行业研究报告/2021中国医疗AI行业研究报告_8/2021中国医疗AI行业研究报告_8/[53, 439, 277, 590]_0.jpg"
with open(img_path, 'rb') as img_file:
    img_base = base64.b64encode(img_file.read()).decode('utf-8')
save_img=r''
os.makedirs(save_img, exist_ok=True)

# img2_path = r"constructor/result_batch_copy_有子图无过滤后res/medical/2021中国医疗AI行业研究报告/2021中国医疗AI行业研究报告_8/2021中国医疗AI行业研究报告_8/[315, 443, 552, 587]_0.jpg"
# with open(img2_path, 'rb') as img_file:
#     img2_base = base64.b64encode(img_file.read()).decode('utf-8')

# 传入当前页面的地址并将结果保存到指定目录下
cur_path=r"constructor/result_batch_copy_有子图无过滤后res/medical/2021中国医疗AI行业研究报告/2021中国医疗AI行业研究报告_8/2021中国医疗AI行业研究报告_8_current_page.png"
with open(cur_path, 'rb') as img_file:
    cur_base = base64.b64encode(img_file.read()).decode('utf-8')
save_cur=r''
os.makedirs(save_cur, exist_ok=True)

# 传入上一页面的地址并将结果保存到指定目录下
pre_path=r"constructor/result_batch_copy_有子图无过滤后res/medical/2021中国医疗AI行业研究报告/2021中国医疗AI行业研究报告_8/2021中国医疗AI行业研究报告_8_prev_page.png"
with open(pre_path, 'rb') as img_file:
    pre_base = base64.b64encode(img_file.read()).decode('utf-8')
save_pre=r''
os.makedirs(save_pre,exist_ok=True)

# 传入下一页面的地址并将结果保存到指定目录下
next_path=r"constructor/result_batch_copy_有子图无过滤后res/medical/2021中国医疗AI行业研究报告/2021中国医疗AI行业研究报告_8/2021中国医疗AI行业研究报告_8_next_page.png"
with open(next_path, 'rb') as img_file:
    next_base = base64.b64encode(img_file.read()).decode('utf-8')
save_next=r''
os.makedirs(save_next,exist_ok=True))

res_txt_path="./constructor/result_batch_copy_有子图无过滤后res/medical/2021中国医疗AI行业研究报告/2021中国医疗AI行业研究报告_8/2021中国医疗AI行业研究报告_8/res_0.txt"






"""对整张图像进行描述"""

prompt0="这是图片,请尽可能多的描述有关该图像的内容。"

client = ZhipuAI(api_key=key2) # 填写您自己的APIKey
response_cur = client.chat.completions.create(
    model="glm-4v-plus",  # 填写需要调用的模型名称
    messages=[
      {
        "role": "user",
        "content": [
          {
            "type": "image_url",
            "image_url": {
                "url": cur_base
            }
          },
          {
            "type": "text",
            "text": prompt0
          }
        ]
      }
    ]
)
print("当前页面表示内容为：",response_cur.choices[0].message.content)
print("--------------------------------------------------------------")

# 当前页面表示内容为： 该图片为一篇关于医疗AI在行业的发展的文章。

# 文章主要讨论了以下几个观点：

# 1. **我国医疗资源分布不均、优质医疗资源过度集中的特点**：一方面，基层医疗机构存在基础设施相对缺乏和医生能力不足等问题；另一方面，全国三甲医院及优秀医师主要集中在发达城市及东部沿海地区，中西部地区医疗资源相对匮乏。

# 2. **医疗AI助力提质增效，能够有效弥补我国高水平医师短缺问题**：利用机器学习等AI技术，可以快速识别病灶并迅速训练模型，提升诊疗效率及准确率，从而补充医生数量缺口。

# 此外，文中还包含了一些数据和图表来支持这些观点：
# - 图表展示了2020年我国医疗机构数量及诊疗人次数据来源为国家卫健委，由36氪研究院制作。
# - 另一个图表则显示了2015年至2020年全国卫生技术人员数（单位：万人）的变化情况。

# 总的来说，这篇文章强调了医疗AI在我国解决医疗资源分配不均和高水平医师短缺方面的重要作用和发展潜力。



client = ZhipuAI(api_key=key2) # 填写您自己的APIKey
response_pre = client.chat.completions.create(
    model="glm-4v-plus",  # 填写需要调用的模型名称
    messages=[
      {
        "role": "user",
        "content": [
          {
            "type": "image_url",
            "image_url": {
                "url": pre_base
            }
          },
          {
            "type": "text",
            "text": prompt0
          }
        ]
      }
    ]
)
print("前一页面表示内容为：", response_pre.choices[0].message.content)
print("--------------------------------------------------------------")

# 前一页面表示内容为： 这是一张关于“1.3.2 行业发展驱动因素分析——技术”的幻灯片页面。标题为：“人工智能技术加速突破，为医疗AI行业发展提供创新动力”。内容主要分为两部分：首先讨论了当前人工智能技术的发展趋势及其在医疗健康领域的应用；然后列举了一些具体的人工智能技术在医疗场景中的应用案例。

# 第一部分：
# - 当前，人工智能技术已经迎来了第三次浪潮，理论和技术的进步使得语音识别等感知智能技术取得了重大突破，机器学习等技术开始广泛应用于各个领域。
# - 与此同时，AI技术与医疗健康领域的融合不断加深，以计算机视觉、自然语言处理和机器学习为代表的技术已广泛渗透在医疗行业的各个场景中，成为提升医疗服务水平的重要驱动力。

# 第二部分详细介绍了三种人工智能子技术在医疗场景中的应用：

# 1. **计算机视觉**:
#    - 定义: 用机器人替代人眼对目标进行识别、跟踪和测量的技术。
#    - 应用: 主要应用于医疗信息化、医学影像、药物研发等领域。例如，智能导诊机器人可以识别患者性别、年龄等信息; 计算机视觉技术可以对CT、MRI等影像进行图像分割、特征提取。

# 2. **自然语言处理(NLP)**:
#    - 定义: 是实现人与计算机之间用自然语言进行有效通信的技术。
#    - 应用: 主要应用于电子病历、健康管理、药物研发等领域。例如，利用自然语言处理能够将诊疗记录、医嘱等进行标准化、结构化重构，形成电子病历数据。

# 3. **机器学习**:
#    - 定义: 通过学习样本数据内在规律、表示层次，使机器具备理解分析和智能决策能力。
#    - 应用: 广泛应用于医疗行业各个场景。例如，学习大量临床影像数据和诊断经验，进行辅助诊疗; 利用深度学习技术对分子结构进行分析与处理，缩短药物研发周期。

# 最后的部分提到我国AI科研产出水平位居全球前列，根据《中国人工智能发展报告2020》的数据，近十年，中国AI专利申请量为389571件，位居世界第一，占全球总量的74.7%，是排名第二的美国专利申请量的8.2倍。未来随着我国医疗AI复合型人才相对稀缺短板的不断补足，人工智能技术与医疗领域的融合将进一步深化，赋能医疗行业高质量发展。

# 此外，页面的右下角有36氪研究院的水印标志。



client = ZhipuAI(api_key=key2) # 填写您自己的APIKey
response_next = client.chat.completions.create(
    model="glm-4v-plus",  # 填写需要调用的模型名称
    messages=[
      {
        "role": "user",
        "content": [
          {
            "type": "image_url",
            "image_url": {
                "url": next_base
            }
          },
          {
            "type": "text",
            "text": prompt0
          }
        ]
      }
    ]
)
print("后一页面表示内容为：",response_next.choices[0].message.content)
print("--------------------------------------------------------------")

# 后一页面表示内容为： 这是一份关于中国医疗AI市场规模的报告，标题为“1.4 行业规模分析”。报告中提到2020年全球医疗AI智能市场规模为42亿美元，预计到2027年将增至345亿美元，复合增长率为35.1%。其中，AI医学影像是增长较快的细分应用市场之一。

# 从细分类别来看，药物研发、医学影像等细分市场的增速较高，预计2025年我国医疗AI市场规模有望突破300亿元。根据动脉网的数据，按照大数据、AI+新药研发和AI+肿瘤诊疗三大赛道市场规模总量估算，2020年中国医疗AI市场规模为66.25亿元，预计2020-2025年的CAGR为39.4%。

# 在AI医学影像方面，国内起步相对较早，目前市场规模较小，但未来随着政策支持及需求拉动下，发展速度将会不断加快。据Frost & Sullivan数据，中国AI医学影像市场规模将由2020年的3.4亿元增至2030年的923.1亿元，2020-2030年的CAGR高达75.1%。

# 图表部分展示了2019年至2025E的中国医疗AI主要应用领域市场规模及增速情况，以及2020至2030E的中国AI医学影像市场规模（单位：亿元）及其增长率。


"""========================================================================================"""
"""背景分析"""
prompt1="这是图片{img_path},请结合图片所在页面{response_cur}，图片的前一页{response_pre}，图片的后一页{response_next}，尽可能多的描述有关该图像的上下文信息,包括标题、图片整体趋势。如果是统计图表则需要点明，接着关注图表类型，柱状图关注分布情况，折线图关注趋势。结合前一张和后一张图片说明。输出格式为：该图片说明了什么内容，前后两张图各自说明了什么内容。图片本身是否为统计图表，若是则什么类型。然后详细描述图表的内容。"

prompt11 = f"""
请根据图片{img_path}及其所在页面{response_cur}、前一页{response_pre}和后一页{response_next}的内容，全面地分析并描述这张图片的上下文信息。具体来说，请注意图片的标题、描述性文字以及它在整个文档中的位置关系。如果图片是统计图表，请明确指出，并进一步分析：

- 如果是折线图，重点描述数据的趋势变化，比如上升、下降、波动等特征；
- 如果是柱状图，重点关注各组数据的分布情况，比如最高值、最低值、集中区域等；
- 对于其他类型的图表（如饼图、散点图等），也请根据其特点进行相应的分析。

此外，请结合文档中图表附近的文字内容，推测图表所传达的主要信息或结论。同时，考虑前后页的相关信息，分析这些信息如何相互补充或对比。最终输出应包括：图片的整体描述、图表的具体分析（如果有）、以及前后页内容对理解该图表的帮助。
"""

client = ZhipuAI(api_key=key2) # 填写您自己的APIKey
response1 = client.chat.completions.create(
    model="glm-4v-plus",  # 填写需要调用的模型名称
    messages=[
      {
        "role": "user",
        "content": [
          {
            "type": "image_url",
            "image_url": {
                "url": img_base
                

            }
          },
          {
            "type": "text",
            "text": prompt11
          }
        ]
      }
    ]
)
print("该子图的背景信息为：",response1.choices[0].message.content)

# 该子图的背景信息为： 该图片展示了医疗卫生机构数和诊疗人次的统计数据。

# 前一页图片（{pre_base}）没有给出具体的信息，因此无法直接描述其内容或与当前图片的关系。

# 当前图片是一张柱状图和折线图的组合：
# - 柱状图显示了医院和基层医疗卫生机构的数量（单位：万个），其中医院的数量是3.54万个，而基层医疗卫生机构的数量是97.01万个。
# - 折线图表示了这两类机构的诊疗人次（单位：亿人次），其中医院的诊疗人次是33.2亿人次，基层医疗卫生机构的诊疗人次是41.2亿人次。

# 从图中可以看出，基层医疗卫生机构的数量远多于医院，但医院的诊疗人次略少于基层医疗卫生机构。

# 后一页图片（{next_base}）也没有给出具体的信息，因此无法直接描述其内容或与当前图片的关系。

# 综上所述，这张图片通过对比医院和基层医疗卫生机构的数量及诊疗人次，揭示了两者在医疗服务体系中的不同角色和贡献。

print("本轮分析结束。=======================================")


