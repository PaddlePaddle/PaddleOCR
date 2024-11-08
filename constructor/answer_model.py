#api_key 
key2='144ef5e548561b44952df83e8d9f3193.ZYUFc6qyYV9KTLC6'

import base64
from zhipuai import ZhipuAI

# img_path =  r"/home/ubuntu/moe/PaddleOCR/constructor/medical_consult/2021中国医疗AI行业研究报告/[48, 1229, 1127, 1639]_15.jpg"
img_path=r'PaddleOCR/constructor/medical_consult/2021中国医疗AI行业研究报告/[84, 662, 1118, 1036]_10.jpg'


with open(img_path, 'rb') as img_file:
    img_base = base64.b64encode(img_file.read()).decode('utf-8')

question1='辅助检测在三类证AI医学影像产品中占比是多少？'

client = ZhipuAI(api_key=key2) # 填写您自己的APIKey
response = client.chat.completions.create(
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
            "text": "请根据图片内容回答{question1}提出的问题,注意，仅返回与问题有关的内容，不需要额外的扩展。"
          }
        ]
      }
    ]
)
answer=response.choices[0].message.content
print(answer)

# 图中是关于获批NMPA三类证的AI医学影像产品分布情况（按应用场景）的扇形统计图，数据来源是国家药监局，动脉网，36氪研究院，统计时间截至2021年11月18日。图中显示：

# - 辅助检测占比23.8%，病种包括肺结节、肋骨骨折等，功能是自动识别病灶，分析影像特征。
# - 辅助分诊及评估占比57.1%，病种包括肺炎、心血管病、骨龄评估等，功能是辅助分诊提示、辅助病情评估。
# - 辅助诊断占比19.0%，病种包括颅内肿瘤、糖网等，功能是辅助判断是否患病、辅助进行疾病分类。