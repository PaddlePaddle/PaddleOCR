"""找到子图和对应路径的列表"""
"""返璞归真用这版"""
"""0114ver 找到路径下[x,x,x,x]_0.jpg和对应的描述文件"""

"""修改root_path修改为对应路径"""
import os

def generate_paths(root_dir,exclude_pattern):
    paths = []  # 用来存储生成的路径
    exclude_pattern=exclude_pattern
    
    # 遍历根目录及其子目录
    for root, dirs, files in os.walk(root_dir):
        # 过滤出jpg文件 排除result.jpg文件
        for file in files:
            if file.endswith('.jpg') and not file.endswith(exclude_pattern):

                # # 获取 img_path
                img_path = os.path.join(root, file)
                
                # 构造 img_description_path、
                # 提取img_path的目录部分
                img_dir=os.path.dirname(img_path)
                last_dir = os.path.basename(img_dir)
                # 替换最后一个子目录
                # glm_result_path = os.path.join(os.path.dirname(img_path), 'glm_result', 'sub')
                # glm_result_path = os.path.join(last_dir, 'glm_result', 'sub')
                new_dir=os.path.join(os.path.dirname(img_dir),'glm_result','sub')
                # 留意：sub文件夹下图片文件夹路径结尾为.json还是txt， 有用 先注释掉 还是response.json
                # img_description_path = os.path.join(glm_result_path, f"{file}.txt", 'rsp.txt')#有用 先注释掉
                
                new_img_description_path=os.path.join(new_dir,'rsp.txt')

                # paths.append((img_path, img_description_path))#有用 先注释掉
                # paths.append(img_path) #输出单独的img_path
                paths.append((img_path,new_img_description_path))
                # 可选：打印路径
                print(f"img_path='{img_path}'")#有用 先注释掉
                # print(f"img_description_path='{img_description_path}'")#有用 先注释掉
                print(f"img_descrp_path={new_img_description_path}")

    return paths  # 返回路径列表


# 输入路径
# root_path = "/home/ubuntu/moe/PaddleOCR_m/constructor/mini/source/20 全球Web3技术产业生态发展报告（2023年）"
root_path="/home/ubuntu/moe/PaddleOCR_m/constructor/result/pdf_test/new_week_in_charts/corporate-commitments-to-nature-have-evolved-since-2022"

result = generate_paths(root_path,'_result.jpg')# 不包含result.jpg

# 结果保存在名为‘week_in_charts/fortune-or-fiction-final-v3.txt’的文件中


# result_dir ="/home/ubuntu/moe/PaddleOCR_m/constructor/mini/find_result"
result_path=os.path.join(root_path, 'find_result.txt')

# 确保目录存在，不存在则创建
os.makedirs(os.path.dirname(result_path), exist_ok=True)

with open(result_path, 'w', encoding='utf-8') as f:
    # 写入路径
    for img_path, img_description_path in result:
        f.write(f"img_path='{img_path}'\n")
        f.write(f"img_description_path='{img_description_path}'\n")
    print('图文对写入成功！')

