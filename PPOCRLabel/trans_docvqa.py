# 通过预定义文件(docvqa_predefine.py)为Label.txt增加关系抽取 (Relation Extraction)，
# 即为标注框增加id、linking以及修改key_cls（question或answer）
import json
import copy
import argparse
from docvqa_predefine import documents


def read_labelfile(filename):
    labeldata = []
    with open(filename) as file:
        lines = file.readlines()
        for line in lines:
            image_path, ocr_info = line.split("\t")
            ocr_infos = json.loads(ocr_info)
            labeldata.append({'image_path': image_path, 'ocr_infos': ocr_infos})
    return labeldata

def rename_key_cls(documents, labeldata):
    """
    根据预定义的question自动匹配answer，并自动命名key_class来建立QA关联关系
    一个question 支持一行多个answer，也支持多行多个answer。
    """
    kiedata = []

    for line in labeldata:
        image_path = line['image_path']
        ocr_infos = line['ocr_infos']
        # 找出文档的header,以此判断是哪个文档
        curr_document = None
        for ocrinfo in ocr_infos:
            for document in documents:
                if ocrinfo['transcription'] in document['headers']:
                    curr_document = document
        if curr_document is None:
            print('错误：%s 未找到文档header，请调整header文本' % image_path)
            continue

        questions = {}
        q = 1
        for ocrinfo in ocr_infos:
            row = 1
            for row_questions in curr_document['questions']:
                col = 1
                col_count = len(row_questions)
                for one_quesion in row_questions:
                    if ocrinfo['transcription'] == one_quesion:
                        ocrinfo['key_cls'] = 'question_' + str(q)
                        ocrinfo['construct'] = [row, col, col_count]  # 这里临时保存当前问题所在的表格的位置
                        questions['q_' + str(row) + '_' + str(col)] = ocrinfo
                        q += 1
                    col += 1
                row += 1

        # 为question匹配answer（支持多行多个answer，根据标注框top和left优先排序）,other,header 保留
        for ocrinfo in ocr_infos:
            for bianhao in questions:
                question = questions[bianhao]
                question_points = question['points']
                [q_row, q_col, q_col_count] = question['construct']
                label_points = ocrinfo['points']

                if ocrinfo['transcription'] in curr_document['headers']:
                    ocrinfo['key_cls'] = "header"
                else:
                    row_height = 1.0
                    row_valign = 'middle'
                    style = curr_document['style']
                    if question['transcription'] in style:
                        if "row_height" in style[question['transcription']]:
                            row_height = style[question['transcription']]["row_height"]
                        if "valign" in style[question['transcription']]:
                            row_valign = style[question['transcription']]["valign"]
                    # 额外增加answer的判定范围,基于Table Row去判断
                    q_lineheight = abs(question_points[2][1] - question_points[1][1]) * (row_height - 1.0) / 2

                    if row_valign == 'top':
                        question_line_min = question_points[1][1]
                        question_line_max = question_points[2][1] + q_lineheight * 2
                    elif row_valign == 'bottom':
                        question_line_min = question_points[1][1] - q_lineheight * 2
                        question_line_max = question_points[2][1]
                    else:
                        question_line_min = question_points[1][1] - q_lineheight
                        question_line_max = question_points[2][1] + q_lineheight

                    # 基于中心点来判断，能更好的适配超过边界的标注框
                    bbox_center_y = label_points[0][1] + (label_points[3][1] - label_points[0][1])
                    if label_points[0][0] > question_points[1][0] and question_line_min < bbox_center_y < question_line_max:
                        if q_col < q_col_count:
                            next_col = q_col + 1
                            bh = 'q_' + str(q_row) + '_' + str(next_col)  # 编号，以行和列序号标记
                            if bh in questions:  # 如果本行存在next question，那么答案的右边不能超过它的左边
                                next_question = questions[bh]
                                if label_points[1][0] < next_question['points'][0][0]:
                                    answer_name = copy.deepcopy(question['key_cls']).replace('question_', 'answer_')
                                    ocrinfo['key_cls'] = answer_name  # 答案辅助标签，与question后的数字对应
                        elif q_col == q_col_count:  # 本行最后一个问题的答案不受限制
                            answer_name = copy.deepcopy(question['key_cls']).replace('question_', 'answer_')
                            ocrinfo['key_cls'] = answer_name
        # 移除辅助信息
        for ocrinfo in ocr_infos:
            if 'key_cls' not in ocrinfo or ocrinfo['key_cls'] in ["o", "None"]:
                # None、other
                ocrinfo["key_cls"] = "other"
            if "construct" in ocrinfo:
                del ocrinfo["construct"]

        kiedata.append({'image_path': image_path, 'ocr_infos': ocr_infos})

    return kiedata


def trans_docvqa(kieData, outputfile):
    """
    根据key_cls的命名规则创建linking并为question和answer生成唯一ID
    说明：PPOCRLabel v3在原有数据格式上增加了ID和linking，作为一种新的doc-vqa格式进行训练。
    :param outputfile: 输出doc-vqa文件路径(也可覆盖Label.txt文件)
    """
    content = ""
    for line in kieData:
        image_path = line['image_path']
        ocr_infos = line['ocr_infos']
        qa = copy.deepcopy(ocr_infos)
        links = {}
        ocr_id = 1
        # 分别提取qa
        q = []
        a = []
        for ocr_info in qa:
            ocr_info['id'] = ocr_id
            if ocr_info['key_cls'][0:8] == "question":
                question_id = ocr_info['key_cls'].replace("question_", "")
                q.append([question_id, ocr_id])
            elif ocr_info['key_cls'][0:6] == "answer":
                question_answer_id = ocr_info['key_cls'].replace("answer_", "")
                a.append([question_answer_id, ocr_id])

            ocr_id = ocr_id + 1
        # qa关系建立
        for cls_id, ocrid in q:
            link = []
            for cls_id2, ocrid2 in a:
                if cls_id == cls_id2:
                    link.append([ocrid, ocrid2])
            links[cls_id] = link

        newocrinfos = []
        ocr_id = 1

        for ocr_info in qa:
            question_id = 0
            linking = []
            if ocr_info['key_cls'][0:8] == "question":
                question_id = ocr_info['key_cls'].replace("question_", "")
                new_key_cls = 'question'
            elif ocr_info['key_cls'][0:6] == "answer":
                question_id = ocr_info['key_cls'].replace("answer_", "")
                new_key_cls = 'answer'
            elif ocr_info['key_cls'] == 'header':
                new_key_cls = "header"
            else:
                new_key_cls = "other"

            if question_id in links:
                linking = links[question_id]

            newocrinfos.append({"id": ocr_id, "transcription": ocr_info['transcription'], "points": ocr_info['points'],
                                "key_cls": new_key_cls, "linking": linking})
            ocr_id = ocr_id + 1
        content += image_path + "\t" + json.dumps(newocrinfos, ensure_ascii=False) + "\n"

    with open(outputfile, 'w+', encoding='utf-8') as f_vqa:
        f_vqa.writelines(content)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--LabelFileName",
        type=str,
        default="Label.txt",
        help="the name of the detection annotation file（标记结果文件Label.txt的路径）")

    parser.add_argument(
        "--outputFileName",
        type=str,
        default="Label_docvqa.txt",
        help="the name of the document-vqa file（输出DOC-VQA格式的文件名）")

    args = parser.parse_args()

    labeldata = read_labelfile(args.LabelFileName)
    print("读取%s文件完成" % args.LabelFileName)
    kie_data = rename_key_cls(documents, labeldata)
    print("重命名key_cls完成")
    trans_docvqa(kie_data, args.outputFileName)
    print("创建RE（linking）完成")
