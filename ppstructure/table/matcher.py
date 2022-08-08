import json
from ppstructure.table.table_master_match import deal_eb_token, deal_bb


def distance(box_1, box_2):
    x1, y1, x2, y2 = box_1
    x3, y3, x4, y4 = box_2
    dis = abs(x3 - x1) + abs(y3 - y1) + abs(x4 - x2) + abs(y4 - y2)
    dis_2 = abs(x3 - x1) + abs(y3 - y1)
    dis_3 = abs(x4 - x2) + abs(y4 - y2)
    return dis + min(dis_2, dis_3)


def compute_iou(rec1, rec2):
    """
    computing IoU
    :param rec1: (y0, x0, y1, x1), which reflects
            (top, left, bottom, right)
    :param rec2: (y0, x0, y1, x1)
    :return: scala value of IoU
    """
    # computing area of each rectangles
    S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
    S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])

    # computing the sum_area
    sum_area = S_rec1 + S_rec2

    # find the each edge of intersect rectangle
    left_line = max(rec1[1], rec2[1])
    right_line = min(rec1[3], rec2[3])
    top_line = max(rec1[0], rec2[0])
    bottom_line = min(rec1[2], rec2[2])

    # judge if there is an intersect
    if left_line >= right_line or top_line >= bottom_line:
        return 0.0
    else:
        intersect = (right_line - left_line) * (bottom_line - top_line)
        return (intersect / (sum_area - intersect)) * 1.0


def matcher_merge(ocr_bboxes, pred_bboxes):
    all_dis = []
    ious = []
    matched = {}
    for i, gt_box in enumerate(ocr_bboxes):
        distances = []
        for j, pred_box in enumerate(pred_bboxes):
            # compute l1 distence and IOU between two boxes
            distances.append((distance(gt_box, pred_box),
                              1. - compute_iou(gt_box, pred_box)))
        sorted_distances = distances.copy()
        # select nearest cell
        sorted_distances = sorted(
            sorted_distances, key=lambda item: (item[1], item[0]))
        if distances.index(sorted_distances[0]) not in matched.keys():
            matched[distances.index(sorted_distances[0])] = [i]
        else:
            matched[distances.index(sorted_distances[0])].append(i)
    return matched  #, sum(ious) / len(ious)


def complex_num(pred_bboxes):
    complex_nums = []
    for bbox in pred_bboxes:
        distances = []
        temp_ious = []
        for pred_bbox in pred_bboxes:
            if bbox != pred_bbox:
                distances.append(distance(bbox, pred_bbox))
                temp_ious.append(compute_iou(bbox, pred_bbox))
        complex_nums.append(temp_ious[distances.index(min(distances))])
    return sum(complex_nums) / len(complex_nums)


def get_rows(pred_bboxes):
    pre_bbox = pred_bboxes[0]
    res = []
    step = 0
    for i in range(len(pred_bboxes)):
        bbox = pred_bboxes[i]
        if bbox[1] - pre_bbox[1] > 2 or bbox[0] - pre_bbox[0] < 0:
            break
        else:
            res.append(bbox)
            step += 1
    for i in range(step):
        pred_bboxes.pop(0)
    return res, pred_bboxes


def refine_rows(pred_bboxes):  # 微调整行的框，使在一条水平线上
    ys_1 = []
    ys_2 = []
    for box in pred_bboxes:
        ys_1.append(box[1])
        ys_2.append(box[3])
    min_y_1 = sum(ys_1) / len(ys_1)
    min_y_2 = sum(ys_2) / len(ys_2)
    re_boxes = []
    for box in pred_bboxes:
        box[1] = min_y_1
        box[3] = min_y_2
        re_boxes.append(box)
    return re_boxes


def matcher_refine_row(gt_bboxes, pred_bboxes):
    before_refine_pred_bboxes = pred_bboxes.copy()
    pred_bboxes = []
    while (len(before_refine_pred_bboxes) != 0):
        row_bboxes, before_refine_pred_bboxes = get_rows(
            before_refine_pred_bboxes)
        print(row_bboxes)
        pred_bboxes.extend(refine_rows(row_bboxes))
    all_dis = []
    ious = []
    matched = {}
    for i, gt_box in enumerate(gt_bboxes):
        distances = []
        #temp_ious = []
        for j, pred_box in enumerate(pred_bboxes):
            distances.append(distance(gt_box, pred_box))
            #temp_ious.append(compute_iou(gt_box, pred_box))
        #all_dis.append(min(distances))
        #ious.append(temp_ious[distances.index(min(distances))])
        if distances.index(min(distances)) not in matched.keys():
            matched[distances.index(min(distances))] = [i]
        else:
            matched[distances.index(min(distances))].append(i)
    return matched  #, sum(ious) / len(ious)


#先挑选出一行，再进行匹配
def matcher_structure_1(gt_bboxes, pred_bboxes_rows, pred_bboxes):
    gt_box_index = 0
    delete_gt_bboxes = gt_bboxes.copy()
    match_bboxes_ready = []
    matched = {}
    while (len(delete_gt_bboxes) != 0):
        row_bboxes, delete_gt_bboxes = get_rows(delete_gt_bboxes)
        row_bboxes = sorted(row_bboxes, key=lambda key: key[0])
        if len(pred_bboxes_rows) > 0:
            match_bboxes_ready.extend(pred_bboxes_rows.pop(0))
        print(row_bboxes)
        for i, gt_box in enumerate(row_bboxes):
            #print(gt_box)
            pred_distances = []
            distances = []
            for pred_bbox in pred_bboxes:
                pred_distances.append(distance(gt_box, pred_bbox))
            for j, pred_box in enumerate(match_bboxes_ready):
                distances.append(distance(gt_box, pred_box))
            index = pred_distances.index(min(distances))
            #print('index', index)
            if index not in matched.keys():
                matched[index] = [gt_box_index]
            else:
                matched[index].append(gt_box_index)
            gt_box_index += 1
    return matched


def matcher_structure(gt_bboxes, pred_bboxes_rows, pred_bboxes):
    '''
    gt_bboxes: 排序后
    pred_bboxes: 
    '''
    pre_bbox = gt_bboxes[0]
    matched = {}
    match_bboxes_ready = []
    match_bboxes_ready.extend(pred_bboxes_rows.pop(0))
    for i, gt_box in enumerate(gt_bboxes):

        pred_distances = []
        for pred_bbox in pred_bboxes:
            pred_distances.append(distance(gt_box, pred_bbox))
        distances = []
        gap_pre = gt_box[1] - pre_bbox[1]
        gap_pre_1 = gt_box[0] - pre_bbox[2]
        #print(gap_pre, len(pred_bboxes_rows))
        if (gap_pre_1 < 0 and len(pred_bboxes_rows) > 0):
            match_bboxes_ready.extend(pred_bboxes_rows.pop(0))
        if len(pred_bboxes_rows) == 1:
            match_bboxes_ready.extend(pred_bboxes_rows.pop(0))
        if len(match_bboxes_ready) == 0 and len(pred_bboxes_rows) > 0:
            match_bboxes_ready.extend(pred_bboxes_rows.pop(0))
        if len(match_bboxes_ready) == 0 and len(pred_bboxes_rows) == 0:
            break
        #print(match_bboxes_ready)
        for j, pred_box in enumerate(match_bboxes_ready):
            distances.append(distance(gt_box, pred_box))
        index = pred_distances.index(min(distances))
        #print(gt_box, index)
        #match_bboxes_ready.pop(distances.index(min(distances)))
        print(gt_box, match_bboxes_ready[distances.index(min(distances))])
        if index not in matched.keys():
            matched[index] = [i]
        else:
            matched[index].append(i)
        pre_bbox = gt_box
    return matched


class TableMatch:
    def __init__(self, filter_ocr_result=False, use_master=False):
        self.filter_ocr_result = filter_ocr_result
        self.use_master = use_master

    def __call__(self, structure_res, dt_boxes, rec_res):
        pred_structures, pred_bboxes = structure_res
        if self.filter_ocr_result:
            dt_boxes, rec_res = self.filter_ocr_result(pred_bboxes, dt_boxes,
                                                       rec_res)
        matched_index = self.match_result(dt_boxes, pred_bboxes)
        if self.use_master:
            pred_html, pred = self.get_pred_html_master(pred_structures,
                                                        matched_index, rec_res)
        else:
            pred_html, pred = self.get_pred_html(pred_structures, matched_index,
                                                 rec_res)
        return pred_html

    def match_result(self, dt_boxes, pred_bboxes):
        matched = {}
        for i, gt_box in enumerate(dt_boxes):
            # gt_box = [np.min(gt_box[:, 0]), np.min(gt_box[:, 1]), np.max(gt_box[:, 0]), np.max(gt_box[:, 1])]
            distances = []
            for j, pred_box in enumerate(pred_bboxes):
                distances.append((distance(gt_box, pred_box),
                                  1. - compute_iou(gt_box, pred_box)
                                  ))  # 获取两两cell之间的L1距离和 1- IOU
            sorted_distances = distances.copy()
            # 根据距离和IOU挑选最"近"的cell
            sorted_distances = sorted(
                sorted_distances, key=lambda item: (item[1], item[0]))
            if distances.index(sorted_distances[0]) not in matched.keys():
                matched[distances.index(sorted_distances[0])] = [i]
            else:
                matched[distances.index(sorted_distances[0])].append(i)
        return matched

    def get_pred_html(self, pred_structures, matched_index, ocr_contents):
        end_html = []
        td_index = 0
        for tag in pred_structures:
            if '</td>' in tag:
                if '<td></td>' == tag:
                    end_html.extend('<td>')
                if td_index in matched_index.keys():
                    b_with = False
                    if '<b>' in ocr_contents[matched_index[td_index][
                            0]] and len(matched_index[td_index]) > 1:
                        b_with = True
                        end_html.extend('<b>')
                    for i, td_index_index in enumerate(matched_index[td_index]):
                        content = ocr_contents[td_index_index][0]
                        if len(matched_index[td_index]) > 1:
                            if len(content) == 0:
                                continue
                            if content[0] == ' ':
                                content = content[1:]
                            if '<b>' in content:
                                content = content[3:]
                            if '</b>' in content:
                                content = content[:-4]
                            if len(content) == 0:
                                continue
                            if i != len(matched_index[
                                    td_index]) - 1 and ' ' != content[-1]:
                                content += ' '
                        end_html.extend(content)
                    if b_with:
                        end_html.extend('</b>')
                if '<td></td>' == tag:
                    end_html.append('</td>')
                else:
                    end_html.append(tag)
                td_index += 1
            else:
                end_html.append(tag)
        return ''.join(end_html), end_html

    def get_pred_html_master(self, pred_structures, matched_index,
                             ocr_contents):
        end_html = []
        td_index = 0
        for token in pred_structures:
            if '</td>' in token:
                txt = ''
                b_with = False
                if td_index in matched_index.keys():
                    if '<b>' in ocr_contents[matched_index[td_index][
                            0]] and len(matched_index[td_index]) > 1:
                        b_with = True
                    for i, td_index_index in enumerate(matched_index[td_index]):
                        content = ocr_contents[td_index_index][0]
                        if len(matched_index[td_index]) > 1:
                            if len(content) == 0:
                                continue
                            if content[0] == ' ':
                                content = content[1:]
                            if '<b>' in content:
                                content = content[3:]
                            if '</b>' in content:
                                content = content[:-4]
                            if len(content) == 0:
                                continue
                            if i != len(matched_index[
                                    td_index]) - 1 and ' ' != content[-1]:
                                content += ' '
                        txt += content
                if b_with:
                    txt = '<b>{}</b>'.format(txt)
                if '<td></td>' == token:
                    token = '<td>{}</td>'.format(txt)
                else:
                    token = '{}</td>'.format(txt)
                td_index += 1
            token = deal_eb_token(token)
            end_html.append(token)
        html = ''.join(end_html)
        html = deal_bb(html)
        return html, end_html

    def filter_ocr_result(self, pred_bboxes, dt_boxes, rec_res):
        y1 = pred_bboxes[:, 1::2].min()
        new_dt_boxes = []
        new_rec_res = []

        for box, rec in zip(dt_boxes, rec_res):
            if np.max(box[1::2]) < y1:
                continue
            new_dt_boxes.append(box)
            new_rec_res.append(rec)
        return new_dt_boxes, new_rec_res
