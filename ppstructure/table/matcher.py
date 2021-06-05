import json
def distance(box_1, box_2):
        x1, y1, x2, y2 = box_1
        x3, y3, x4, y4 = box_2
        # min_x = (x1 + x2) / 2
        # min_y = (y1 + y2) / 2
        # max_x = (x3 + x4) / 2
        # max_y = (y3 + y4) / 2
        dis = abs(x3 - x1) + abs(y3 - y1) + abs(x4- x2) + abs(y4 - y2)
        dis_2 = abs(x3 - x1) + abs(y3 - y1)
        dis_3 = abs(x4- x2) + abs(y4 - y2)
        #dis = pow(min_x - max_x, 2) + pow(min_y - max_y, 2) + pow(x3 - x1, 2) + pow(y3 - y1, 2) + pow(x4- x2, 2) + pow(y4 - y2, 2) + abs(x3 - x1) + abs(y3 - y1) + abs(x4- x2) + abs(y4 - y2)
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
    rec1, rec2 = rec1 * 1000, rec2 * 1000
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
        return 0
    else:
        intersect = (right_line - left_line) * (bottom_line - top_line)
        return (intersect / (sum_area - intersect))*1.0
 


def matcher_merge(ocr_bboxes, pred_bboxes): # ocr_bboxes: OCR pred_bboxes：端到端
    all_dis = []
    ious = []
    matched = {}
    for i, gt_box in enumerate(ocr_bboxes):
        distances = []
        for j, pred_box in enumerate(pred_bboxes):
            distances.append((distance(gt_box, pred_box), 1. - compute_iou(gt_box, pred_box))) #获取两两cell之间的L1距离和 1- IOU
        sorted_distances = distances.copy()
        # 根据距离和IOU挑选最"近"的cell
        sorted_distances = sorted(sorted_distances, key = lambda item: (item[1], item[0])) 
        if distances.index(sorted_distances[0]) not in matched.keys(): 
            matched[distances.index(sorted_distances[0])] = [i]
        else:
            matched[distances.index(sorted_distances[0])].append(i)
    return matched#, sum(ious) / len(ious)
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
def refine_rows(pred_bboxes): # 微调整行的框，使在一条水平线上
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
    while(len(before_refine_pred_bboxes) != 0):
        row_bboxes, before_refine_pred_bboxes = get_rows(before_refine_pred_bboxes)
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
    return matched#, sum(ious) / len(ious)



#先挑选出一行，再进行匹配
def matcher_structure_1(gt_bboxes, pred_bboxes_rows, pred_bboxes):
    gt_box_index = 0
    delete_gt_bboxes = gt_bboxes.copy()
    match_bboxes_ready = []
    matched = {}
    while(len(delete_gt_bboxes) != 0):
        row_bboxes, delete_gt_bboxes = get_rows(delete_gt_bboxes)
        row_bboxes = sorted(row_bboxes, key = lambda key: key[0])
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
