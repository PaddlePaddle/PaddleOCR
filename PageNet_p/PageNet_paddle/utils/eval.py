import numpy as np
import edit_distance as ed 


def eval_page_performance(output, line_results, label, char_num, converter, log=True):
    to_log = ''
    
    char_labels = label[:, 0].numpy().astype(np.int32)
    line_split = np.cumsum(char_num.numpy())
    line_split = np.concatenate(([0], line_split)).astype(np.int32)

    line_labels = []
    for l_i in range(len(line_split)-1):
        line_labels.append(char_labels[line_split[l_i]:line_split[l_i+1]])

    De = 0
    Se = 0
    Ie = 0
    Len = 0.0

    pred_cls_labels = []
    for line_result in line_results:
        line_cls_result = output[line_result][:, 5:]
        pred_cls_label = np.argmax(line_cls_result, -1)
        pred_cls_labels.append(pred_cls_label)

    AR_mt = np.zeros((len(line_labels), len(pred_cls_labels)))
    error_mt = np.zeros((len(line_labels), len(pred_cls_labels), 3))
    for i, line_label in enumerate(line_labels):
        for j, pred_cls_label in enumerate(pred_cls_labels):
            dis, error, _, _, _ = cal_distance(line_label, pred_cls_label)
            AR_mt[i, j] = 1- float(dis) / len(line_label)
            error_mt[i, j] = error
    
    label_matched = np.zeros(len(line_labels))
    pred_matched = np.zeros(len(pred_cls_labels))
    sorted_indies = np.argsort(AR_mt.flatten())[::-1]
    for index in sorted_indies:
        if (label_matched == 1).all() or (pred_matched == 1).all():
            break
        i = index // len(pred_cls_labels)
        j = index % len(pred_cls_labels)
        if label_matched[i] == 0 and pred_matched[j] == 0:
            label_matched[i] = 1
            pred_matched[j] = 1
            error = error_mt[i, j]
            De += error[0]
            Se += error[1]
            Ie += error[2]
            Len += len(line_labels[i])
            if log:
                to_log += converter.decode(pred_cls_labels[j])
                to_log += '\t'
                to_log += converter.decode(line_labels[i])
                to_log += '\t'
                AR = (len(line_labels[i]) - error[0] - error[1] - error[2]) / len(line_labels[i]) 
                CR = (len(line_labels[i]) - error[0] - error[1]) / len(line_labels[i])
                to_log += 'AR: {:6f} CR: {:6f} De: {} Se: {} Ie: {} Len: {}'.format(AR, CR, error[0], error[1], error[2], len(line_labels[i]))
                to_log += '\n'
    
    for m_i, match in enumerate(pred_matched):
        if not match:
            Ie += len(pred_cls_labels[m_i])
            if log:
                to_log += converter.decode(pred_cls_labels[m_i])
                to_log += '\n'

    for m_i, match in enumerate(label_matched):
        if not match:
            De += len(line_labels[m_i])
            Len += len(line_labels[m_i])
            if log:
                to_log += '\t'
                to_log += converter.decode(line_labels[m_i])
                to_log += '\n'

    AR = (Len - De - Se - Ie) / Len 
    CR = (Len - De - Se) / Len
    if log:
        to_log += 'AR: {:6f} CR: {:6f} De: {} Se: {} Ie: {} Len: {}'.format(AR, CR, De, Se, Ie, Len)
    
    return De, Se, Ie ,Len, to_log


def cal_distance(label_list, pre_list):
    y = ed.SequenceMatcher(a = label_list, b = pre_list)
    yy = y.get_opcodes()
    insert = 0
    delete = 0
    replace = 0
    label_index = []
    pre_index = []
    consec_eql = []
    # import pdb;pdb.set_trace()
    for i, item in enumerate(yy):
        if item[0] == 'insert':
            insert += item[-1]-item[-2]
        if item[0] == 'delete':
            delete += item[2]-item[1]
        if item[0] == 'replace':
            replace += item[-1]-item[-2]  
        if item[0] == 'equal':
            label_index.append(item[1])
            pre_index.append(item[3])
            if i != (len(yy)-1):
                if yy[i+1][0] == 'equal':
                    consec_eql.append(item[3])
            else:
                consec_eql.append(item[3])
    distance = insert+delete+replace     
    return distance, (delete, replace, insert), label_index, pre_index, consec_eql