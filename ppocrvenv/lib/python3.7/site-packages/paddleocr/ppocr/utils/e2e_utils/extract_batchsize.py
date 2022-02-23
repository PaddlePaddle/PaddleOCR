import paddle
import numpy as np
import copy


def org_tcl_rois(batch_size, pos_lists, pos_masks, label_lists, tcl_bs):
    """
    """
    pos_lists_, pos_masks_, label_lists_ = [], [], []
    img_bs = batch_size
    ngpu = int(batch_size / img_bs)
    img_ids = np.array(pos_lists, dtype=np.int32)[:, 0, 0].copy()
    pos_lists_split, pos_masks_split, label_lists_split = [], [], []
    for i in range(ngpu):
        pos_lists_split.append([])
        pos_masks_split.append([])
        label_lists_split.append([])

    for i in range(img_ids.shape[0]):
        img_id = img_ids[i]
        gpu_id = int(img_id / img_bs)
        img_id = img_id % img_bs
        pos_list = pos_lists[i].copy()
        pos_list[:, 0] = img_id
        pos_lists_split[gpu_id].append(pos_list)
        pos_masks_split[gpu_id].append(pos_masks[i].copy())
        label_lists_split[gpu_id].append(copy.deepcopy(label_lists[i]))
    # repeat or delete
    for i in range(ngpu):
        vp_len = len(pos_lists_split[i])
        if vp_len <= tcl_bs:
            for j in range(0, tcl_bs - vp_len):
                pos_list = pos_lists_split[i][j].copy()
                pos_lists_split[i].append(pos_list)
                pos_mask = pos_masks_split[i][j].copy()
                pos_masks_split[i].append(pos_mask)
                label_list = copy.deepcopy(label_lists_split[i][j])
                label_lists_split[i].append(label_list)
        else:
            for j in range(0, vp_len - tcl_bs):
                c_len = len(pos_lists_split[i])
                pop_id = np.random.permutation(c_len)[0]
                pos_lists_split[i].pop(pop_id)
                pos_masks_split[i].pop(pop_id)
                label_lists_split[i].pop(pop_id)
    # merge
    for i in range(ngpu):
        pos_lists_.extend(pos_lists_split[i])
        pos_masks_.extend(pos_masks_split[i])
        label_lists_.extend(label_lists_split[i])
    return pos_lists_, pos_masks_, label_lists_


def pre_process(label_list, pos_list, pos_mask, max_text_length, max_text_nums,
                pad_num, tcl_bs):
    label_list = label_list.numpy()
    batch, _, _, _ = label_list.shape
    pos_list = pos_list.numpy()
    pos_mask = pos_mask.numpy()
    pos_list_t = []
    pos_mask_t = []
    label_list_t = []
    for i in range(batch):
        for j in range(max_text_nums):
            if pos_mask[i, j].any():
                pos_list_t.append(pos_list[i][j])
                pos_mask_t.append(pos_mask[i][j])
                label_list_t.append(label_list[i][j])
    pos_list, pos_mask, label_list = org_tcl_rois(batch, pos_list_t, pos_mask_t,
                                                  label_list_t, tcl_bs)
    label = []
    tt = [l.tolist() for l in label_list]
    for i in range(tcl_bs):
        k = 0
        for j in range(max_text_length):
            if tt[i][j][0] != pad_num:
                k += 1
            else:
                break
        label.append(k)
    label = paddle.to_tensor(label)
    label = paddle.cast(label, dtype='int64')
    pos_list = paddle.to_tensor(pos_list)
    pos_mask = paddle.to_tensor(pos_mask)
    label_list = paddle.squeeze(paddle.to_tensor(label_list), axis=2)
    label_list = paddle.cast(label_list, dtype='int32')
    return pos_list, pos_mask, label_list, label
