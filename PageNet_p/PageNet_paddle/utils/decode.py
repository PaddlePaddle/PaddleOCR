import numpy as np

def det_rec_nms(pred_det_rec, img_shape, dis_weight, conf_thres, nms_thres):
    pred_box = pred_det_rec[:, :4]
    pred_box[:, 1] *= img_shape[0]
    pred_box[:, 0] *= img_shape[1]
    pred_dis = pred_det_rec[:, 4]
    pred_cls = pred_det_rec[:, 5:]
    pred_cls_prob = np.max(pred_cls, 1)
    pred_det_rec[:, 4] = dis_weight * pred_dis + (1 - dis_weight) * pred_cls_prob
    pred_det_rec = pred_det_rec[pred_det_rec[:, 4] > conf_thres]

    if len(pred_det_rec) == 0:
        keep = []
    else:
        keep = py_cpu_nms(pred_det_rec[:, :5], nms_thres)
    pred_det_rec = pred_det_rec[keep]
    return pred_det_rec

def py_cpu_nms(dets, thresh):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0] - dets[:, 2] / 2
    y1 = dets[:, 1] - dets[:, 3] / 2
    x2 = dets[:, 0] + dets[:, 2] / 2
    y2 = dets[:, 1] + dets[:, 3] / 2
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


class PageDecoder(object):

    def __init__(self, se_thres, max_steps, layout):
        self.se_thres = se_thres
        self.max_steps = max_steps
        self.layout = layout
        self.change_coor = np.array([(0, -1), (1, 0), (0, 1), (-1, 0)])
        self.dir_num = 4
    
    def decode(self, output, pred_read, pred_start, pred_end, img_shape, return_inter_path=False):
        nGh, nGw = pred_read.shape[:2]
        bx = output[:, 0]
        gx = bx / img_shape[1] * nGw
        gx = gx.astype(np.int32)
        gx = gx[:, np.newaxis]
        by = output[:, 1]
        gy = by / img_shape[0] * nGh
        gy = gy.astype(np.int32)
        gy = gy[:, np.newaxis]
        cors = np.concatenate((gx, gy), -1)
        conf = output[:, 4]
        is_start = self._is_start_(cors, pred_start)
        is_end = self._is_end_(cors, pred_end)
        next_indies, paths = self.find_next(cors, pred_read, conf)
        char_paths = self.get_all_paths(next_indies)
        remain_locs = self.filter_out_paths_wo_se(char_paths, is_start, is_end)
        char_paths = char_paths[remain_locs]
        char_paths = self.filter_out_connections(char_paths, next_indies, cors)
        # import pdb;pdb.set_trace()
        remain_locs = self.filter_out_paths_wo_se(char_paths, is_start, is_end)
        char_paths = char_paths[remain_locs]
        line_results = self.split_start_end(char_paths, is_start, is_end)

        if return_inter_path:
            inter_paths = self.get_final_paths(paths, line_results, cors, next_indies)
        else:
            inter_paths = None 
        return line_results, inter_paths

    def get_final_paths(self, paths, line_results, cors, next_indies):
        inter_paths = []
        for line_result in line_results:
            line_result = np.append(line_result, next_indies[line_result[-1]])
            inter_path = []
            for i in range(len(line_result) - 1):
                pre_i = line_result[i]
                next_i = line_result[i+1]
                cur_path = paths[:, pre_i, :][1:]
                valid_path_indies = np.where(np.sum(cur_path, -1) != -2)
                valid_path = cur_path[valid_path_indies]
                if next_i != -1:
                    if (valid_path[-1] == cors[next_i]).all():
                        valid_path = valid_path[:-1]
                inter_path.append(valid_path)
            inter_paths.append(inter_path)
        return inter_paths


    def char_paths2line_results(self, char_paths):
        line_results = []
        for char_path in char_paths:
            line_results.append(char_path[char_path!=-1])
        return line_results

    def split_start_end(self, char_paths, is_start, is_end):
        line_results = []
        for char_path in char_paths:
            prev_index = 0
            prev_flag = 0
            for cur_index in range(len(char_path)):
                cur_cor_index = char_path[cur_index]
                if cur_cor_index == -1:
                    break
                cur_is_start = is_start[cur_cor_index]
                cur_is_end = is_end[cur_cor_index]
                # import pdb;pdb.set_trace()
                if (not cur_is_start) and (not cur_is_end):
                    continue
                if cur_is_start and (prev_flag == 0):
                    prev_index = cur_index
                    prev_flag = 1
                elif cur_is_start and (prev_flag == 2):
                    line_results.append(char_path[prev_index:cur_index])
                    prev_index = cur_index
                    prev_flag = 1
                if cur_is_end:
                    prev_flag = 2
            if prev_index == cur_index:
                line_results.append([char_path[prev_index]])
            else:
                line_results.append(char_path[prev_index:cur_index])
        return line_results
                

    def filter_out_connections(self, char_paths, next_indies, cors):
        bins = np.bincount(next_indies[next_indies!=-1], minlength=len(next_indies))
        dup_connection_indies = np.where(bins > 1)[0]
        # import pdb;pdb.set_trace()
        for dup_connection_index in dup_connection_indies:
            line_indies, inline_indies = np.where(char_paths == dup_connection_index)
            if len(line_indies) <= 1:
                continue
            pre_inline_cors = cors[char_paths[line_indies, inline_indies-1]]
            dup_cor = cors[dup_connection_index]
            if (inline_indies == 1).all():
                dists = np.abs(pre_inline_cors - dup_cor)[:, -1]
                relative_keep_line_index = np.argmin(dists)
                keep_line_index = line_indies[relative_keep_line_index]
                pre_inline_index_kept = char_paths[line_indies, inline_indies-1][relative_keep_line_index]
            else:
                line_indies_filtered = line_indies[inline_indies!=1]
                inline_indies_filtered = inline_indies[inline_indies!=1]
                pre_inline_cors_filtered = pre_inline_cors[inline_indies!=1]
                if len(line_indies_filtered) == 1:
                    relative_keep_line_index = 0
                    keep_line_index = line_indies_filtered[0]
                else:
                    start_inline_cors_filtered = cors[char_paths[line_indies, 0]][inline_indies!=1]
                    tilt_prev = self._tilt_(start_inline_cors_filtered, pre_inline_cors_filtered)
                    tilt_cur = self._tilt_(pre_inline_cors_filtered, dup_cor)
                    relative_keep_line_index = np.argmin(np.abs(tilt_cur - tilt_prev))
                    keep_line_index = line_indies_filtered[relative_keep_line_index]
                pre_inline_index_kept = char_paths[line_indies_filtered, inline_indies_filtered-1][relative_keep_line_index]
            char_paths = self.remove_dup_from_paths(char_paths, line_indies, inline_indies, keep_line_index, pre_inline_index_kept)
        return char_paths

    def remove_dup_from_paths(self, char_paths, line_indies, inline_indies, keep_line_index, pre_inline_index_kept):
        # import pdb;pdb.set_trace()
        del_line_indies = line_indies[line_indies != keep_line_index]
        del_inline_indies = inline_indies[line_indies != keep_line_index]
        for del_line_index, del_inline_index in zip(del_line_indies, del_inline_indies):
            if char_paths[del_line_index, del_inline_index-1] != pre_inline_index_kept:
                char_paths[del_line_index, del_inline_index:] = -1
        return char_paths

    def _tilt_(self, start_cors, end_cors):
        diffs = end_cors - start_cors
        if self.layout == 'vertical':
            return diffs[:, 0] / (diffs[:, 1] + 1e-16)
        else:
            return diffs[:, 1] / (diffs[:, 0] + 1e-16)

    def get_all_paths(self, next_indies, max_length=50):
        prev_indies = self.start_node(next_indies)
        next_indies = np.append(next_indies, -1)
        paths = np.ones((len(prev_indies), max_length)).astype(np.int32) * -1
        paths[:, 0] = prev_indies
        cor_counts = np.zeros((len(prev_indies), len(next_indies)+1)).astype(np.int32)
        cor_counts_indies = np.arange(len(prev_indies))
        cor_counts[cor_counts_indies, prev_indies] += 1
        count = 1
        while True:
            next_indies_ = next_indies[prev_indies]
            cyc_pos = (cor_counts[cor_counts_indies, next_indies_] == 1) & (next_indies_ != -1)
            next_indies_[cyc_pos] = -1
            cor_counts[cor_counts_indies, next_indies_] += 1
            if (next_indies_ == -1).all():
                break
            prev_indies = next_indies_ 
            paths[:, count] = next_indies_ 
            count += 1
            if count == max_length:
                break
        return paths

    def filter_out_paths_wo_se(self, paths, is_start, is_end):
        is_start = np.append(is_start, False)
        is_end = np.append(is_end, False)
        is_start_paths = is_start[paths]
        is_end_paths = is_end[paths]
        is_se_paths = is_start_paths | is_end_paths
        have_se = np.sum(is_se_paths, -1)
        return have_se >= 1

    def _is_start_(self, cors, pred_start):
        is_start = pred_start[cors[:, 1], cors[:, 0]]
        return is_start > self.se_thres

    def _is_end_(self, cors, pred_end):
        is_end = pred_end[cors[:, 1], cors[:, 0]]
        return is_end > self.se_thres

    def find_next(self, cors, pred_read, conf):
        n_samples = len(cors)
        unfound = np.ones(n_samples).astype(np.bool)
        uncycle = np.ones(n_samples).astype(np.bool)
        reach_border = np.zeros(n_samples).astype(np.bool)
        next_indies = np.ones(n_samples).astype(np.int32) * -1
        paths = np.ones((self.max_steps+1, n_samples, 2)).astype(np.int) * -1
        paths[0, :, :] = cors 
        pred_dir = np.argsort(pred_read, -1)[:, :, ::-1]
        for s_i in range(self.max_steps):
            process_locs = unfound & uncycle & (~reach_border)
            next_cors = self.next_step(cors, unfound, uncycle, reach_border, process_locs, paths, pred_dir, 0, s_i)
            reach_border_locs = self.is_reach_border(next_cors, pred_dir.shape[1], pred_dir.shape[0])
            found_indies, cycle_indies, found_next_indies = self.is_found(cors, next_cors, pred_dir, conf, process_locs, s_i)
            self._assign_(unfound, process_locs, found_indies, 0)
            self._assign_(uncycle, process_locs, cycle_indies, 0)
            self._assign_(uncycle, process_locs, self.is_cycle(next_cors), 0)
            self._assign_(reach_border, process_locs, reach_border_locs, 1)
            self._assign_(next_indies, process_locs, found_indies, found_next_indies)
            paths[s_i+1, :, :][process_locs] = next_cors
            if ((process_locs) == False).all():
                return next_indies, paths
        return next_indies, paths 

    def start_node(self, next_indies):
        bins = np.bincount(next_indies[next_indies!=-1], minlength=len(next_indies))
        return np.where(bins==0)[0]

    def _assign_(self, arr, locs, indies, value):
        arr_locs = arr[locs]
        arr_locs[indies] = value
        arr[locs] = arr_locs

    def is_reach_border(self, next_cors, nGw, nGh):
        reach_w_border = (next_cors[:, 0] >= nGw) | (next_cors[:, 0] < 0)
        reach_h_border = (next_cors[:, 1] >= nGh) | (next_cors[:, 1] < 0)
        reach_border = reach_h_border | reach_w_border
        return reach_border

    def next_step(self, cors, unfound, uncycle, reach_border, process_locs, paths, pred_dir, dir_index, cur_step):
        paths = paths[:, process_locs, :]
        unfound = unfound[process_locs]
        uncycle = uncycle[process_locs]
        reach_border = reach_border[process_locs]
        cur_cors = paths[cur_step, :, :]
        cur_dirs = pred_dir[cur_cors[:, 1], cur_cors[:, 0], dir_index]
        cur_changes = self.change_coor[cur_dirs]
        next_cors = cur_cors + cur_changes
        next_cyclic_locs = self.is_cyclic(paths, cur_step, next_cors)
        if len(next_cyclic_locs) == 0:
            return next_cors
        else:
            if dir_index == (self.dir_num - 1):
                next_cors[next_cyclic_locs] = -1
                return next_cors
            cyclic_cors =  cur_cors[next_cyclic_locs]
            cyclic_unfound = unfound[next_cyclic_locs]
            cyclic_uncycle = uncycle[next_cyclic_locs]
            cyclic_reach_border = reach_border[next_cyclic_locs]
            cyclic_process_los = cyclic_unfound & cyclic_uncycle & (~cyclic_reach_border)
            cyclic_paths = paths[:, next_cyclic_locs, :]
            cyclic_next_cors = self.next_step(cyclic_cors, cyclic_unfound, cyclic_uncycle, cyclic_reach_border, cyclic_process_los,
                                    cyclic_paths, pred_dir, dir_index + 1, cur_step)
            next_cors[next_cyclic_locs] = cyclic_next_cors
        return next_cors
    
    def is_found(self, cors, next_cors, pred_dir, conf, process_locs, step_index):
        n_samples = next_cors.size // 2
        cors_repeated = np.repeat(cors[:, np.newaxis, :], n_samples, axis=1)
        dists = np.sum(np.abs(cors_repeated - next_cors), -1)
        indies1, indies2 = np.where(dists <= 1)
        cycle_locs = (np.arange(len(cors))[process_locs][indies2] == indies1)
        if step_index == 0:
            cycle_indies = []
        else:
            cycle_indies = indies2[cycle_locs]
        indies1 = indies1[~cycle_locs]
        indies2 = indies2[~cycle_locs]
        found_indies, indies2_count = np.unique(indies2, return_counts=True)
        if len(found_indies) == len(indies2):
            return indies2, cycle_indies, indies1
        dup_indies2 = found_indies[indies2_count > 1]
        for dup_index in dup_indies2:
            dup_locs = (indies2 == dup_index)
            dup_indies1 = indies1[dup_locs]
            dup_dists = dists[dup_indies1, indies2[dup_locs]]
            if np.min(dup_dists) == 0:
                dup_next_index1 = dup_indies1[dup_dists==0]
                if not isinstance(dup_next_index1, int):
                    dup_next_index1 = dup_next_index1[0]
            else:
                dup_next_cor = next_cors[dup_index]
                dup_next_cor_dir = pred_dir[dup_next_cor[1], dup_next_cor[0]][0]
                dup_next_cor_next = dup_next_cor + self.change_coor[dup_next_cor_dir]
                dup_cors = cors[dup_indies1]
                if dup_next_cor_next in dup_cors:
                    dup_next_index1 = -1
                else:
                    dup_indies1_conf = conf[dup_indies1]
                    dup_next_index1 = dup_indies1[np.argmax(dup_indies1_conf)]
            indies1 = indies1[~dup_locs]
            indies2 = indies2[~dup_locs]
            if dup_next_index1 != -1:
                indies1 = np.append(indies1, dup_next_index1)
                indies2 = np.append(indies2, dup_index)
        return indies2, cycle_indies, indies1
        
    def is_cycle(self, next_cors):
        return np.where(np.sum(next_cors, -1) == -2)

    def is_cyclic(self, paths, cur_step, next_cors):
        prev_paths = paths[:cur_step+1, :, :]
        dists = np.sum(np.abs(prev_paths - next_cors), -1)
        _, cyclic_locs = np.where(dists == 0)
        return cyclic_locs
