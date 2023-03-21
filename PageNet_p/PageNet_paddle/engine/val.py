import os
import paddle
from tqdm import tqdm 

from utils.eval import eval_page_performance
from utils.decode import det_rec_nms, PageDecoder

@paddle.no_grad()
def validate(model, dataloader, converter, cfg):
    model.eval()  

    layout = cfg['POST_PROCESS']['LAYOUT'] if 'LAYOUT' in cfg['POST_PROCESS'] else 'generic'
    page_decoder = PageDecoder(
        se_thres=cfg['POST_PROCESS']['SOL_EOL_CONF_THRES'],  #0.9
        max_steps=cfg['POST_PROCESS']['READ_ORDER_MAX_STEP'],  #20
        layout=layout  #generic
    )

    total_De = 0
    total_Se = 0
    total_Ie = 0
    total_Len = 0
    to_log = ''
    for sample in tqdm(dataloader):
        # images = sample['image'].cuda()
        paddle.set_device("gpu")
        images = sample['image']
        labels = sample['label']
        num_chars = sample['num_char_per_line']
        filename = sample['filename']

        pred_det_rec, pred_read_order, pred_sol, pred_eol = model(images)
        pred_det_rec = pred_det_rec[0].cpu().numpy()
        pred_read_order = pred_read_order[0].cpu().numpy()
        pred_sol = pred_sol[0].cpu().numpy()
        pred_eol = pred_eol[0].cpu().numpy()

        pred_det_rec = det_rec_nms(
            pred_det_rec=pred_det_rec, 
            img_shape=images.shape[-2:],
            dis_weight=cfg['POST_PROCESS']['DIS_WEIGHT'],
            conf_thres=cfg['POST_PROCESS']['CONF_THRES'],
            nms_thres=cfg['POST_PROCESS']['NMS_THRES']
        )

        line_results, _ = page_decoder.decode(
            output=pred_det_rec,
            pred_read=pred_read_order,
            pred_start=pred_sol,
            pred_end=pred_eol,
            img_shape=images.shape[-2:],
        )

        De, Se, Ie, Len, to_log_ = eval_page_performance(pred_det_rec, line_results, labels[0], num_chars[0], converter)
        total_De += De 
        total_Se += Se 
        total_Ie += Ie 
        total_Len += Len 
        total_AR = (total_Len - total_De - total_Se - total_Ie) / total_Len
        total_CR = (total_Len - total_De - total_Se) / total_Len


        to_log += (filename[0] + '\n')
        to_log += to_log_
        to_log += '\ntotally AR: {:6f} CR: {:6f} De: {} Se: {} Ie: {} Len: {}\n'.format(total_AR, total_CR, total_De, total_Se, total_Ie, total_Len)


    log_path = os.path.join(cfg['OUTPUT_FOLDER'], 'val_log.txt')
    with open(log_path, 'w') as f:
        f.write(to_log)  




