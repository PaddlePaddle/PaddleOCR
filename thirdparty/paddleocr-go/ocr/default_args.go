package ocr

var (
	defaultArgs = map[string]interface{}{
		"use_gpu":         true,
		"ir_optim":        true,
		"enable_mkldnn":   false,
		"use_tensorrt":    false,
		"num_cpu_threads": 6,
		"gpu_id":          0,
		"gpu_mem":         2000,

		"det_algorithm":    "DB",
		"det_model_dir":    "https://paddleocr.bj.bcebos.com/20-09-22/mobile/det/ch_ppocr_mobile_v1.1_det_infer.tar",
		"det_max_side_len": 960,

		"det_db_thresh":       0.3,
		"det_db_box_thresh":   0.5,
		"det_db_unclip_ratio": 2.0,

		"det_east_score_thresh": 0.8,
		"det_east_cover_thresh": 0.1,
		"det_east_nms_thresh":   0.2,

		"rec_algorithm":      "CRNN",
		"rec_model_dir":      "https://paddleocr.bj.bcebos.com/20-09-22/mobile/rec/ch_ppocr_mobile_v1.1_rec_infer.tar",
		"rec_image_shape":    []interface{}{3, 32, 320},
		"rec_char_type":      "ch",
		"rec_batch_num":      30,
		"max_text_length":    25,
		"rec_char_dict_path": "https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/develop/ppocr/utils/ppocr_keys_v1.txt",
		"use_space_char":     true,

		"use_angle_cls":   false,
		"cls_model_dir":   "https://paddleocr.bj.bcebos.com/20-09-22/cls/ch_ppocr_mobile_v1.1_cls_infer.tar",
		"cls_image_shape": []interface{}{3, 48, 192},
		"label_list":      []interface{}{"0", "180"},
		"cls_batch_num":   30,
		"cls_thresh":      0.9,

		"lang": "ch",
		"det":  true,
		"rec":  true,
		"cls":  false,
	}
)
