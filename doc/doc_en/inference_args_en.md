# PaddleOCR Model Inference Parameter Explanation

When using PaddleOCR for model inference, you can customize the modification parameters to modify the model, data, preprocessing, postprocessing, etc. (parameter file: [utility.py](../../tools/infer/utility.py))，The detailed parameter explanation is as follows:

* Global parameters

| parameters | type | default | implication |
| :--: | :--: | :--: | :--: |
|  image_dir | str | None, must be specified explicitly | Image or folder path |
|  page_num | int | 0 | Valid when the input type is pdf file, specify to predict the previous page_num pages, all pages are predicted by default |
|  vis_font_path | str | "./doc/fonts/simfang.ttf" | font path for visualization |
|  drop_score | float | 0.5 | Results with a recognition score less than this value will be discarded and will not be returned as results |
|  use_pdserving | bool | False | Whether to use Paddle Serving for prediction |
|  warmup | bool | False | Whether to enable warmup, this method can be used when statistical prediction time |
|  draw_img_save_dir | str | "./inference_results" | The saving folder of the system's tandem prediction OCR results |
|  save_crop_res | bool | False  | Whether to save the recognized text image for OCR |
|  crop_res_save_dir | str | "./output" | Save the text image path recognized by OCR |
|  use_mp | bool | False | Whether to enable multi-process prediction  |
|  total_process_num | int | 6 | The number of processes, which takes effect when `use_mp` is `True` |
|  process_id | int | 0 | The id number of the current process, no need to modify it yourself |
|  benchmark | bool | False | Whether to enable benchmark, and make statistics on prediction speed, memory usage, etc. |
|  save_log_path | str | "./log_output/" | Folder where log results are saved when `benchmark` is enabled |
|  show_log | bool | True | Whether to show the log information in the inference |
|  use_onnx | bool | False | Whether to enable onnx prediction |


* Prediction engine related parameters

| parameters | type | default | implication |
| :--: | :--: | :--: | :--: |
|  use_gpu | bool | True | Whether to use GPU for prediction |
|  ir_optim | bool | True | Whether to analyze and optimize the calculation graph. The prediction process can be accelerated when `ir_optim` is enabled |
|  use_tensorrt | bool | False | Whether to enable tensorrt |
|  min_subgraph_size | int | 15 | The minimum subgraph size in tensorrt. When the size of the subgraph is greater than this value, it will try to use the trt engine to calculate the subgraph. |
|  precision | str | fp32 | The precision of prediction, supports `fp32`, `fp16`, `int8` |
|  enable_mkldnn | bool | True | Whether to enable mkldnn |
|  cpu_threads | int | 10 | When mkldnn is enabled, the number of threads predicted by the cpu |

* Text detection model related parameters

| parameters | type | default | implication |
| :--: | :--: | :--: | :--: |
|  det_algorithm | str | "DB" | Text detection algorithm name, currently supports `DB`, `EAST`, `SAST`, `PSE`, `DB++`, `FCE` |
|  det_model_dir | str | xx | Detection inference model paths |
|  det_limit_side_len | int | 960 | image side length limit |
|  det_limit_type | str | "max" | The side length limit type, currently supports `min`and `max`. `min` means to ensure that the shortest side of the image is not less than `det_limit_side_len`, `max` means to ensure that the longest side of the image is not greater than `det_limit_side_len` |

The relevant parameters of the DB algorithm are as follows

| parameters | type | default | implication |
| :--: | :--: | :--: | :--: |
|  det_db_thresh | float | 0.3 | In the probability map output by DB, only pixels with a score greater than this threshold will be considered as text pixels |
|  det_db_box_thresh | float | 0.6 | Within the detection box, when the average score of all pixels is greater than the threshold, the result will be considered as a text area |
|  det_db_unclip_ratio | float | 1.5 | The expansion factor of the `Vatti clipping` algorithm, which is used to expand the text area |
|  max_batch_size | int | 10 | max batch size |
|  use_dilation | bool | False | Whether to inflate the segmentation results to obtain better detection results |
|  det_db_score_mode | str | "fast" | DB detection result score calculation method, supports `fast` and `slow`, `fast` calculates the average score according to all pixels within the bounding rectangle of the polygon, `slow` calculates the average score according to all pixels within the original polygon, The calculation speed is relatively slower, but more accurate. |

The relevant parameters of the EAST algorithm are as follows

| parameters | type | default | implication |
| :--: | :--: | :--: | :--: |
|  det_east_score_thresh | float | 0.8 | Threshold for score map in EAST postprocess |
|  det_east_cover_thresh | float | 0.1 | Average score threshold for text boxes in EAST postprocess |
|  det_east_nms_thresh | float | 0.2 | Threshold of nms in EAST postprocess |

The relevant parameters of the SAST algorithm are as follows

| parameters | type | default | implication |
| :--: | :--: | :--: | :--: |
|  det_sast_score_thresh | float | 0.5 | Score thresholds in SAST postprocess |
|  det_sast_nms_thresh | float | 0.5 | Thresholding of nms in SAST postprocess |
|  det_box_type | str | 'quad' | Whether polygon detection, curved text scene (such as Total-Text) is set to 'poly' |

The relevant parameters of the PSE algorithm are as follows

| parameters | type | default | implication |
| :--: | :--: | :--: | :--: |
|  det_pse_thresh | float | 0.0 | Threshold for binarizing the output image |
|  det_pse_box_thresh | float | 0.85 | Threshold for filtering boxes, below this threshold is discarded |
|  det_pse_min_area | float | 16 | The minimum area of the box, below this threshold is discarded |
|  det_box_type | str | "quad" | The type of the returned box, quad: four point coordinates, poly: all point coordinates of the curved text |
|  det_pse_scale | int | 1 | The ratio of the input image relative to the post-processed image, such as an image of `640*640`, the network output is `160*160`, and when the scale is 2, the shape of the post-processed image is `320*320`. Increasing this value can speed up the post-processing speed, but it will bring about a decrease in accuracy |

* Text recognition model related parameters

| parameters | type | default | implication |
| :--: | :--: | :--: | :--: |
|  rec_algorithm | str | "CRNN" | Text recognition algorithm name, currently supports `CRNN`, `SRN`, `RARE`, `NETR`, `SAR`, `ViTSTR`, `ABINet`, `VisionLAN`, `SPIN`, `RobustScanner`, `SVTR`, `SVTR_LCNet` |
|  rec_model_dir | str | None, it is required if using the recognition model | recognition inference model paths |
|  rec_image_shape | str | "3,48,320" ] | Image size at the time of recognition |
|  rec_batch_num | int | 6 | batch size |
|  max_text_length | int | 25 | The maximum length of the recognition result, valid in `SRN` |
|  rec_char_dict_path | str | "./ppocr/utils/ppocr_keys_v1.txt" | character dictionary file |
|  use_space_char | bool | True | Whether to include spaces, if `True`, the `space` character will be added at the end of the character dictionary |


* End-to-end text detection and recognition model related parameters

| parameters | type | default | implication |
| :--: | :--: | :--: | :--: |
|  e2e_algorithm | str | "PGNet" | End-to-end algorithm name, currently supports `PGNet` |
|  e2e_model_dir | str | None, it is required if using the end-to-end model | end-to-end model inference model path |
|  e2e_limit_side_len | int | 768 | End-to-end input image side length limit |
|  e2e_limit_type | str | "max" | End-to-end side length limit type, currently supports `min` and `max`. `min` means to ensure that the shortest side of the image is not less than `e2e_limit_side_len`, `max` means to ensure that the longest side of the image is not greater than `e2e_limit_side_len` |
|  e2e_pgnet_score_thresh | float | 0.5 | End-to-end score threshold, results below this threshold are discarded |
|  e2e_char_dict_path | str | "./ppocr/utils/ic15_dict.txt" | Recognition dictionary file path |
|  e2e_pgnet_valid_set | str | "totaltext" | The name of the validation set, currently supports `totaltext`, `partvgg`, the post-processing methods corresponding to different data sets are different, and it can be consistent with the training process |
|  e2e_pgnet_mode | str | "fast" | PGNet's detection result score calculation method, supports `fast` and `slow`, `fast` calculates the average score according to all pixels within the bounding rectangle of the polygon, `slow` calculates the average score according to all pixels within the original polygon, The calculation speed is relatively slower, but more accurate. |


* Angle classifier model related parameters

| parameters | type | default | implication |
| :--: | :--: | :--: | :--: |
|  use_angle_cls | bool | False | whether to use an angle classifier |
|  cls_model_dir | str | None, if you need to use, you must specify the path explicitly | angle classifier inference model path |
|  cls_image_shape | str | "3,48,192" | prediction shape |
|  label_list | list | ['0', '180'] | The angle value corresponding to the class id |
|  cls_batch_num | int | 6 | batch size |
|  cls_thresh | float | 0.9 | Prediction threshold, when the model prediction result is 180 degrees, and the score is greater than the threshold, the final prediction result is considered to be 180 degrees and needs to be flipped |


* OCR image preprocessing parameters

| parameters | type | default | implication |
| :--: | :--: | :--: | :--: |
|  invert | bool | False | whether to invert image before processing |
|  binarize | bool | False | whether to threshold binarize image before processing |
|  alphacolor | tuple | "255,255,255" | Replacement color for the alpha channel, if the latter is present; R,G,B integers |
