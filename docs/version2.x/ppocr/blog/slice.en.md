---
comments: true
---


# Slice Operator

If you have a very large image/document that you would like to run PaddleOCR (detection and recognition) on, you can use the slice operation as follows:

`ocr_inst  =  PaddleOCR(**ocr_settings)`
`results  =  ocr_inst.ocr(img, det=True,rec=True, slice=slice, cls=False,bin=False,inv=False,alpha_color=False)`

where
`slice  = {'horizontal_stride': h_stride, 'vertical_stride':v_stride, 'merge_x_thres':x_thres, 'merge_y_thres': y_thres}`

Here, `h_stride`, `v_stride`, `x_thres`, and `y_thres` are user-configurable values and need to be set manually. The way the `slice` operator works is that it runs a sliding window across the large input image, creating slices of it and runs the OCR algorithms on it.

The fragmented slice-level results are then merged together to output image-level detection and recognition results. The horizontal and vertical strides cannot be lower than a certain limit (as too low values would create so many slices it would be very computationally expensive to get results for each of them). However, as an example the recommended values for an image with dimensions 6616x14886 would be as follows.

`slice = {'horizontal_stride': 300, 'vertical_stride':500, 'merge_x_thres':50, 'merge_y_thres': 35}`

All slice-level detections with bounding boxes as close as `merge_x_thres` and `merge_y_thres` will be merged together.
