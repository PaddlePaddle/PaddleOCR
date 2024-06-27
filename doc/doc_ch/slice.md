# 切片操作

如果希望运行 PaddleOCR 处理一张非常大的图像或文档，对其进行检测和识别，可以使用切片操作，如下所示：

```python
ocr_inst = PaddleOCR(**ocr_settings)
results = ocr_inst.ocr(img, det=True, rec=True, slice=slice, cls=False, bin=False, inv=False, alpha_color=False)
```

其中，
`slice = {'horizontal_stride': h_stride, 'vertical_stride': v_stride, 'merge_x_thres': x_thres, 'merge_y_thres': y_thres}`

这里的 `h_stride`、`v_stride`、`x_thres` 和 `y_thres` 是用户可配置的参数，需要手动设置。切片操作符的工作原理是，在大图像上运行一个滑动窗口，创建图像的切片，并在这些切片上运行 OCR 算法。

然后将这些切片级别的零散结果合并，生成图像级别的检测和识别结果。水平和垂直步幅不能低于一定限度，因为过低的值会产生太多切片，导致计算结果非常耗时。例如，对于尺寸为 6616x14886 的图像，推荐使用以下参数：

```python
slice = {'horizontal_stride': 300, 'vertical_stride': 500, 'merge_x_thres': 50, 'merge_y_thres': 35}
```

所有边界框接近 `merge_x_thres` 和 `merge_y_thres` 的切片级检测结果将被合并在一起。
