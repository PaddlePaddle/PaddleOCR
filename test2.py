import paddlehub as hub

# 載入 OCR 模型
ocr = hub.Module(name="chinese_ocr_db_crnn_server")

# 讀入圖片
img_path = 'path/to/image.jpg'
img = cv2.imread(img_path)

# 辨識繁體中文
results = ocr.recognize_text(
    images=[img],
    use_gpu=False,
    text_detection=True,
    enable_mkldnn=True,
    lang='chinese_cht',
)

# 顯示辨識結果
for result in results:
    print(result['data'])
