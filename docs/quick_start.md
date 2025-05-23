---
comments: true
hide:
  - navigation
---

### 安装

#### 1. 安装PaddlePaddle

CPU端安装：

```bash
python -m pip install paddlepaddle==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
```

GPU端安装，由于GPU端需要根据具体CUDA版本来对应安装使用，以下仅以Linux平台，pip安装英伟达GPU， CUDA11.8为例，其他平台，请参考[飞桨官网安装文档](https://www.paddlepaddle.org.cn/install/quick)中的说明进行操作。

```bash
python -m pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/
```

#### 2. 安装`paddleocr`

```bash
pip install paddleocr==3.0.0
```

### 命令行使用

=== "PP-OCRv5"

    ```bash linenums="1"
    paddleocr ocr -i ./general_ocr_002.png --use_doc_orientation_classify False --use_doc_unwarping False --use_textline_orientation False
    ```

=== "PP-OCRv5文本检测模块"

    ```bash linenums="1"
    paddleocr text_detection -i ./general_ocr_001.png 
    ```

=== "PP-OCRv5文本识别模块"

    ```bash linenums="1"
    paddleocr text_recognition -i ./general_ocr_rec_001.png
    ```

=== "PP-StructureV3"

    ```bash linenums="1"
    paddleocr pp_structurev3 -i ./pp_structure_v3_demo.png --use_doc_orientation_classify False --use_doc_unwarping False
    ```

### Python脚本使用

=== "PP-OCRv5"

    ```python linenums="1"
    from paddleocr import PaddleOCR

    ocr = PaddleOCR(
        use_doc_orientation_classify=False, 
        use_doc_unwarping=False, 
        use_textline_orientation=False) # 文本检测+文本识别
    # ocr = PaddleOCR(use_doc_orientation_classify=True, use_doc_unwarping=True) # 文本图像预处理+文本检测+方向分类+文本识别
    # ocr = PaddleOCR(use_doc_orientation_classify=False, use_doc_unwarping=False) # 文本检测+文本行方向分类+文本识别
    # ocr = PaddleOCR(
    #     text_detection_model_name="PP-OCRv5_server_det",
    #     text_recognition_model_name="PP-OCRv5_server_rec",
    #     use_doc_orientation_classify=False,
    #     use_doc_unwarping=False,
    #     use_textline_orientation=False) # 更换 PP-OCRv5_server 模型
    result = ocr.predict("./general_ocr_002.png")
    for res in result:
        res.print()
        res.save_to_img("output")
        res.save_to_json("output")
    ```

    输出示例：

    ```bash
    {'res': {'input_path': '/root/.paddlex/predict_input/general_ocr_002.png', 'page_index': None, 'model_settings': {'use_doc_preprocessor': True, 'use_textline_orientation': False}, 'doc_preprocessor_res': {'input_path': None, 'page_index': None, 'model_settings': {'use_doc_orientation_classify': False, 'use_doc_unwarping': False}, 'angle': -1}, 'dt_polys': array([[[  3,  10],
            ...,
            [  4,  30]],

        ...,

        [[ 99, 456],
            ...,
            [ 99, 479]]], dtype=int16), 'text_det_params': {'limit_side_len': 736, 'limit_type': 'min', 'thresh': 0.3, 'max_side_limit': 4000, 'box_thresh': 0.6, 'unclip_ratio': 1.5}, 'text_type': 'general', 'textline_orientation_angles': array([-1, ..., -1]), 'text_rec_score_thresh': 0.0, 'rec_texts': ['www.997700', '', 'Cm', '登机牌', 'BOARDING', 'PASS', 'CLASS', '序号SERIAL NO.', '座位号', 'SEAT NO.', '航班FLIGHT', '日期DATE', '舱位', '', 'W', '035', '12F', 'MU2379', '03DEc', '始发地', 'FROM', '登机口', 'GATE', '登机时间BDT', '目的地TO', '福州', 'TAIYUAN', 'G11', 'FUZHOU', '身份识别IDNO.', '姓名NAME', 'ZHANGQIWEI', '票号TKT NO.', '张祺伟', '票价FARE', 'ETKT7813699238489/1', '登机口于起飞前10分钟关闭 GATESCL0SE10MINUTESBEFOREDEPARTURETIME'], 'rec_scores': array([0.67582953, ..., 0.97418666]), 'rec_polys': array([[[  3,  10],
            ...,
            [  4,  30]],

        ...,

        [[ 99, 456],
            ...,
            [ 99, 479]]], dtype=int16), 'rec_boxes': array([[  3, ...,  30],
        ...,
        [ 99, ..., 479]], dtype=int16)}}
    ```

=== "PP-OCRv5文本检测模块"

    ```python linenums="1"
    from paddleocr import TextDetection

    model = TextDetection()
    output = model.predict("general_ocr_001.png")
    for res in output:
        res.print()
        res.save_to_img(save_path="./output/")
        res.save_to_json(save_path="./output/res.json")
    ```

    ```bash
    {'res': {'input_path': 'general_ocr_001.png', 'page_index': None, 'dt_polys': array([[[ 77, 551],
        ...,
        [ 78, 587]],

       ...,

       [[ 34, 408],
        ...,
        [ 36, 456]]], dtype=int16), 'dt_scores': [0.8562385635646694, 0.8818259002228059, 0.8406072284043453, 0.8855339313157491]}}
    ```

=== "PP-OCRv5文本识别模块"

    ```python linenums="1"
    from paddleocr import TextRecognition

    model = TextRecognition()
    output = model.predict(input="general_ocr_rec_001.png")
    for res in output:
        res.print()
        res.save_to_img(save_path="./output/")
        res.save_to_json(save_path="./output/res.json")
    ```

    输出示例：
    
    ```bash
    {'res': {'input_path': 'general_ocr_rec_001.png', 'page_index': None, 'rec_text': '绿洲仕格维花园公寓', 'rec_score': 0.990813672542572}}
    ```

=== "PP-StructureV3"

    ```python linenums="1"
    from paddleocr import PPStructureV3

    pipeline = PPStructureV3(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False)
    output = pipeline.predict(
        input="./pp_structure_v3_demo.png",          
    )
    for res in output:
        res.print()
        res.save_to_json(save_path="output")
        res.save_to_markdown(save_path="output")
    ```

    输出示例：

    ```bash
    {'res': {'input_path': './pp_structure_v3_demo.png', 'page_index': None, 'model_settings': {'use_doc_preprocessor': False, 'use_seal_recognition': True, 'use_table_recognition': True, 'use_formula_recognition': True, 'use_chart_recognition': False, 'use_region_detection': True}, 'layout_det_res': {'input_path': None, 'page_index': None, 'boxes': [{'cls_id': 1, 'label': 'image', 'score': 0.9864752888679504, 'coordinate': [774.821, 201.05177, 1502.1008, 685.7733]}, {'cls_id': 2, 'label': 'text', 'score': 0.9859225749969482, 'coordinate': [769.8655, 776.2446, 1121.5986, 1058.417]}, {'cls_id': 2, 'label': 'text', 'score': 0.9857110381126404, 'coordinate': [1151.98, 1112.5356, 1502.7852, 1346.3569]}, {'cls_id': 2, 'label': 'text', 'score': 0.9847239255905151, 'coordinate': [389.0322, 1136.3547, 740.2322, 1345.928]}, {'cls_id': 2, 'label': 'text', 'score': 0.9842492938041687, 'coordinate': [1152.1504, 800.1625, 1502.1265, 986.1522]}, {'cls_id': 2, 'label': 'text', 'score': 0.9840831160545349, 'coordinate': [9.158066, 848.8696, 358.5725, 1057.832]}, {'cls_id': 2, 'label': 'text', 'score': 0.9802583456039429, 'coordinate': [9.335953, 201.10046, 358.31543, 338.78876]}, {'cls_id': 2, 'label': 'text', 'score': 0.9801402688026428, 'coordinate': [389.1556, 297.4113, 740.07556, 435.41647]}, {'cls_id': 2, 'label': 'text', 'score': 0.9793564081192017, 'coordinate': [389.18976, 752.0959, 740.0832, 889.88043]}, {'cls_id': 2, 'label': 'text', 'score': 0.9793409109115601, 'coordinate': [389.02496, 896.34143, 740.7431, 1033.9465]}, {'cls_id': 2, 'label': 'text', 'score': 0.9776486754417419, 'coordinate': [8.950775, 1184.7842, 358.75067, 1297.8755]}, {'cls_id': 2, 'label': 'text', 'score': 0.9773538708686829, 'coordinate': [770.7178, 1064.5714, 1121.2249, 1177.9928]}, {'cls_id': 2, 'label': 'text', 'score': 0.9773064255714417, 'coordinate': [389.38086, 609.7071, 740.0553, 745.3206]}, {'cls_id': 2, 'label': 'text', 'score': 0.9765821099281311, 'coordinate': [1152.0112, 992.296, 1502.4927, 1106.1166]}, {'cls_id': 2, 'label': 'text', 'score': 0.9761461019515991, 'coordinate': [9.46727, 536.993, 358.2047, 651.32025]}, {'cls_id': 2, 'label': 'text', 'score': 0.975399911403656, 'coordinate': [9.353531, 1064.3059, 358.45312, 1177.8347]}, {'cls_id': 2, 'label': 'text', 'score': 0.9730532169342041, 'coordinate': [9.932312, 345.36237, 358.03476, 435.1646]}, {'cls_id': 2, 'label': 'text', 'score': 0.9722575545310974, 'coordinate': [388.91736, 200.93637, 740.00793, 290.80692]}, {'cls_id': 2, 'label': 'text', 'score': 0.9710633158683777, 'coordinate': [389.39496, 1040.3186, 740.0091, 1129.7168]}, {'cls_id': 2, 'label': 'text', 'score': 0.9696939587593079, 'coordinate': [9.6145935, 658.1123, 359.06088, 770.0288]}, {'cls_id': 2, 'label': 'text', 'score': 0.9664146900177002, 'coordinate': [770.235, 1280.4562, 1122.0927, 1346.4742]}, {'cls_id': 2, 'label': 'text', 'score': 0.9597565531730652, 'coordinate': [389.66678, 537.5609, 740.06274, 603.17725]}, {'cls_id': 2, 'label': 'text', 'score': 0.9594324827194214, 'coordinate': [10.162949, 776.86414, 359.08307, 842.1771]}, {'cls_id': 2, 'label': 'text', 'score': 0.9484634399414062, 'coordinate': [10.402863, 1304.7743, 358.9441, 1346.3749]}, {'cls_id': 0, 'label': 'paragraph_title', 'score': 0.9476125240325928, 'coordinate': [28.159409, 456.7627, 339.5631, 514.9665]}, {'cls_id': 0, 'label': 'paragraph_title', 'score': 0.9427680969238281, 'coordinate': [790.6992, 1200.3663, 1102.3799, 1259.1647]}, {'cls_id': 0, 'label': 'paragraph_title', 'score': 0.9424256682395935, 'coordinate': [409.02832, 456.6831, 718.8154, 515.5757]}, {'cls_id': 10, 'label': 'doc_title', 'score': 0.9376171827316284, 'coordinate': [133.77905, 36.8844, 1379.6667, 123.46869]}, {'cls_id': 2, 'label': 'text', 'score': 0.9020252823829651, 'coordinate': [584.9165, 159.1416, 927.22876, 179.01605]}, {'cls_id': 2, 'label': 'text', 'score': 0.895164430141449, 'coordinate': [1154.3364, 776.74646, 1331.8564, 794.2301]}, {'cls_id': 6, 'label': 'figure_title', 'score': 0.7892374396324158, 'coordinate': [808.9641, 704.2555, 1484.0623, 747.2296]}]}, 'overall_ocr_res': {'input_path': None, 'page_index': None, 'model_settings': {'use_doc_preprocessor': False, 'use_textline_orientation': False}, 'dt_polys': array([[[ 129,   42],
            ...,
            [ 129,  140]],

        ...,

        [[1156, 1330],
            ...,
            [1156, 1351]]], dtype=int16), 'text_det_params': {'limit_side_len': 736, 'limit_type': 'min', 'thresh': 0.3, 'max_side_limit': 4000, 'box_thresh': 0.6, 'unclip_ratio': 1.5}, 'text_type': 'general', 'textline_orientation_angles': array([-1, ..., -1]), 'text_rec_score_thresh': 0.0, 'rec_texts': ['助力双方交往', '搭建友谊桥梁', '本报记者沈小晓', '任', '彦', '黄培昭', '身着中国传统民族服装的厄立特里亚青', '厄立特里亚高等教育与研究院合作建立，开', '年依次登台表演中国民族舞、现代舞、扇子舞', '设了中国语言课程和中国文化课程，注册学', '等，曼妙的舞姿赢得现场观众阵阵掌声。这', '生2万余人次。10余年来，厄特孔院已成为', '是日前危立特里亚高等教育与研究院孔子学', '当地民众了解中国的一扇窗口。', '院(以下简称“厄特孔院")举办“喜迎新年"中国', '黄鸣飞表示，随着来学习中文的人日益', '歌舞比赛的场景。', '增多，阿斯马拉大学教学点已难以满足教学', '中国和厄立特里亚传统友谊深厚。近年', '需要。2024年4月，由中企蜀道集团所属四', '来，在高质量共建“一带一路”框架下，中厄两', '川路桥承建的孔院教学楼项目在阿斯马拉开', '国人文交流不断深化，互利合作的民意基础', '工建设，预计今年上半年竣工，建成后将为危', '日益深厚。', '特孔院提供全新的办学场地。', '“学好中文，我们的', '“在中国学习的经历', '未来不是梦”', '让我看到更广阔的世界”', '“鲜花曾告诉我你怎样走过，大地知道你', '多年来，厄立特里亚广大赴华留学生和', '心中的每一个角落……"厄立特里亚阿斯马拉', '培训人员积极投身国家建设，成为助力该国', '大学综合楼二层，一阵优美的歌声在走廊里回', '发展的人才和厄中友好的见证者和推动者。', '响。循着熟悉的旋律轻轻推开一间教室的门，', '在厄立特里亚全国妇女联盟工作的约翰', '学生们正跟着老师学唱中文歌曲《同一首歌》。', '娜·特韦尔德·凯莱塔就是其中一位。她曾在', '这是厄特孔院阿斯马拉大学教学点的一', '中华女子学院攻读硕士学位，研究方向是女', '节中文歌曲课。为了让学生们更好地理解歌', '性领导力与社会发展。其间，她实地走访中国', '词大意，老师尤斯拉·穆罕默德萨尔·侯赛因逐', '多个地区，获得了观察中国社会发展的第一', '在厄立特里亚不久前举办的第六届中国风筝文化节上，当地小学生体验风筝制作。', '字翻译和解释歌词。随着伴奏声响起，学生们', '手资料。', '中国驻厄立特里亚大使馆供图', '边唱边随着节拍摇动身体，现场气氛热烈。', '谈起在中国求学的经历，约翰娜记忆犹', '“这是中文歌曲初级班，共有32人。学', '新：“中国的发展在当今世界是独一无二的。', '“不管远近都是客人，请不用客气；相约', '瓦的北红海省博物馆。', '生大部分来自首都阿斯马拉的中小学，年龄', '沿着中国特色社会主义道路坚定前行，中国', '好了在一起，我们欢迎你……”在一场中厄青', '博物馆二层陈列着一个发掘自阿杜利', '最小的仅有6岁。”尤斯拉告诉记者。', '创造了发展奇迹，这一切都离不开中国共产党', '年联谊活动上，四川路桥中方员工同当地大', '斯古城的中国古代陶制酒器，罐身上写着', '尤斯拉今年23岁，是厄立特里亚一所公立', '的领导。中国的发展经验值得许多国家学习', '学生合唱《北京欢迎你》。厄立特里亚技术学', '“万”“和”“禅”“山”等汉字。“这件文物证', '学校的艺术老师。她12岁开始在厄特孔院学', '借鉴。”', '院计算机科学与工程专业学生鲁夫塔·谢拉', '明，很早以前我们就通过海上丝绸之路进行', '习中文，在2017年第十届“汉语桥"世界中学生', '正在西南大学学习的厄立特里亚博士生', '是其中一名演唱者，她很早便在孔院学习中', '贸易往来与文化交流。这也是厄立特里亚', '中文比赛中获得厄立特里亚赛区第一名，并和', '穆卢盖塔·泽穆伊对中国怀有深厚感情。8', '文，一直在为去中国留学作准备。“这句歌词', '与中国友好交往历史的有力证明。”北红海', '同伴代表厄立特里亚前往中国参加决赛，获得', '年前，在北京师范大学获得硕士学位后，穆卢', '是我们两国人民友谊的生动写照。无论是投', '省博物馆研究与文献部负责人伊萨亚斯·特', '团体优胜奖。2022年起，尤斯拉开始在厄特孔', '盖塔在社交媒体上写下这样一段话：“这是我', '身于厄立特里亚基础设施建设的中企员工，', '斯法兹吉说。', '院兼职教授中文歌曲，每周末两个课时。“中国', '人生的重要一步，自此我拥有了一双坚固的', '还是在中国留学的厄立特里亚学子，两国人', '厄立特里亚国家博物馆考古学和人类学', '文化博大精深，我希望我的学生们能够通过中', '鞋子，赋予我穿越荆棘的力量。”', '民携手努力，必将推动两国关系不断向前发', '研究员菲尔蒙·特韦尔德十分喜爱中国文', '文歌曲更好地理解中国文化。”她说。', '穆卢盖塔密切关注中国在经济、科技、教', '展。”鲁夫塔说。', '化。他表示：“学习彼此的语言和文化，将帮', '“姐姐，你想去中国吗?”“非常想！我想', '育等领域的发展，“中国在科研等方面的实力', '厄立特里亚高等教育委员会主任助理萨', '助厄中两国人民更好地理解彼此，助力双方', '去看故宫、爬长城。”尤斯拉的学生中有一对', '与日俱增。在中国学习的经历让我看到更广', '马瑞表示：“每年我们都会组织学生到中国访', '交往，搭建友谊桥梁。”', '能歌善舞的姐妹，姐姐露娅今年15岁，妹妹', '阔的世界，从中受益匪浅。”', '问学习，自前有超过5000名厄立特里亚学生', '厄立特里亚国家博物馆馆长塔吉丁·努', '莉娅14岁，两人都已在厄特孔院学习多年，', '23岁的莉迪亚·埃斯蒂法诺斯已在厄特', '在中国留学。学习中国的教育经验，有助于', '重达姆·优素福曾多次访问中国，对中华文明', '中文说得格外流利。', '孔院学习3年，在中国书法、中国画等方面表', '提升厄立特里亚的教育水平。”', '的传承与创新、现代化博物馆的建设与发展', '露娅对记者说：“这些年来，怀着对中文', '现干分优秀，在2024年厄立特里亚赛区的', '印象深刻。“中国博物馆不仅有许多保存完好', '“共同向世界展示非', '和中国文化的热爱，我们姐妹俩始终相互鼓', '“汉语桥”比赛中获得一等奖。莉迪亚说：“学', '的文物，还充分运用先进科技手段进行展示，', '励，一起学习。我们的中文一天比一天好，还', '习中国书法让我的内心变得安宁和纯粹。我', '洲和亚洲的灿烂文明”', '帮助人们更好理解中华文明。”塔吉丁说，“危', '学会了中文歌和中国舞。我们一定要到中国', '也喜欢中国的服饰，希望未来能去中国学习，', '立特里亚与中国都拥有悠久的文明，始终相', '去。学好中文，我们的未来不是梦！”', '把中国不同民族元素融入服装设计中，创作', '从阿斯马拉出发，沿着蜿蜒曲折的盘山', '互理解、相互尊重。我希望未来与中国同行', '据厄特孔院中方院长黄鸣飞介绍，这所', '出更多精美作品，也把厄特文化分享给更多', '公路一路向东寻找丝路印迹。驱车两个小', '加强合作，共同向世界展示非洲和亚洲的灿', '孔院成立于2013年3月，由贵州财经大学和', '的中国朋友。”', '时，记者来到位于厄立特里亚港口城市马萨', '烂文明。”'], 'rec_scores': array([0.99113536, ..., 0.95110035]), 'rec_polys': array([[[ 129,   42],
            ...,
            [ 129,  140]],

        ...,

        [[1156, 1330],
            ...,
            [1156, 1351]]], dtype=int16), 'rec_boxes': array([[ 129, ...,  140],
        ...,
        [1156, ..., 1351]], dtype=int16)}}}
    ```
