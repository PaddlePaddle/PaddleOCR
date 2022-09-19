import sys
import tarfile
import os
import time
import datetime
import functools 
import cv2
import platform
import numpy as np
from qtpy.QtWidgets import QApplication, QWidget, QPushButton, QProgressBar, \
                           QGridLayout, QMessageBox, QLabel, QFileDialog
from qtpy.QtCore import Signal, QThread, QObject
from qtpy.QtGui import QImage, QPixmap, QIcon

file = os.path.dirname(os.path.abspath(__file__))
root = os.path.abspath(os.path.join(file, '../../'))
sys.path.append(file)
sys.path.insert(0, root)

from ppstructure.predict_system import StructureSystem, save_structure_res
from ppstructure.utility import parse_args, draw_structure_result
from ppocr.utils.network import download_with_progressbar
from ppstructure.recovery.recovery_to_doc import sorted_layout_boxes, convert_info_docx
# from ScreenShotWidget import ScreenShotWidget

__APPNAME__ = "pdf2word"
__VERSION__ = "0.1.1"

URLs_EN = {
    # 下载超英文轻量级PP-OCRv3模型的检测模型并解压
    "en_PP-OCRv3_det_infer": "https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_infer.tar",
    # 下载英文轻量级PP-OCRv3模型的识别模型并解压
    "en_PP-OCRv3_rec_infer": "https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_rec_infer.tar",
    # 下载超轻量级英文表格英文模型并解压
    "en_ppstructure_mobile_v2.0_SLANet_infer": "https://paddleocr.bj.bcebos.com/ppstructure/models/slanet/en_ppstructure_mobile_v2.0_SLANet_infer.tar",
    # 英文版面分析模型
    "picodet_lcnet_x1_0_fgd_layout_infer": "https://paddleocr.bj.bcebos.com/ppstructure/models/layout/picodet_lcnet_x1_0_fgd_layout_infer.tar",
}
DICT_EN = {
    "rec_char_dict_path": "en_dict.txt",
    "layout_dict_path": "layout_publaynet_dict.txt",
}

URLs_CN = {
    # 下载超中文轻量级PP-OCRv3模型的检测模型并解压
    "cn_PP-OCRv3_det_infer": "https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_infer.tar",
    # 下载中文轻量级PP-OCRv3模型的识别模型并解压
    "cn_PP-OCRv3_rec_infer": "https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_infer.tar",
    # 下载超轻量级英文表格英文模型并解压
    "cn_ppstructure_mobile_v2.0_SLANet_infer": "https://paddleocr.bj.bcebos.com/ppstructure/models/slanet/en_ppstructure_mobile_v2.0_SLANet_infer.tar",
    # 中文版面分析模型
    "picodet_lcnet_x1_0_fgd_layout_cdla_infer": "https://paddleocr.bj.bcebos.com/ppstructure/models/layout/picodet_lcnet_x1_0_fgd_layout_cdla_infer.tar",
}
DICT_CN = {
    "rec_char_dict_path":  "ppocr_keys_v1.txt",
    "layout_dict_path": "layout_cdla_dict.txt",
}



def QImageToCvMat(incomingImage) -> np.array:
    '''  
    Converts a QImage into an opencv MAT format  
    '''

    incomingImage = incomingImage.convertToFormat(QImage.Format.Format_RGBA8888)

    width = incomingImage.width()
    height = incomingImage.height()

    ptr = incomingImage.bits()
    ptr.setsize(height * width * 4)
    arr = np.frombuffer(ptr, np.uint8).reshape((height, width, 4))
    return arr


def readImage(image_file) -> list:
    if os.path.basename(image_file)[-3:] in ['pdf']:
        import fitz
        from PIL import Image
        imgs = []
        with fitz.open(image_file) as pdf:
            for pg in range(0, pdf.pageCount):
                page = pdf[pg]
                mat = fitz.Matrix(2, 2)
                pm = page.getPixmap(matrix=mat, alpha=False)

                # if width or height > 2000 pixels, don't enlarge the image
                if pm.width > 2000 or pm.height > 2000:
                    pm = page.getPixmap(matrix=fitz.Matrix(1, 1), alpha=False)

                img = Image.frombytes("RGB", [pm.width, pm.height], pm.samples)
                img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                imgs.append(img)
    else:
        img = cv2.imread(image_file, cv2.IMREAD_COLOR)
        if img is not None:
            imgs = [img]
    
    return imgs


class Worker(QThread):
    progressBarValue = Signal(int)
    endsignal = Signal()
    loopFlag = True

    def __init__(self, predictors, save_pdf, vis_font_path):
        super(Worker, self).__init__()
        self.predictors = predictors
        self.save_pdf = save_pdf
        self.vis_font_path = vis_font_path
        self.lang = 'EN'
        self.imagePaths = []
        self.outputDir = None
        self.setStackSize(1024*1024)

    def setImagePath(self, imagePaths):
        self.imagePaths = imagePaths

    def setLang(self, lang):
        self.lang = lang

    def setOutputDir(self, outputDir):
        self.outputDir = outputDir

    def predictAndSave(self, imgs, img_name):
        all_res = []
        for index, img in enumerate(imgs):
            res, time_dict = self.predictors[self.lang](img)

            # save output
            save_structure_res(res, self.outputDir, img_name)
            draw_img = draw_structure_result(img, res, self.vis_font_path)
            img_save_path = os.path.join(self.outputDir, img_name, 'show_{}.jpg'.format(index))
            if res != []:
                cv2.imwrite(img_save_path, draw_img)

            # recovery
            h, w, _ = img.shape
            res = sorted_layout_boxes(res, w)
            all_res += res

        try:
            convert_info_docx(img, all_res, self.outputDir, img_name, self.save_pdf)
        except Exception as ex:
            print(self,
                "error in layout recovery image:{}, err msg: {}".format(
                img_name, ex))

        print('result save to {}'.format(self.outputDir)) 

    def run(self):
        try:
            findex = 0
            os.makedirs(self.outputDir, exist_ok=True)
            for i, image_file in enumerate(self.imagePaths):
                if self.loopFlag == True:
                    imgs = readImage(image_file)
                    if len(imgs) == 0:
                        continue
                    img_name = os.path.basename(image_file).split('.')[0]
                    os.makedirs(os.path.join(self.outputDir, img_name), exist_ok=True)
                    self.predictAndSave(imgs, img_name)
                    findex += 1
                    self.progressBarValue.emit(findex)
                else:
                    break
            self.endsignal.emit()
            self.exec()
        except Exception as e:
            print(e)
            raise


class APP_Image2Doc(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedHeight(90)
        self.setFixedWidth(400)

        # settings
        self.imagePaths = []
        # self.screenShotWg = ScreenShotWidget()
        self.screenShot = None
        self.save_pdf = False
        self.output_dir = None
        self.vis_font_path = os.path.join(root,
                "doc", "fonts", "simfang.ttf")

        # ProgressBar
        self.pb = QProgressBar()
        self.pb.setRange(0, 100)
        self.pb.setValue(0)

        # 初始化界面
        self.setupUi()

        # 下载模型
        self.downloadModels(URLs_EN)
        self.downloadModels(URLs_CN)

        # 初始化模型
        predictors = { 
            'EN': self.initPredictor('EN'),
            'CN': self.initPredictor('CN'),
        }

        # 设置工作进程
        self._thread = Worker(predictors, self.save_pdf, self.vis_font_path)
        self._thread.progressBarValue.connect(self.handleProgressBarSingal)
        self._thread.endsignal.connect(self.handleEndsignalSignal)
        self._thread.finished.connect(QObject.deleteLater)
        self.time_start = 0  # save start time

    def setupUi(self):
        self.setObjectName("MainWindow")
        self.setWindowTitle(__APPNAME__ + " " + __VERSION__)

        layout = QGridLayout()

        self.openFileButton = QPushButton("打开文件")
        self.openFileButton.setIcon(QIcon(QPixmap("./icons/folder-plus.png")))
        layout.addWidget(self.openFileButton, 0, 0, 1, 1)
        self.openFileButton.clicked.connect(self.handleOpenFileSignal)

        # screenShotButton = QPushButton("截图识别")
        # layout.addWidget(screenShotButton, 0, 1, 1, 1)
        # screenShotButton.clicked.connect(self.screenShotSlot)
        # screenShotButton.setEnabled(False) # temporarily disenble

        self.startCNButton = QPushButton("中文转换")
        self.startCNButton.setIcon(QIcon(QPixmap("./icons/chinese.png")))
        layout.addWidget(self.startCNButton, 0, 1, 1, 1)
        self.startCNButton.clicked.connect(
            functools.partial(self.handleStartSignal, 'CN'))

        self.startENButton = QPushButton("英文转换")
        self.startENButton.setIcon(QIcon(QPixmap("./icons/english.png")))
        layout.addWidget(self.startENButton, 0, 2, 1, 1)
        self.startENButton.clicked.connect(
            functools.partial(self.handleStartSignal, 'EN'))

        self.showResultButton = QPushButton("显示结果")
        self.showResultButton.setIcon(QIcon(QPixmap("./icons/folder-open.png")))
        layout.addWidget(self.showResultButton, 0, 3, 1, 1)
        self.showResultButton.clicked.connect(self.handleShowResultSignal)

        # ProgressBar
        layout.addWidget(self.pb, 2, 0, 1, 4)
        # time estimate label
        self.timeEstLabel = QLabel(
            ("Time Left: --"))
        layout.addWidget(self.timeEstLabel, 3, 0, 1, 4)

        self.setLayout(layout)

    def downloadModels(self, URLs):
        # using custom model
        tar_file_name_list = [
            'inference.pdiparams', 
            'inference.pdiparams.info', 
            'inference.pdmodel',
            'model.pdiparams', 
            'model.pdiparams.info', 
            'model.pdmodel'
        ]
        model_path = os.path.join(root, 'inference')
        os.makedirs(model_path, exist_ok=True)

        # download and unzip models
        for name in URLs.keys():
            url = URLs[name]
            print("Try downloading file: {}".format(url))
            tarname = url.split('/')[-1]
            tarpath = os.path.join(model_path, tarname)
            if os.path.exists(tarpath):
                print("File have already exist. skip")
            else:
                try:
                    download_with_progressbar(url, tarpath)
                except Exception as e:
                    print("Error occurred when downloading file, error message:")
                    print(e)
            
            # unzip model tar
            try:
                with tarfile.open(tarpath, 'r') as tarObj:
                    storage_dir = os.path.join(model_path, name)
                    os.makedirs(storage_dir, exist_ok=True)
                    for member in tarObj.getmembers():
                        filename = None
                        for tar_file_name in tar_file_name_list:
                            if tar_file_name in member.name:
                                filename = tar_file_name
                        if filename is None:
                            continue
                        file = tarObj.extractfile(member)
                        with open(
                                os.path.join(storage_dir, filename),
                                'wb') as f:
                            f.write(file.read())
            except Exception as e:
                    print("Error occurred when unziping file, error message:")
                    print(e)

    def initPredictor(self, lang='EN'):
        # init predictor args
        args = parse_args()
        args.table_max_len = 488
        args.ocr = True
        args.recovery = True
        args.save_pdf = self.save_pdf
        args.table_char_dict_path = os.path.join(root, 
                "ppocr", "utils", "dict", "table_structure_dict.txt")
        if lang == 'EN':
            args.det_model_dir = os.path.join(root,  # 此处从这里找到模型存放位置
                "inference", "en_PP-OCRv3_det_infer")
            args.rec_model_dir = os.path.join(root, 
                "inference", "en_PP-OCRv3_rec_infer")
            args.table_model_dir = os.path.join(root, 
                "inference", "en_ppstructure_mobile_v2.0_SLANet_infer")
            args.output = os.path.join(root, "output") # 结果保存路径
            args.layout_model_dir = os.path.join(root,
                "inference", "picodet_lcnet_x1_0_fgd_layout_infer")
            lang_dict = DICT_EN
        elif lang == 'CN':
            args.det_model_dir = os.path.join(root,  # 此处从这里找到模型存放位置
                "inference", "cn_PP-OCRv3_det_infer")
            args.rec_model_dir = os.path.join(root, 
                "inference", "cn_PP-OCRv3_rec_infer")
            args.table_model_dir = os.path.join(root, 
                "inference", "cn_ppstructure_mobile_v2.0_SLANet_infer")
            args.output = os.path.join(root, "output") # 结果保存路径
            args.layout_model_dir = os.path.join(root,
                "inference", "picodet_lcnet_x1_0_fgd_layout_cdla_infer")
            lang_dict = DICT_CN
        else:
            raise ValueError("Unsupported language")
        args.rec_char_dict_path = os.path.join(root, 
                "ppocr", "utils", 
                lang_dict['rec_char_dict_path'])
        args.layout_dict_path = os.path.join(root,
                "ppocr", "utils", "dict", "layout_dict", 
                lang_dict['layout_dict_path'])
        # init predictor
        return StructureSystem(args)
    
    def handleOpenFileSignal(self):
        '''
        可以多选图像文件
        '''
        selectedFiles = QFileDialog.getOpenFileNames(self, 
            "多文件选择", "/", "图片文件 (*.png *.jpeg *.jpg *.bmp *.pdf)")[0]
        if len(selectedFiles) > 0:
            self.imagePaths = selectedFiles
            self.screenShot = None # discard screenshot temp image
            self.pb.setRange(0, len(self.imagePaths))
            self.pb.setValue(0)

    # def screenShotSlot(self):
    #     '''
    #     选定图像文件和截图的转换过程只能同时进行一个
    #     截图只能同时转换一个
    #     '''
    #     self.screenShotWg.start()
    #     if self.screenShotWg.captureImage:
    #         self.screenShot = self.screenShotWg.captureImage
    #         self.imagePaths.clear() # discard openfile temp list
    #         self.pb.setRange(0, 1)
    #         self.pb.setValue(0)

    def handleStartSignal(self, lang):
        if self.screenShot: # for screenShot
            img_name = 'screenshot_' + time.strftime("%Y%m%d%H%M%S", time.localtime())
            image = QImageToCvMat(self.screenShot)
            self.predictAndSave(image, img_name, lang)
            # update Progress Bar
            self.pb.setValue(1)
            QMessageBox.information(self, 
                u'Information', "文档提取完成")
        elif len(self.imagePaths) > 0 : # for image file selection
            # Must set image path list and language before start
            self.output_dir = os.path.join(
                os.path.dirname(self.imagePaths[0]), "output")  # output_dir shold be same as imagepath
            self._thread.setOutputDir(self.output_dir)
            self._thread.setImagePath(self.imagePaths)
            self._thread.setLang(lang)
            # disenble buttons
            self.openFileButton.setEnabled(False)
            self.startCNButton.setEnabled(False)
            self.startENButton.setEnabled(False)
            # 启动工作进程
            self._thread.start()
            self.time_start = time.time() # log start time
            QMessageBox.information(self, 
                u'Information', "开始转换")
        else:
            QMessageBox.warning(self, 
                u'Information', "请选择要识别的文件或截图")

    def handleShowResultSignal(self):
        if self.output_dir is None:
            return
        if os.path.exists(self.output_dir):
            if platform.system() == 'Windows':
                os.startfile(self.output_dir)
            else:
                os.system('open ' + os.path.normpath(self.output_dir))
        else:
            QMessageBox.information(self, 
                u'Information', "输出文件不存在")

    def handleProgressBarSingal(self, i):
        self.pb.setValue(i)
        # calculate time left of recognition
        lenbar = self.pb.maximum()
        avg_time = (time.time() - self.time_start) / i  # Use average time to prevent time fluctuations
        time_left = str(datetime.timedelta(seconds=avg_time * (lenbar - i))).split(".")[0]  # Remove microseconds
        self.timeEstLabel.setText(f"Time Left: {time_left}")  # show time left

    def handleEndsignalSignal(self):
        # enble buttons
        self.openFileButton.setEnabled(True)
        self.startCNButton.setEnabled(True)
        self.startENButton.setEnabled(True)
        QMessageBox.information(self, u'Information', "转换结束")


def main():
    app = QApplication(sys.argv)

    window = APP_Image2Doc()  # 创建对象
    window.show()  # 全屏显示窗口

    QApplication.processEvents()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
