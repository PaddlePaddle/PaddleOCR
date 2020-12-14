package ocr

import (
	"bytes"
	"encoding/json"
	"errors"
	"image"
	"image/color"
	"io"
	"log"
	"math"
	"net/http"
	"path"
	"path/filepath"
	"sort"
	"strings"

	"github.com/LKKlein/gocv"
	"github.com/PaddlePaddle/PaddleOCR/thirdparty/paddleocr-go/paddle"
)

type PaddleModel struct {
	predictor *paddle.Predictor
	input     *paddle.ZeroCopyTensor
	outputs   []*paddle.ZeroCopyTensor

	useGPU      bool
	deviceID    int
	initGPUMem  int
	numThreads  int
	useMKLDNN   bool
	useTensorRT bool
	useIROptim  bool
}

func NewPaddleModel(args map[string]interface{}) *PaddleModel {
	return &PaddleModel{
		useGPU:      getBool(args, "use_gpu", false),
		deviceID:    getInt(args, "gpu_id", 0),
		initGPUMem:  getInt(args, "gpu_mem", 1000),
		numThreads:  getInt(args, "num_cpu_threads", 6),
		useMKLDNN:   getBool(args, "enable_mkldnn", false),
		useTensorRT: getBool(args, "use_tensorrt", false),
		useIROptim:  getBool(args, "ir_optim", true),
	}
}

func (model *PaddleModel) LoadModel(modelDir string) {
	config := paddle.NewAnalysisConfig()
	config.DisableGlogInfo()

	config.SetModel(modelDir+"/model", modelDir+"/params")
	if model.useGPU {
		config.EnableUseGpu(model.initGPUMem, model.deviceID)
	} else {
		config.DisableGpu()
		config.SetCpuMathLibraryNumThreads(model.numThreads)
		if model.useMKLDNN {
			config.EnableMkldnn()
		}
	}

	// config.EnableMemoryOptim()
	if model.useIROptim {
		config.SwitchIrOptim(true)
	}

	// false for zero copy tensor
	config.SwitchUseFeedFetchOps(false)
	config.SwitchSpecifyInputNames(true)

	model.predictor = paddle.NewPredictor(config)
	model.input = model.predictor.GetInputTensors()[0]
	model.outputs = model.predictor.GetOutputTensors()
}

type OCRText struct {
	BBox  [][]int `json:"bbox"`
	Text  string  `json:"text"`
	Score float64 `json:"score"`
}

type TextPredictSystem struct {
	detector *DBDetector
	cls      *TextClassifier
	rec      *TextRecognizer
}

func NewTextPredictSystem(args map[string]interface{}) *TextPredictSystem {
	sys := &TextPredictSystem{
		detector: NewDBDetector(getString(args, "det_model_dir", ""), args),
		rec:      NewTextRecognizer(getString(args, "rec_model_dir", ""), args),
	}
	if getBool(args, "use_angle_cls", false) {
		sys.cls = NewTextClassifier(getString(args, "cls_model_dir", ""), args)
	}
	return sys
}

func (sys *TextPredictSystem) sortBoxes(boxes [][][]int) [][][]int {
	sort.Slice(boxes, func(i, j int) bool {
		if boxes[i][0][1] < boxes[j][0][1] {
			return true
		}
		if boxes[i][0][1] > boxes[j][0][1] {
			return false
		}
		return boxes[i][0][0] < boxes[j][0][0]
	})

	for i := 0; i < len(boxes)-1; i++ {
		if math.Abs(float64(boxes[i+1][0][1]-boxes[i][0][1])) < 10 && boxes[i+1][0][0] < boxes[i][0][0] {
			boxes[i], boxes[i+1] = boxes[i+1], boxes[i]
		}
	}
	return boxes
}

func (sys *TextPredictSystem) getRotateCropImage(img gocv.Mat, box [][]int) gocv.Mat {
	cropW := int(math.Sqrt(math.Pow(float64(box[0][0]-box[1][0]), 2) + math.Pow(float64(box[0][1]-box[1][1]), 2)))
	cropH := int(math.Sqrt(math.Pow(float64(box[0][0]-box[3][0]), 2) + math.Pow(float64(box[0][1]-box[3][1]), 2)))
	ptsstd := make([]image.Point, 4)
	ptsstd[0] = image.Pt(0, 0)
	ptsstd[1] = image.Pt(cropW, 0)
	ptsstd[2] = image.Pt(cropW, cropH)
	ptsstd[3] = image.Pt(0, cropH)

	points := make([]image.Point, 4)
	points[0] = image.Pt(box[0][0], box[0][1])
	points[1] = image.Pt(box[1][0], box[1][1])
	points[2] = image.Pt(box[2][0], box[2][1])
	points[3] = image.Pt(box[3][0], box[3][1])

	M := gocv.GetPerspectiveTransform(points, ptsstd)
	defer M.Close()
	dstimg := gocv.NewMat()
	gocv.WarpPerspectiveWithParams(img, &dstimg, M, image.Pt(cropW, cropH),
		gocv.InterpolationCubic, gocv.BorderReplicate, color.RGBA{0, 0, 0, 0})

	if float64(dstimg.Rows()) >= float64(dstimg.Cols())*1.5 {
		srcCopy := gocv.NewMat()
		gocv.Transpose(dstimg, &srcCopy)
		defer dstimg.Close()
		gocv.Flip(srcCopy, &srcCopy, 0)
		return srcCopy
	}
	return dstimg
}

func (sys *TextPredictSystem) Run(img gocv.Mat) []OCRText {
	srcimg := gocv.NewMat()
	defer srcimg.Close()
	img.CopyTo(&srcimg)
	boxes := sys.detector.Run(img)
	if len(boxes) == 0 {
		return nil
	}

	boxes = sys.sortBoxes(boxes)
	cropimages := make([]gocv.Mat, len(boxes))
	for i := 0; i < len(boxes); i++ {
		tmpbox := make([][]int, len(boxes[i]))
		for j := 0; j < len(tmpbox); j++ {
			tmpbox[j] = make([]int, len(boxes[i][j]))
			copy(tmpbox[j], boxes[i][j])
		}
		cropimg := sys.getRotateCropImage(srcimg, tmpbox)
		cropimages[i] = cropimg
	}
	if sys.cls != nil {
		cropimages = sys.cls.Run(cropimages)
	}
	recResult := sys.rec.Run(cropimages, boxes)
	return recResult
}

type OCRSystem struct {
	args map[string]interface{}
	tps  *TextPredictSystem
}

func NewOCRSystem(confFile string, a map[string]interface{}) *OCRSystem {
	args, err := ReadYaml(confFile)
	if err != nil {
		log.Printf("Read config file %v failed! Please check. err: %v\n", confFile, err)
		log.Println("Program will use default config.")
		args = defaultArgs
	}
	for k, v := range a {
		args[k] = v
	}
	return &OCRSystem{
		args: args,
		tps:  NewTextPredictSystem(args),
	}
}

func (ocr *OCRSystem) StartServer(port string) {
	http.HandleFunc("/ocr", ocr.predictHandler)
	log.Println("OCR Server has been started on port :", port)
	err := http.ListenAndServe(":"+port, nil)
	if err != nil {
		log.Panicf("http error! error: %v\n", err)
	}
}

func (ocr *OCRSystem) predictHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		w.Write([]byte(errors.New("post method only").Error()))
		return
	}
	r.ParseMultipartForm(32 << 20)
	var buf bytes.Buffer
	file, header, err := r.FormFile("image")
	if err != nil {
		w.Write([]byte(err.Error()))
		return
	}
	defer file.Close()
	ext := strings.ToLower(path.Ext(header.Filename))
	if ext != ".jpg" && ext != ".png" {
		w.Write([]byte(errors.New("only support image endswith jpg/png").Error()))
		return
	}

	io.Copy(&buf, file)
	img, err2 := gocv.IMDecode(buf.Bytes(), gocv.IMReadColor)
	defer img.Close()
	if err2 != nil {
		w.Write([]byte(err2.Error()))
		return
	}
	result := ocr.PredictOneImage(img)
	if output, err3 := json.Marshal(result); err3 != nil {
		w.Write([]byte(err3.Error()))
	} else {
		w.Write(output)
	}
}

func (ocr *OCRSystem) PredictOneImage(img gocv.Mat) []OCRText {
	return ocr.tps.Run(img)
}

func (ocr *OCRSystem) PredictDirImages(dirname string) map[string][]OCRText {
	if dirname == "" {
		return nil
	}

	imgs, _ := filepath.Glob(dirname + "/*.jpg")
	tmpimgs, _ := filepath.Glob(dirname + "/*.png")
	imgs = append(imgs, tmpimgs...)
	results := make(map[string][]OCRText, len(imgs))
	for i := 0; i < len(imgs); i++ {
		imgname := imgs[i]
		img := ReadImage(imgname)
		defer img.Close()
		res := ocr.PredictOneImage(img)
		results[imgname] = res
	}
	return results
}
