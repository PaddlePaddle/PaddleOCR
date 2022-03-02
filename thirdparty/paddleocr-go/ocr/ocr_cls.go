package ocr

import (
	"log"
	"os"
	"time"

	"github.com/LKKlein/gocv"
)

type TextClassifier struct {
	*PaddleModel
	batchNum int
	thresh   float64
	shape    []int
	labels   []string
}

type ClsResult struct {
	Score float32
	Label int64
}

func NewTextClassifier(modelDir string, args map[string]interface{}) *TextClassifier {
	shapes := []int{3, 48, 192}
	if v, ok := args["cls_image_shape"]; ok {
		for i, s := range v.([]interface{}) {
			shapes[i] = s.(int)
		}
	}
	cls := &TextClassifier{
		PaddleModel: NewPaddleModel(args),
		batchNum:    getInt(args, "cls_batch_num", 30),
		thresh:      getFloat64(args, "cls_thresh", 0.9),
		shape:       shapes,
	}
	if checkModelExists(modelDir) {
		home, _ := os.UserHomeDir()
		modelDir, _ = downloadModel(home+"/.paddleocr/cls", modelDir)
	} else {
		log.Panicf("cls model path: %v not exist! Please check!", modelDir)
	}
	cls.LoadModel(modelDir)
	return cls
}

func (cls *TextClassifier) Run(imgs []gocv.Mat) []gocv.Mat {
	batch := cls.batchNum
	var clsTime int64 = 0
	clsout := make([]ClsResult, len(imgs))
	srcimgs := make([]gocv.Mat, len(imgs))
	c, h, w := cls.shape[0], cls.shape[1], cls.shape[2]
	for i := 0; i < len(imgs); i += batch {
		j := i + batch
		if len(imgs) < j {
			j = len(imgs)
		}

		normImgs := make([]float32, (j-i)*c*h*w)
		for k := i; k < j; k++ {
			tmp := gocv.NewMat()
			imgs[k].CopyTo(&tmp)
			srcimgs[k] = tmp
			img := clsResize(imgs[k], cls.shape)
			data := normPermute(img, []float32{0.5, 0.5, 0.5}, []float32{0.5, 0.5, 0.5}, 255.0)
			copy(normImgs[(k-i)*c*h*w:], data)
		}

		st := time.Now()
		cls.input.SetValue(normImgs)
		cls.input.Reshape([]int32{int32(j - i), int32(c), int32(h), int32(w)})

		cls.predictor.SetZeroCopyInput(cls.input)
		cls.predictor.ZeroCopyRun()
		cls.predictor.GetZeroCopyOutput(cls.outputs[0])
		cls.predictor.GetZeroCopyOutput(cls.outputs[1])

		var probout [][]float32
		var labelout []int64
		if len(cls.outputs[0].Shape()) == 2 {
			probout = cls.outputs[0].Value().([][]float32)
		} else {
			labelout = cls.outputs[0].Value().([]int64)
		}

		if len(cls.outputs[1].Shape()) == 2 {
			probout = cls.outputs[1].Value().([][]float32)
		} else {
			labelout = cls.outputs[1].Value().([]int64)
		}
		clsTime += int64(time.Since(st).Milliseconds())

		for no, label := range labelout {
			score := probout[no][label]
			clsout[i+no] = ClsResult{
				Score: score,
				Label: label,
			}

			if label%2 == 1 && float64(score) > cls.thresh {
				gocv.Rotate(srcimgs[i+no], &srcimgs[i+no], gocv.Rotate180Clockwise)
			}
		}
	}
	log.Println("cls num: ", len(clsout), ", cls time elapse: ", clsTime, "ms")
	return srcimgs
}
