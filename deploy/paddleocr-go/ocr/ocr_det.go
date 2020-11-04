package ocr

import (
	"log"
	"os"
	"time"

	"github.com/LKKlein/gocv"
)

type DBDetector struct {
	*PaddleModel
	preProcess  DetPreProcess
	postProcess DetPostProcess
}

func NewDBDetector(modelDir string, args map[string]interface{}) *DBDetector {
	maxSideLen := getInt(args, "det_max_side_len", 960)
	thresh := getFloat64(args, "det_db_thresh", 0.3)
	boxThresh := getFloat64(args, "det_db_box_thresh", 0.5)
	unClipRatio := getFloat64(args, "det_db_unclip_ratio", 2.0)

	detector := &DBDetector{
		PaddleModel: NewPaddleModel(args),
		preProcess:  NewDBProcess(make([]int, 0), maxSideLen),
		postProcess: NewDBPostProcess(thresh, boxThresh, unClipRatio),
	}
	if checkModelExists(modelDir) {
		home, _ := os.UserHomeDir()
		modelDir, _ = downloadModel(home+"/.paddleocr/det", modelDir)
	} else {
		log.Panicf("det model path: %v not exist! Please check!", modelDir)
	}
	detector.LoadModel(modelDir)
	return detector
}

func (det *DBDetector) Run(img gocv.Mat) [][][]int {
	oriH := img.Rows()
	oriW := img.Cols()
	data, resizeH, resizeW := det.preProcess.Run(img)
	st := time.Now()
	det.input.SetValue(data)
	det.input.Reshape([]int32{1, 3, int32(resizeH), int32(resizeW)})

	det.predictor.SetZeroCopyInput(det.input)
	det.predictor.ZeroCopyRun()
	det.predictor.GetZeroCopyOutput(det.outputs[0])

	ratioH, ratioW := float64(resizeH)/float64(oriH), float64(resizeW)/float64(oriW)
	boxes := det.postProcess.Run(det.outputs[0], oriH, oriW, ratioH, ratioW)
	log.Println("det_box num: ", len(boxes), ", time elapse: ", time.Since(st))
	return boxes
}
