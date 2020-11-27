package ocr

import (
	"image"
	"image/color"
	"math"

	"github.com/LKKlein/gocv"
)

func resizeByShape(img gocv.Mat, resizeShape []int) (gocv.Mat, int, int) {
	resizeH := resizeShape[0]
	resizeW := resizeShape[1]
	gocv.Resize(img, &img, image.Pt(resizeW, resizeH), 0, 0, gocv.InterpolationLinear)
	return img, resizeH, resizeW
}

func resizeByMaxLen(img gocv.Mat, maxLen int) (gocv.Mat, int, int) {
	oriH := img.Rows()
	oriW := img.Cols()
	var resizeH, resizeW int = oriH, oriW

	var ratio float64 = 1.0
	if resizeH > maxLen || resizeW > maxLen {
		if resizeH > resizeW {
			ratio = float64(maxLen) / float64(resizeH)
		} else {
			ratio = float64(maxLen) / float64(resizeW)
		}
	}

	resizeH = int(float64(resizeH) * ratio)
	resizeW = int(float64(resizeW) * ratio)

	if resizeH%32 == 0 {
		resizeH = resizeH
	} else if resizeH/32 <= 1 {
		resizeH = 32
	} else {
		resizeH = (resizeH/32 - 1) * 32
	}

	if resizeW%32 == 0 {
		resizeW = resizeW
	} else if resizeW/32 <= 1 {
		resizeW = 32
	} else {
		resizeW = (resizeW/32 - 1) * 32
	}

	if resizeW <= 0 || resizeH <= 0 {
		return gocv.NewMat(), 0, 0
	}

	gocv.Resize(img, &img, image.Pt(resizeW, resizeH), 0, 0, gocv.InterpolationLinear)
	return img, resizeH, resizeW
}

func normPermute(img gocv.Mat, mean []float32, std []float32, scaleFactor float32) []float32 {
	img.ConvertTo(&img, gocv.MatTypeCV32F)
	img.DivideFloat(scaleFactor)

	c := gocv.Split(img)
	data := make([]float32, img.Rows()*img.Cols()*img.Channels())
	for i := 0; i < 3; i++ {
		c[i].SubtractFloat(mean[i])
		c[i].DivideFloat(std[i])
		defer c[i].Close()
		x, _ := c[i].DataPtrFloat32()
		copy(data[i*img.Rows()*img.Cols():], x)
	}
	return data
}

type DetPreProcess interface {
	Run(gocv.Mat) ([]float32, int, int)
}

type DBPreProcess struct {
	resizeType  int
	imageShape  []int
	maxSideLen  int
	mean        []float32
	std         []float32
	scaleFactor float32
}

func NewDBProcess(shape []int, sideLen int) *DBPreProcess {
	db := &DBPreProcess{
		resizeType:  0,
		imageShape:  shape,
		maxSideLen:  sideLen,
		mean:        []float32{0.485, 0.456, 0.406},
		std:         []float32{0.229, 0.224, 0.225},
		scaleFactor: 255.0,
	}
	if len(shape) > 0 {
		db.resizeType = 1
	}
	if sideLen == 0 {
		db.maxSideLen = 2400
	}
	return db
}

func (d *DBPreProcess) Run(img gocv.Mat) ([]float32, int, int) {
	var resizeH, resizeW int
	if d.resizeType == 0 {
		img, resizeH, resizeW = resizeByMaxLen(img, d.maxSideLen)
	} else {
		img, resizeH, resizeW = resizeByShape(img, d.imageShape)
	}

	im := normPermute(img, d.mean, d.std, d.scaleFactor)
	return im, resizeH, resizeW
}

func clsResize(img gocv.Mat, resizeShape []int) gocv.Mat {
	imgH, imgW := resizeShape[1], resizeShape[2]
	h, w := img.Rows(), img.Cols()
	ratio := float64(w) / float64(h)
	var resizeW int
	if math.Ceil(float64(imgH)*ratio) > float64(imgW) {
		resizeW = imgW
	} else {
		resizeW = int(math.Ceil(float64(imgH) * ratio))
	}
	gocv.Resize(img, &img, image.Pt(resizeW, imgH), 0, 0, gocv.InterpolationLinear)
	if resizeW < imgW {
		gocv.CopyMakeBorder(img, &img, 0, 0, 0, imgW-resizeW, gocv.BorderConstant, color.RGBA{0, 0, 0, 0})
	}
	return img
}

func crnnPreprocess(img gocv.Mat, resizeShape []int, mean []float32, std []float32,
	scaleFactor float32, whRatio float64, charType string) []float32 {
	imgH := resizeShape[1]
	imgW := resizeShape[2]
	if charType == "ch" {
		imgW = int(32 * whRatio)
	}
	h, w := img.Rows(), img.Cols()
	ratio := float64(w) / float64(h)
	var resizeW int
	if math.Ceil(float64(imgH)*ratio) > float64(imgW) {
		resizeW = imgW
	} else {
		resizeW = int(math.Ceil(float64(imgH) * ratio))
	}
	gocv.Resize(img, &img, image.Pt(resizeW, imgH), 0, 0, gocv.InterpolationLinear)

	img.ConvertTo(&img, gocv.MatTypeCV32F)
	img.DivideFloat(scaleFactor)
	img.SubtractScalar(gocv.NewScalar(float64(mean[0]), float64(mean[1]), float64(mean[2]), 0))
	img.DivideScalar(gocv.NewScalar(float64(std[0]), float64(std[1]), float64(std[2]), 0))

	if resizeW < imgW {
		gocv.CopyMakeBorder(img, &img, 0, 0, 0, imgW-resizeW, gocv.BorderConstant, color.RGBA{0, 0, 0, 0})
	}

	c := gocv.Split(img)
	data := make([]float32, img.Rows()*img.Cols()*img.Channels())
	for i := 0; i < 3; i++ {
		defer c[i].Close()
		x, _ := c[i].DataPtrFloat32()
		copy(data[i*img.Rows()*img.Cols():], x)
	}
	return data
}
