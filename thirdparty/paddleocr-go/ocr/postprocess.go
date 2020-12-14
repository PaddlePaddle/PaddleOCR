package ocr

import (
	"image"
	"image/color"
	"math"
	"sort"

	"github.com/LKKlein/gocv"
	"github.com/PaddlePaddle/PaddleOCR/thirdparty/paddleocr-go/paddle"
	clipper "github.com/ctessum/go.clipper"
)

type xFloatSortBy [][]float32

func (a xFloatSortBy) Len() int           { return len(a) }
func (a xFloatSortBy) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }
func (a xFloatSortBy) Less(i, j int) bool { return a[i][0] < a[j][0] }

type xIntSortBy [][]int

func (a xIntSortBy) Len() int           { return len(a) }
func (a xIntSortBy) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }
func (a xIntSortBy) Less(i, j int) bool { return a[i][0] < a[j][0] }

type DetPostProcess interface {
	Run(output *paddle.ZeroCopyTensor, oriH, oriW int, ratioH, ratioW float64) [][][]int
}

type DBPostProcess struct {
	thresh        float64
	boxThresh     float64
	maxCandidates int
	unClipRatio   float64
	minSize       int
}

func NewDBPostProcess(thresh, boxThresh, unClipRatio float64) *DBPostProcess {
	return &DBPostProcess{
		thresh:        thresh,
		boxThresh:     boxThresh,
		unClipRatio:   unClipRatio,
		maxCandidates: 1000,
		minSize:       3,
	}
}

func (d *DBPostProcess) getMinBoxes(rect gocv.RotatedRect) [][]float32 {
	points := gocv.NewMat()
	gocv.BoxPoints(rect, &points)
	defer points.Close()
	array := d.mat2slice(points)
	sort.Sort(xFloatSortBy(array))

	point1, point2, point3, point4 := array[0], array[1], array[2], array[3]
	if array[3][1] <= array[2][1] {
		point2, point3 = array[3], array[2]
	} else {
		point2, point3 = array[2], array[3]
	}

	if array[1][1] <= array[0][1] {
		point1, point4 = array[1], array[0]
	} else {
		point1, point4 = array[0], array[1]
	}

	array = [][]float32{point1, point2, point3, point4}
	return array
}

func (d *DBPostProcess) mat2slice(mat gocv.Mat) [][]float32 {
	array := make([][]float32, mat.Rows())
	for i := 0; i < mat.Rows(); i++ {
		tmp := make([]float32, mat.Cols())
		for j := 0; j < mat.Cols(); j++ {
			tmp[j] = mat.GetFloatAt(i, j)
		}
		array[i] = tmp
	}
	return array
}

func (d *DBPostProcess) boxScoreFast(array [][]float32, pred gocv.Mat) float64 {
	height, width := pred.Rows(), pred.Cols()
	boxX := []float32{array[0][0], array[1][0], array[2][0], array[3][0]}
	boxY := []float32{array[0][1], array[1][1], array[2][1], array[3][1]}

	xmin := clip(int(math.Floor(float64(minf(boxX)))), 0, width-1)
	xmax := clip(int(math.Ceil(float64(maxf(boxX)))), 0, width-1)
	ymin := clip(int(math.Floor(float64(minf(boxY)))), 0, height-1)
	ymax := clip(int(math.Ceil(float64(maxf(boxY)))), 0, height-1)

	mask := gocv.NewMatWithSize(ymax-ymin+1, xmax-xmin+1, gocv.MatTypeCV8UC1)
	defer mask.Close()
	ppt := make([][]image.Point, 1)
	ppt[0] = make([]image.Point, 4)
	ppt[0][0] = image.Point{int(array[0][0]) - xmin, int(array[0][1]) - ymin}
	ppt[0][1] = image.Point{int(array[1][0]) - xmin, int(array[1][1]) - ymin}
	ppt[0][2] = image.Point{int(array[2][0]) - xmin, int(array[2][1]) - ymin}
	ppt[0][3] = image.Point{int(array[3][0]) - xmin, int(array[3][1]) - ymin}
	gocv.FillPoly(&mask, ppt, color.RGBA{0, 0, 1, 0})
	croppedImg := pred.Region(image.Rect(xmin, ymin, xmax+1, ymax+1))
	s := croppedImg.MeanWithMask(mask)
	return s.Val1
}

func (d *DBPostProcess) unClip(box [][]float32) gocv.RotatedRect {
	var area, dist float64
	for i := 0; i < 4; i++ {
		area += float64(box[i][0]*box[(i+1)%4][1] - box[i][1]*box[(i+1)%4][0])
		dist += math.Sqrt(float64(
			(box[i][0]-box[(i+1)%4][0])*(box[i][0]-box[(i+1)%4][0]) +
				(box[i][1]-box[(i+1)%4][1])*(box[i][1]-box[(i+1)%4][1]),
		))
	}
	area = math.Abs(area / 2.0)
	distance := area * d.unClipRatio / dist
	offset := clipper.NewClipperOffset()
	path := make([]*clipper.IntPoint, 4)
	path[0] = &clipper.IntPoint{X: clipper.CInt(box[0][0]), Y: clipper.CInt(box[0][1])}
	path[1] = &clipper.IntPoint{X: clipper.CInt(box[1][0]), Y: clipper.CInt(box[1][1])}
	path[2] = &clipper.IntPoint{X: clipper.CInt(box[2][0]), Y: clipper.CInt(box[2][1])}
	path[3] = &clipper.IntPoint{X: clipper.CInt(box[3][0]), Y: clipper.CInt(box[3][1])}
	offset.AddPath(clipper.Path(path), clipper.JtRound, clipper.EtClosedPolygon)
	soln := offset.Execute(distance)

	points := make([]image.Point, 0, 4)
	for i := 0; i < len(soln); i++ {
		for j := 0; j < len(soln[i]); j++ {
			points = append(points, image.Point{int(soln[i][j].X), int(soln[i][j].Y)})
		}
	}

	var res gocv.RotatedRect
	if len(points) <= 0 {
		points = make([]image.Point, 4)
		points[0] = image.Pt(0, 0)
		points[1] = image.Pt(1, 0)
		points[2] = image.Pt(1, 1)
		points[3] = image.Pt(0, 1)
		res = gocv.RotatedRect{
			Contour:      points,
			BoundingRect: image.Rect(0, 0, 1, 1),
			Center:       gocv.Point2f{X: 0.5, Y: 0.5},
			Width:        1,
			Height:       1,
			Angle:        0,
		}
	} else {
		res = gocv.MinAreaRect(points)
	}
	return res
}

func (d *DBPostProcess) boxesFromBitmap(pred gocv.Mat, mask gocv.Mat, ratioH float64, ratioW float64) [][][]int {
	height, width := mask.Rows(), mask.Cols()
	mask.MultiplyUChar(255)
	contours := gocv.FindContours(mask, gocv.RetrievalList, gocv.ChainApproxSimple)
	numContours := len(contours)
	if numContours > d.maxCandidates {
		numContours = d.maxCandidates
	}

	boxes := make([][][]int, 0, numContours)
	for i := 0; i < numContours; i++ {
		contour := contours[i]
		boundingbox := gocv.MinAreaRect(contour)
		if boundingbox.Width < float32(d.minSize) || boundingbox.Height < float32(d.minSize) {
			continue
		}
		points := d.getMinBoxes(boundingbox)
		score := d.boxScoreFast(points, pred)
		if score < d.boxThresh {
			continue
		}

		box := d.unClip(points)
		if box.Width < float32(d.minSize+2) || box.Height < float32(d.minSize+2) {
			continue
		}

		cliparray := d.getMinBoxes(box)
		dstHeight, dstWidth := pred.Rows(), pred.Cols()
		intcliparray := make([][]int, 4)
		for i := 0; i < 4; i++ {
			p := []int{
				int(float64(clip(int(math.Round(
					float64(cliparray[i][0]/float32(width)*float32(dstWidth)))), 0, dstWidth)) / ratioW),
				int(float64(clip(int(math.Round(
					float64(cliparray[i][1]/float32(height)*float32(dstHeight)))), 0, dstHeight)) / ratioH),
			}
			intcliparray[i] = p
		}
		boxes = append(boxes, intcliparray)
	}
	return boxes
}

func (d *DBPostProcess) orderPointsClockwise(box [][]int) [][]int {
	sort.Sort(xIntSortBy(box))
	leftmost := [][]int{box[0], box[1]}
	rightmost := [][]int{box[2], box[3]}

	if leftmost[0][1] > leftmost[1][1] {
		leftmost[0], leftmost[1] = leftmost[1], leftmost[0]
	}

	if rightmost[0][1] > rightmost[1][1] {
		rightmost[0], rightmost[1] = rightmost[1], rightmost[0]
	}

	return [][]int{leftmost[0], rightmost[0], rightmost[1], leftmost[1]}
}

func (d *DBPostProcess) filterTagDetRes(boxes [][][]int, oriH, oriW int) [][][]int {
	points := make([][][]int, 0, len(boxes))
	for i := 0; i < len(boxes); i++ {
		boxes[i] = d.orderPointsClockwise(boxes[i])
		for j := 0; j < len(boxes[i]); j++ {
			boxes[i][j][0] = clip(boxes[i][j][0], 0, oriW-1)
			boxes[i][j][1] = clip(boxes[i][j][1], 0, oriH-1)
		}
	}

	for i := 0; i < len(boxes); i++ {
		rectW := int(math.Sqrt(math.Pow(float64(boxes[i][0][0]-boxes[i][1][0]), 2.0) +
			math.Pow(float64(boxes[i][0][1]-boxes[i][1][1]), 2.0)))
		rectH := int(math.Sqrt(math.Pow(float64(boxes[i][0][0]-boxes[i][3][0]), 2.0) +
			math.Pow(float64(boxes[i][0][1]-boxes[i][3][1]), 2.0)))
		if rectW <= 4 || rectH <= 4 {
			continue
		}
		points = append(points, boxes[i])
	}
	return points
}

func (d *DBPostProcess) Run(output *paddle.ZeroCopyTensor, oriH, oriW int, ratioH, ratioW float64) [][][]int {
	v := output.Value().([][][][]float32)

	shape := output.Shape()
	height, width := int(shape[2]), int(shape[3])

	pred := gocv.NewMatWithSize(height, width, gocv.MatTypeCV32F)
	bitmap := gocv.NewMatWithSize(height, width, gocv.MatTypeCV8UC1)
	thresh := float32(d.thresh)
	for i := 0; i < height; i++ {
		for j := 0; j < width; j++ {
			pred.SetFloatAt(i, j, v[0][0][i][j])
			if v[0][0][i][j] > thresh {
				bitmap.SetUCharAt(i, j, 1)
			} else {
				bitmap.SetUCharAt(i, j, 0)
			}
		}
	}

	mask := gocv.NewMat()
	kernel := gocv.GetStructuringElement(gocv.MorphRect, image.Point{2, 2})
	gocv.Dilate(bitmap, &mask, kernel)
	boxes := d.boxesFromBitmap(pred, mask, ratioH, ratioW)
	dtboxes := d.filterTagDetRes(boxes, oriH, oriW)
	return dtboxes
}
