package ocr

import (
	"archive/tar"
	"io"
	"io/ioutil"
	"log"
	"net/http"
	"os"
	"path"
	"path/filepath"
	"strings"

	"github.com/LKKlein/gocv"
	"gopkg.in/yaml.v3"
)

func getString(args map[string]interface{}, key string, dv string) string {
	if f, ok := args[key]; ok {
		return f.(string)
	}
	return dv
}

func getFloat64(args map[string]interface{}, key string, dv float64) float64 {
	if f, ok := args[key]; ok {
		return f.(float64)
	}
	return dv
}

func getInt(args map[string]interface{}, key string, dv int) int {
	if i, ok := args[key]; ok {
		return i.(int)
	}
	return dv
}

func getBool(args map[string]interface{}, key string, dv bool) bool {
	if b, ok := args[key]; ok {
		return b.(bool)
	}
	return dv
}

func ReadImage(image_path string) gocv.Mat {
	img := gocv.IMRead(image_path, gocv.IMReadColor)
	if img.Empty() {
		log.Printf("Could not read image %s\n", image_path)
		os.Exit(1)
	}
	return img
}

func clip(value, min, max int) int {
	if value <= min {
		return min
	} else if value >= max {
		return max
	}
	return value
}

func minf(data []float32) float32 {
	v := data[0]
	for _, val := range data {
		if val < v {
			v = val
		}
	}
	return v
}

func maxf(data []float32) float32 {
	v := data[0]
	for _, val := range data {
		if val > v {
			v = val
		}
	}
	return v
}

func mini(data []int) int {
	v := data[0]
	for _, val := range data {
		if val < v {
			v = val
		}
	}
	return v
}

func maxi(data []int) int {
	v := data[0]
	for _, val := range data {
		if val > v {
			v = val
		}
	}
	return v
}

func argmax(arr []float32) (int, float32) {
	max_value, index := arr[0], 0
	for i, item := range arr {
		if item > max_value {
			max_value = item
			index = i
		}
	}
	return index, max_value
}

func checkModelExists(modelPath string) bool {
	if isPathExist(modelPath+"/model") && isPathExist(modelPath+"/params") {
		return true
	}
	if strings.HasPrefix(modelPath, "http://") ||
		strings.HasPrefix(modelPath, "ftp://") || strings.HasPrefix(modelPath, "https://") {
		return true
	}
	return false
}

func downloadFile(filepath, url string) error {
	resp, err := http.Get(url)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	out, err := os.Create(filepath)
	if err != nil {
		return err
	}
	defer out.Close()

	_, err = io.Copy(out, resp.Body)
	log.Println("[download_file] from:", url, " to:", filepath)
	return err
}

func isPathExist(path string) bool {
	if _, err := os.Stat(path); err == nil {
		return true
	} else if os.IsNotExist(err) {
		return false
	}
	return false
}

func downloadModel(modelDir, modelPath string) (string, error) {
	if modelPath != "" && (strings.HasPrefix(modelPath, "http://") ||
		strings.HasPrefix(modelPath, "ftp://") || strings.HasPrefix(modelPath, "https://")) {
		if checkModelExists(modelDir) {
			return modelDir, nil
		}
		_, suffix := path.Split(modelPath)
		outPath := filepath.Join(modelDir, suffix)
		outDir := filepath.Dir(outPath)
		if !isPathExist(outDir) {
			os.MkdirAll(outDir, os.ModePerm)
		}

		if !isPathExist(outPath) {
			err := downloadFile(outPath, modelPath)
			if err != nil {
				return "", err
			}
		}

		if strings.HasSuffix(outPath, ".tar") && !checkModelExists(modelDir) {
			unTar(modelDir, outPath)
			os.Remove(outPath)
			return modelDir, nil
		}
		return modelDir, nil
	}
	return modelPath, nil
}

func unTar(dst, src string) (err error) {
	fr, err := os.Open(src)
	if err != nil {
		return err
	}
	defer fr.Close()

	tr := tar.NewReader(fr)
	for {
		hdr, err := tr.Next()

		switch {
		case err == io.EOF:
			return nil
		case err != nil:
			return err
		case hdr == nil:
			continue
		}

		var dstFileDir string
		if strings.Contains(hdr.Name, "model") {
			dstFileDir = filepath.Join(dst, "model")
		} else if strings.Contains(hdr.Name, "params") {
			dstFileDir = filepath.Join(dst, "params")
		}

		switch hdr.Typeflag {
		case tar.TypeDir:
			continue
		case tar.TypeReg:
			file, err := os.OpenFile(dstFileDir, os.O_CREATE|os.O_RDWR, os.FileMode(hdr.Mode))
			if err != nil {
				return err
			}
			_, err2 := io.Copy(file, tr)
			if err2 != nil {
				return err2
			}
			file.Close()
		}
	}

	return nil
}

func readLines2StringSlice(filepath string) []string {
	if strings.HasPrefix(filepath, "http://") || strings.HasPrefix(filepath, "https://") {
		home, _ := os.UserHomeDir()
		dir := home + "/.paddleocr/rec/"
		_, suffix := path.Split(filepath)
		f := dir + suffix
		if !isPathExist(f) {
			err := downloadFile(f, filepath)
			if err != nil {
				log.Println("download ppocr key file error! You can specify your local dict path by conf.yaml.")
				return nil
			}
		}
		filepath = f
	}
	content, err := ioutil.ReadFile(filepath)
	if err != nil {
		log.Println("read ppocr key file error!")
		return nil
	}
	lines := strings.Split(string(content), "\n")
	return lines
}

func ReadYaml(yamlPath string) (map[string]interface{}, error) {
	data, err := ioutil.ReadFile(yamlPath)
	if err != nil {
		return nil, err
	}
	var body interface{}
	if err := yaml.Unmarshal(data, &body); err != nil {
		return nil, err
	}

	body = convertYaml2Map(body)
	return body.(map[string]interface{}), nil
}

func convertYaml2Map(i interface{}) interface{} {
	switch x := i.(type) {
	case map[interface{}]interface{}:
		m2 := map[string]interface{}{}
		for k, v := range x {
			m2[k.(string)] = convertYaml2Map(v)
		}
		return m2
	case []interface{}:
		for i, v := range x {
			x[i] = convertYaml2Map(v)
		}
	}
	return i
}
