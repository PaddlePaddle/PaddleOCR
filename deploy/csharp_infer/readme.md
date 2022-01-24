# C#/.NET PaddleOCR Usage

## Windows: Detection and Recognition(All)
1. Install NuGet Packages:
   ```
   Sdcb.PaddleInference
   Sdcb.PaddleInference.runtime.win64.mkl
   Sdcb.PaddleOCR
   Sdcb.PaddleOCR.KnownModels
   OpenCvSharp4
   OpenCvSharp4.runtime.win
   ```

2. Using following C# code to get result:
   ```csharp
   OCRModel model = KnownOCRModel.PPOcrV2;
   await model.EnsureAll();
   
   byte[] sampleImageData;
   string sampleImageUrl = @"https://www.tp-link.com.cn/content/images/detail/2164/TL-XDR5450易展Turbo版-3840px_03.jpg";
   using (HttpClient http = new HttpClient())
   {
       Console.WriteLine("Download sample image from: " + sampleImageUrl);
       sampleImageData = await http.GetByteArrayAsync(sampleImageUrl);
   }
   
   using (PaddleOcrAll all = new PaddleOcrAll(model.RootDirectory, model.KeyPath)
   {
       AllowRotateDetection = true, /* 允许识别有角度的文字 */ 
       Enable180Classification = false, /* 允许识别旋转角度大于90度的文字 */
   })
   {
       // Load local file by following code:
       // using (Mat src2 = Cv2.ImRead(@"C:\test.jpg"))
       using (Mat src = Cv2.ImDecode(sampleImageData, ImreadModes.Color))
       {
           PaddleOcrResult result = all.Run(src);
           Console.WriteLine("Detected all texts: \n" + result.Text);
           foreach (PaddleOcrResultRegion region in result.Regions)
           {
               Console.WriteLine($"Text: {region.Text}, Score: {region.Score}, RectCenter: {region.Rect.Center}, RectSize:    {region.Rect.Size}, Angle: {region.Rect.Angle}");
           }
       }
   }
   ```

## Linux(Ubuntu 20.04): Detection and Recognition(All)
1. Use `sdflysha/ubuntu20-dotnet6-paddleocr2.2.1:20211223` to replace `mcr.microsoft.com/dotnet/aspnet:6.0` in `Dockerfile` as docker base image.

The build steps for `ubuntu20-dotnet6-paddleocr` was described [here](./build/docker/ubuntu20-dotnet6-paddleocr2.2.1/Dockerfile).

And also, we also provided another dotnet6-sdk `Dockerfile`, described [here](./build/docker/ubuntu20-dotnet6sdk-paddleocr2.2.1/Dockerfile).

2. Install NuGet Packages:
```ps
dotnet add package Sdcb.PaddleInference
dotnet add package Sdcb.PaddleOCR
dotnet add package Sdcb.PaddleOCR.KnownModels
dotnet add package OpenCvSharp4
dotnet add package OpenCvSharp4.runtime.ubuntu.18.04-x64
```

Please aware in `Linux`, the native binding library is not required, instead, you should compile your own `OpenCV`/`PaddleInference` library, or just use the `Docker` image.

3. write following C# code to get result(also can be exactly the same as windows):
```csharp
OCRModel model = KnownOCRModel.PPOcrV2;
await model.EnsureAll();
using (PaddleOcrAll all = new PaddleOcrAll(model.RootDirectory, model.KeyPath))
// Load in-memory data by following code:
// using (Mat src = Cv2.ImDecode(sampleImageData, ImreadModes.Color))
using (Mat src = Cv2.ImRead(@"/app/test.jpg"))
{
    Console.WriteLine(all.Run(src).Text);
}
```

## Detection Only
```csharp
// Install following packages:
// Sdcb.PaddleInference
// Sdcb.PaddleInference.runtime.win64.mkl (required in Windows)
// Sdcb.PaddleOCR
// Sdcb.PaddleOCR.KnownModels
// OpenCvSharp4
// OpenCvSharp4.runtime.win (required in Windows)
// OpenCvSharp4.runtime.linux18.04 (required in Linux)
byte[] sampleImageData;
string sampleImageUrl = @"https://www.tp-link.com.cn/content/images/detail/2164/TL-XDR5450易展Turbo版-3840px_03.jpg";
using (HttpClient http = new HttpClient())
{
    Console.WriteLine("Download sample image from: " + sampleImageUrl);
    sampleImageData = await http.GetByteArrayAsync(sampleImageUrl);
}

OCRModel model = KnownOCRModel.PPOcrV2;
await model.EnsureAll();
using (PaddleOcrDetector detector = new PaddleOcrDetector(model.DetectionDirectory))
using (Mat src = Cv2.ImDecode(sampleImageData, ImreadModes.Color))
{
    RotatedRect[] rects = detector.Run(src);
    using (Mat visualized = PaddleOcrDetector.Visualize(src, rects, Scalar.Red, thickness: 2))
    {
        string outputFile = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.MyPictures), "output.jpg");
        Console.WriteLine("OutputFile: " + outputFile);
        visualized.ImWrite(outputFile);
    }
}
```

# Language supports

| Language             | 中文名             | Code                               |
| -------------------- | ------------------ | ---------------------------------- |
| Chinese              | 简体中文           | `KnownOCRModel.PPOcrV2`            |
| Chinese Server       | 简体中文(服务器版) | `KnownOCRModel.PPOcrServerV2`      |
| English              | 英文               | `KnownOCRModel.EnglishMobileV2`    |
| Tranditional Chinese | 繁体中文           | `KnownOCRModel.EnglishMobileV2`    |
| French               | 法文               | `KnownOCRModel.FrenchMobileV2`     |
| German               | 德文               | `KnownOCRModel.GermanMobileV2`     |
| Korean               | 韩文               | `KnownOCRModel.KoreanMobileV2`     |
| Japanese             | 日文               | `KnownOCRModel.JapaneseMobileV2`   |
| Telugu               | 泰卢固文           | `KnownOCRModel.TeluguMobileV2`     |
| Kannada              | 卡纳达文           | `KnownOCRModel.KannadaMobileV2`    |
| Tamil                | 泰米尔文           | `KnownOCRModel.TamilMobileV2`      |
| Latin                | 拉丁文             | `KnownOCRModel.LatinMobileV2`      |
| Arabic               | 阿拉伯字母         | `KnownOCRModel.ArabicMobileV2`     |
| Cyrillic             | 斯拉夫字母         | `KnownOCRModel.CyrillicMobileV2`   |
| Devanagari           | 梵文字母           | `KnownOCRModel.DevanagariMobileV2` |

Just replace the `KnownOCRModel.PPOcrV2` in demo code with your speicific language in `Code` column above, then you can use the language.

# Technical details

There is 3 steps to do OCR:
1. Detection - Detect text's position, angle and area (`PaddleOCRDetector`)
2. Classification - Determin whether text should rotate 180 degreee.
3. Recognization - Recognize the area into text

# Optimize parameters and performance hints
## PaddleOcrAll.Enable180Classification
Default value: `false`

This directly effect the step 2, set to `false` can skip this step, which will unable to detect text from right to left(which should be acceptable because most text direction is from left to right).

Close this option can make the full process about  `~10%` faster.


## PaddleOcrAll.AllowRotateDetection
Default value: `true`

This allows detect any rotated texts. If your subject is 0 degree text (like scaned table or screenshot), you can set this parameter to `false`, which will improve OCR accurancy and little bit performance.


## PaddleOcrAll.Detector.MaxSize
Default value: `2048`

This effect the the max size of step #1, lower this value can improve performance and reduce memory usage, but will also lower the accurancy.

You can also set this value to `null`, in that case, images will not scale-down to detect, performance will drop and memory will high, but should able to get better accurancy.

## PaddleConfig.Defaults.UseGpu
Default value: `false`

If you wants to use GPU, you should refer to FAQ `How to enable GPU?` section, CUDA/cuDNN/TensorRT need to be installed manually.

## How can I improve performance?
Please review the `Technical details` section and read the `Optimize parameters and performance hints` section, or UseGpu.