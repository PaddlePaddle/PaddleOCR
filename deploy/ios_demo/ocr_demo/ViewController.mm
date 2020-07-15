//
// Created by lvxiangxiang on 2020/7/10.
// Copyright (c) 2020 baidu. All rights reserved.
//

#import <opencv2/opencv.hpp>
#import <opencv2/imgcodecs/ios.h>
#import <opencv2/videoio/cap_ios.h>
//#import <opencv2/highgui/ios.h>
#import "ViewController.h"
#import "BoxLayer.h"

#include "include/paddle_api.h"
#include "timer.h"
#import "pdocr/ocr_db_post_process.h"
#import "pdocr/ocr_crnn_process.h"

using namespace paddle::lite_api;
using namespace cv;

struct Object {
    int batch_id;
    cv::Rect rec;
    int class_id;
    float prob;
};

std::mutex mtx;
std::shared_ptr<PaddlePredictor> net_ocr1;
std::shared_ptr<PaddlePredictor> net_ocr2;
Timer tic;
long long count = 0;

double tensor_mean(const Tensor &tin) {
    auto shape = tin.shape();
    int64_t size = 1;
    for (int i = 0; i < shape.size(); i++) {
        size *= shape[i];
    }
    double mean = 0.;
    auto ptr = tin.data<float>();
    for (int i = 0; i < size; i++) {
        mean += ptr[i];
    }
    return mean / size;
}

cv::Mat resize_img_type0(const cv::Mat &img, int max_size_len, float *ratio_h, float *ratio_w) {
    int w = img.cols;
    int h = img.rows;

    float ratio = 1.f;
    int max_wh = w >= h ? w : h;
    if (max_wh > max_size_len) {
        if (h > w) {
            ratio = float(max_size_len) / float(h);
        } else {
            ratio = float(max_size_len) / float(w);
        }
    }

    int resize_h = int(float(h) * ratio);
    int resize_w = int(float(w) * ratio);
    if (resize_h % 32 == 0)
        resize_h = resize_h;
    else if (resize_h / 32 < 1)
        resize_h = 32;
    else
        resize_h = (resize_h / 32 - 1) * 32;

    if (resize_w % 32 == 0)
        resize_w = resize_w;
    else if (resize_w / 32 < 1)
        resize_w = 32;
    else
        resize_w = (resize_w / 32 - 1) * 32;

    cv::Mat resize_img;
    cv::resize(img, resize_img, cv::Size(resize_w, resize_h));

    *ratio_h = float(resize_h) / float(h);
    *ratio_w = float(resize_w) / float(w);
    return resize_img;
}

void neon_mean_scale(const float *din, float *dout, int size, std::vector<float> mean, std::vector<float> scale) {
    float32x4_t vmean0 = vdupq_n_f32(mean[0]);
    float32x4_t vmean1 = vdupq_n_f32(mean[1]);
    float32x4_t vmean2 = vdupq_n_f32(mean[2]);
    float32x4_t vscale0 = vdupq_n_f32(1.f / scale[0]);
    float32x4_t vscale1 = vdupq_n_f32(1.f / scale[1]);
    float32x4_t vscale2 = vdupq_n_f32(1.f / scale[2]);

    float *dout_c0 = dout;
    float *dout_c1 = dout + size;
    float *dout_c2 = dout + size * 2;

    int i = 0;
    for (; i < size - 3; i += 4) {
        float32x4x3_t vin3 = vld3q_f32(din);
        float32x4_t vsub0 = vsubq_f32(vin3.val[0], vmean0);
        float32x4_t vsub1 = vsubq_f32(vin3.val[1], vmean1);
        float32x4_t vsub2 = vsubq_f32(vin3.val[2], vmean2);
        float32x4_t vs0 = vmulq_f32(vsub0, vscale0);
        float32x4_t vs1 = vmulq_f32(vsub1, vscale1);
        float32x4_t vs2 = vmulq_f32(vsub2, vscale2);
        vst1q_f32(dout_c0, vs0);
        vst1q_f32(dout_c1, vs1);
        vst1q_f32(dout_c2, vs2);

        din += 12;
        dout_c0 += 4;
        dout_c1 += 4;
        dout_c2 += 4;
    }
    for (; i < size; i++) {
        *(dout_c0++) = (*(din++) - mean[0]) / scale[0];
        *(dout_c1++) = (*(din++) - mean[1]) / scale[1];
        *(dout_c2++) = (*(din++) - mean[2]) / scale[2];
    }
}


// fill tensor with mean and scale, neon speed up
void fill_tensor_with_cvmat(const Mat &img_in, Tensor &tout, int width, int height,
        std::vector<float> mean, std::vector<float> scale, bool is_scale) {
    if (img_in.channels() == 4) {
        cv::cvtColor(img_in, img_in, CV_RGBA2RGB);
    }
    cv::Mat im;
    cv::resize(img_in, im, cv::Size(width, height), 0.f, 0.f);
    cv::Mat imgf;
    float scale_factor = is_scale ? 1 / 255.f : 1.f;
    im.convertTo(imgf, CV_32FC3, scale_factor);
    const float *dimg = reinterpret_cast<const float *>(imgf.data);
    float *dout = tout.mutable_data<float>();
    neon_mean_scale(dimg, dout, width * height, mean, scale);
}

std::vector<Object> detect_object(const float *data,
        int count,
        const std::vector<std::vector<uint64_t>> &lod,
        const float thresh,
        Mat &image) {
    std::vector<Object> rect_out;
    const float *dout = data;
    for (int iw = 0; iw < count; iw++) {
        int oriw = image.cols;
        int orih = image.rows;
        if (dout[1] > thresh && static_cast<int>(dout[0]) > 0) {
            Object obj;
            int x = static_cast<int>(dout[2] * oriw);
            int y = static_cast<int>(dout[3] * orih);
            int w = static_cast<int>(dout[4] * oriw) - x;
            int h = static_cast<int>(dout[5] * orih) - y;
            cv::Rect rec_clip = cv::Rect(x, y, w, h) & cv::Rect(0, 0, image.cols, image.rows);
            obj.batch_id = 0;
            obj.class_id = static_cast<int>(dout[0]);
            obj.prob = dout[1];
            obj.rec = rec_clip;
            if (w > 0 && h > 0 && obj.prob <= 1) {
                rect_out.push_back(obj);
                cv::rectangle(image, rec_clip, cv::Scalar(255, 0, 0));
            }
        }
        dout += 6;
    }
    return rect_out;
}

@interface ViewController () <CvVideoCameraDelegate>
@property(weak, nonatomic) IBOutlet UIImageView *imageView;
@property(weak, nonatomic) IBOutlet UISwitch *flag_process;
@property(weak, nonatomic) IBOutlet UISwitch *flag_video;
@property(weak, nonatomic) IBOutlet UIImageView *preView;
@property(weak, nonatomic) IBOutlet UISwitch *flag_back_cam;
@property(weak, nonatomic) IBOutlet UILabel *result;
@property(nonatomic, strong) CvVideoCamera *videoCamera;
@property(nonatomic, strong) UIImage *image;
@property(nonatomic) bool flag_init;
@property(nonatomic) bool flag_cap_photo;
@property(nonatomic) std::vector<float> scale;
@property(nonatomic) std::vector<float> mean;
@property(nonatomic) NSArray *labels;
@property(nonatomic) cv::Mat cvimg;
@property(nonatomic, strong) UIImage *ui_img_test;
@property(strong, nonatomic) CALayer *boxLayer;

@end

@implementation ViewController
@synthesize imageView;

- (OcrData *)paddleOcrRec:(cv::Mat)image {

    OcrData *result = [OcrData new];

    std::vector<float> mean = {0.5f, 0.5f, 0.5f};
    std::vector<float> scale = {1 / 0.5f, 1 / 0.5f, 1 / 0.5f};

    cv::Mat crop_img;
    image.copyTo(crop_img);
    cv::Mat resize_img;


    float wh_ratio = float(crop_img.cols) / float(crop_img.rows);

    resize_img = crnn_resize_img(crop_img, wh_ratio);
    resize_img.convertTo(resize_img, CV_32FC3, 1 / 255.f);

    const float *dimg = reinterpret_cast<const float *>(resize_img.data);

    std::unique_ptr<Tensor> input_tensor0(std::move(net_ocr2->GetInput(0)));
    input_tensor0->Resize({1, 3, resize_img.rows, resize_img.cols});
    auto *data0 = input_tensor0->mutable_data<float>();

    neon_mean_scale(dimg, data0, resize_img.rows * resize_img.cols, mean, scale);

    //// Run CRNN predictor
    net_ocr2->Run();

    // Get output and run postprocess
    std::unique_ptr<const Tensor> output_tensor0(std::move(net_ocr2->GetOutput(0)));
    auto *rec_idx = output_tensor0->data<int>();

    auto rec_idx_lod = output_tensor0->lod();
    auto shape_out = output_tensor0->shape();
    NSMutableString *text = [[NSMutableString alloc] init];
    for (int n = int(rec_idx_lod[0][0]); n < int(rec_idx_lod[0][1] * 2); n += 2) {
        if (rec_idx[n] >= self.labels.count) {
            std::cout << "Index " << rec_idx[n] << " out of text dict range!" << std::endl;
            continue;
        }
        [text appendString:self.labels[rec_idx[n]]];
    }

    result.label = text;
    // get score
    std::unique_ptr<const Tensor> output_tensor1(std::move(net_ocr2->GetOutput(1)));
    auto *predict_batch = output_tensor1->data<float>();
    auto predict_shape = output_tensor1->shape();

    auto predict_lod = output_tensor1->lod();

    int argmax_idx;
    int blank = predict_shape[1];
    float score = 0.f;
    int count = 0;
    float max_value = 0.0f;

    for (int n = predict_lod[0][0]; n < predict_lod[0][1] - 1; n++) {
        argmax_idx = int(argmax(&predict_batch[n * predict_shape[1]], &predict_batch[(n + 1) * predict_shape[1]]));
        max_value = float(*std::max_element(&predict_batch[n * predict_shape[1]], &predict_batch[(n + 1) * predict_shape[1]]));

        if (blank - 1 - argmax_idx > 1e-5) {
            score += max_value;
            count += 1;
        }

    }
    score /= count;
    result.accuracy = score;
    return result;
}
- (NSArray *) ocr_infer:(cv::Mat) originImage{
    int max_side_len = 960;
    float ratio_h{};
    float ratio_w{};
    cv::Mat image;
    cv::cvtColor(originImage, image, cv::COLOR_RGB2BGR);

    cv::Mat img;
    image.copyTo(img);

    img = resize_img_type0(img, max_side_len, &ratio_h, &ratio_w);
    cv::Mat img_fp;
    img.convertTo(img_fp, CV_32FC3, 1.0 / 255.f);

    std::unique_ptr<Tensor> input_tensor(net_ocr1->GetInput(0));
    input_tensor->Resize({1, 3, img_fp.rows, img_fp.cols});
    auto *data0 = input_tensor->mutable_data<float>();
    const float *dimg = reinterpret_cast<const float *>(img_fp.data);
    neon_mean_scale(dimg, data0, img_fp.rows * img_fp.cols, self.mean, self.scale);
    tic.clear();
    tic.start();
    net_ocr1->Run();
    std::unique_ptr<const Tensor> output_tensor(std::move(net_ocr1->GetOutput(0)));
    auto *outptr = output_tensor->data<float>();
    auto shape_out = output_tensor->shape();

    int64_t out_numl = 1;
    double sum = 0;
    for (auto i : shape_out) {
        out_numl *= i;
    }

    int s2 = int(shape_out[2]);
    int s3 = int(shape_out[3]);

    cv::Mat pred_map = cv::Mat::zeros(s2, s3, CV_32F);
    memcpy(pred_map.data, outptr, s2 * s3 * sizeof(float));
    cv::Mat cbuf_map;
    pred_map.convertTo(cbuf_map, CV_8UC1, 255.0f);

    const double threshold = 0.1 * 255;
    const double maxvalue = 255;
    cv::Mat bit_map;
    cv::threshold(cbuf_map, bit_map, threshold, maxvalue, cv::THRESH_BINARY);

    auto boxes = boxes_from_bitmap(pred_map, bit_map);

    std::vector<std::vector<std::vector<int>>> filter_boxes = filter_tag_det_res(boxes, ratio_h, ratio_w, image);


    cv::Point rook_points[filter_boxes.size()][4];

    for (int n = 0; n < filter_boxes.size(); n++) {
        for (int m = 0; m < filter_boxes[0].size(); m++) {
            rook_points[n][m] = cv::Point(int(filter_boxes[n][m][0]), int(filter_boxes[n][m][1]));
        }
    }

    NSMutableArray *result = [[NSMutableArray alloc] init];

    for (int i = 0; i < filter_boxes.size(); i++) {
        cv::Mat crop_img;
        crop_img = get_rotate_crop_image(image, filter_boxes[i]);
        OcrData *r = [self paddleOcrRec:crop_img ];
        NSMutableArray *points = [NSMutableArray new];
        for (int jj = 0; jj < 4; ++jj) {
            NSValue *v = [NSValue valueWithCGPoint:CGPointMake(
                    rook_points[i][jj].x / CGFloat(originImage.cols),
                    rook_points[i][jj].y / CGFloat(originImage.rows))];
            [points addObject:v];
        }
        r.polygonPoints = points;
        [result addObject:r];
    }
    NSArray* rec_out =[[result reverseObjectEnumerator] allObjects];
    tic.end();
    std::cout<<"infer time: "<<tic.get_sum_ms()<<"ms"<<std::endl;
    return rec_out;
}
- (NSArray *)readLabelsFromFile:(NSString *)labelFilePath {

    NSString *content = [NSString stringWithContentsOfFile:labelFilePath encoding:NSUTF8StringEncoding error:nil];
    NSArray *lines = [content componentsSeparatedByCharactersInSet:[NSCharacterSet newlineCharacterSet]];
    NSMutableArray *ret = [[NSMutableArray alloc] init];
    for (int i = 0; i < lines.count; ++i) {
        [ret addObject:@""];
    }
    NSUInteger cnt = 0;
    for (id line in lines) {
        NSString *l = [(NSString *) line stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceAndNewlineCharacterSet]];
        if ([l length] == 0)
            continue;
        NSArray *segs = [l componentsSeparatedByString:@":"];
        NSUInteger key;
        NSString *value;
        if ([segs count] != 2) {
            key = cnt;
            value = [segs[0] stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceAndNewlineCharacterSet]];
        } else {
            key = [[segs[0] stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceAndNewlineCharacterSet]] integerValue];
            value = [segs[1] stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceAndNewlineCharacterSet]];
        }

        ret[key] = value;
        cnt += 1;
    }
    return [NSArray arrayWithArray:ret];
}

- (void)viewDidAppear:(BOOL)animated {
    [super viewDidAppear:animated];
    self.boxLayer = [[CALayer alloc] init];

    CGRect r = AVMakeRectWithAspectRatioInsideRect(self.imageView.frame.size, self.imageView.bounds);
    std::cout<<self.imageView.frame.size.width<<","<<self.imageView.frame.size.height<<std::endl;
    self.boxLayer.frame = r;

    [self.imageView.layer addSublayer:self.boxLayer];

    NSString *label_file_path = [[NSBundle mainBundle] pathForResource:[NSString stringWithFormat:@"%@", @"label_list"] ofType:@"txt"];

    self.labels = [self readLabelsFromFile:label_file_path];
    self.mean = {0.485f, 0.456f, 0.406f};
    self.scale = {1 / 0.229f, 1 / 0.224f, 1 / 0.225f};

    NSString *model1_path = [[NSBundle mainBundle] pathForResource:[NSString stringWithFormat:@"%@", @"ch_det_mv3_db_opt"] ofType:@"nb"];
    NSString *model2_path = [[NSBundle mainBundle] pathForResource:[NSString stringWithFormat:@"%@", @"ch_rec_mv3_crnn_opt"] ofType:@"nb"];
    std::string model1_path_str = std::string([model1_path UTF8String]);
    std::string model2_path_str =  std::string([model2_path UTF8String]);
    MobileConfig config;
    config.set_model_from_file(model1_path_str);
    net_ocr1 = CreatePaddlePredictor<MobileConfig>(config);
    MobileConfig config2;
    config2.set_model_from_file(model2_path_str);
    net_ocr2 = CreatePaddlePredictor<MobileConfig>(config2);


    cv::Mat originImage;
    UIImageToMat(self.image, originImage);
    NSArray *rec_out = [self ocr_infer:originImage];
    [_boxLayer.sublayers makeObjectsPerformSelector:@selector(removeFromSuperlayer)];
    std::cout<<self.imageView.image.size.width<<","<<self.imageView.image.size.height<<std::endl;

    CGFloat h = _boxLayer.frame.size.height;
    CGFloat w = _boxLayer.frame.size.width;
    std::ostringstream result2;
    NSInteger cnt = 0;
    for (id obj in rec_out) {
        OcrData *data = obj;
        BoxLayer *singleBox = [[BoxLayer alloc] init];
        [singleBox renderOcrPolygon:data withHeight:h withWidth:w withLabel:YES];
        [_boxLayer addSublayer:singleBox];
        result2<<[data.label UTF8String] <<","<<data.accuracy<<"\n";
        cnt += 1;
    }
    self.flag_init = true;
}

- (void)viewDidLoad {
    [super viewDidLoad];
    // Do any additional setup after loading the view, typically from a nib.
    _flag_process.on = NO;
    _flag_back_cam.on = NO;
    _flag_video.on = NO;
    _flag_cap_photo = false;
    _image = [UIImage imageNamed:@"ocr.png"];
    if (_image != nil) {
        printf("load image successed\n");
        imageView.image = _image;
    } else {
        printf("load image failed\n");
    }

    [_flag_process addTarget:self action:@selector(PSwitchValueChanged:) forControlEvents:UIControlEventValueChanged];
    [_flag_back_cam addTarget:self action:@selector(CSwitchValueChanged:) forControlEvents:UIControlEventValueChanged];

    self.videoCamera = [[CvVideoCamera alloc] initWithParentView:self.preView];
    self.videoCamera.delegate = self;
    self.videoCamera.defaultAVCaptureDevicePosition = AVCaptureDevicePositionFront;
    self.videoCamera.defaultAVCaptureSessionPreset = AVCaptureSessionPreset1920x1080;
    self.videoCamera.defaultAVCaptureVideoOrientation = AVCaptureVideoOrientationPortrait;
    self.videoCamera.rotateVideo = 90;
    self.videoCamera.defaultFPS = 30;
    [self.view insertSubview:self.imageView atIndex:0];
}

- (IBAction)swith_video_photo:(UISwitch *)sender {
    NSLog(@"%@", sender.isOn ? @"video ON" : @"video OFF");
    if (sender.isOn) {
        self.flag_video.on = YES;
    } else {
        self.flag_video.on = NO;
    }
}

- (IBAction)cap_photo:(id)sender {
    if (!self.flag_process.isOn) {
        self.result.text = @"please turn on the camera firstly";
    } else {
        self.flag_cap_photo = true;
    }
}

- (void)PSwitchValueChanged:(UISwitch *)sender {
    NSLog(@"%@", sender.isOn ? @"process ON" : @"process OFF");
    if (sender.isOn) {
        [self.videoCamera start];
    } else {
        [self.videoCamera stop];
    }
}

- (void)CSwitchValueChanged:(UISwitch *)sender {
    NSLog(@"%@", sender.isOn ? @"back ON" : @"back OFF");
    if (sender.isOn) {
        if (self.flag_process.isOn) {
            [self.videoCamera stop];
        }
        self.videoCamera.defaultAVCaptureDevicePosition = AVCaptureDevicePositionBack ;
        if (self.flag_process.isOn) {
            [self.videoCamera start];
        }
    } else {
        if (self.flag_process.isOn) {
            [self.videoCamera stop];
        }
        self.videoCamera.defaultAVCaptureDevicePosition = AVCaptureDevicePositionFront;
        if (self.flag_process.isOn) {
            [self.videoCamera start];
        }
    }
}

- (void)processImage:(cv::Mat &)image {

    dispatch_async(dispatch_get_main_queue(), ^{
        if (self.flag_process.isOn) {
            if (self.flag_init) {
                if (self.flag_video.isOn || self.flag_cap_photo) {
                    self.flag_cap_photo = false;
                    if (image.channels() == 4) {
                        cvtColor(image, self->_cvimg, CV_RGBA2RGB);
                    }
                    auto rec_out =[self ocr_infer:self->_cvimg];
                    std::ostringstream result;
                    NSInteger cnt = 0;
                    [_boxLayer.sublayers makeObjectsPerformSelector:@selector(removeFromSuperlayer)];

                    CGFloat h = _boxLayer.frame.size.height;
                    CGFloat w = _boxLayer.frame.size.width;
                    for (id obj in rec_out) {
                        OcrData *data = obj;
                        BoxLayer *singleBox = [[BoxLayer alloc] init];
                        [singleBox renderOcrPolygon:data withHeight:h withWidth:w withLabel:YES];
                        [_boxLayer addSublayer:singleBox];
                        result<<[data.label UTF8String] <<","<<data.accuracy<<"\n";
                        cnt += 1;
                    }
                    cvtColor(self->_cvimg, self->_cvimg, CV_RGB2BGR);
                    self.imageView.image = MatToUIImage(self->_cvimg);
                }
            }
        }
    });
}

- (void)didReceiveMemoryWarning {
    [super didReceiveMemoryWarning];
    // Dispose of any resources that can be recreated.
}

@end
