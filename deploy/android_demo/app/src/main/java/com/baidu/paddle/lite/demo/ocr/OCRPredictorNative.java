package com.baidu.paddle.lite.demo.ocr;

import android.graphics.Bitmap;
import android.util.Log;

import java.util.ArrayList;
import java.util.concurrent.atomic.AtomicBoolean;

public class OCRPredictorNative {

    private static final AtomicBoolean isSOLoaded = new AtomicBoolean();

    public static void loadLibrary() throws RuntimeException {
        if (!isSOLoaded.get() && isSOLoaded.compareAndSet(false, true)) {
            try {
                System.loadLibrary("Native");
            } catch (Throwable e) {
                RuntimeException exception = new RuntimeException(
                        "Load libNative.so failed, please check it exists in apk file.", e);
                throw exception;
            }
        }
    }

    private Config config;

    private long nativePointer = 0;

    public OCRPredictorNative(Config config) {
        this.config = config;
        loadLibrary();
        nativePointer = init(config.detModelFilename, config.recModelFilename,
                config.cpuThreadNum, config.cpuPower);
        Log.i("OCRPredictorNative", "load success " + nativePointer);

    }

    public void release(){
        if (nativePointer != 0){
            nativePointer = 0;
            destory(nativePointer);
        }
    }

    public ArrayList<OcrResultModel> runImage(float[] inputData, int width, int height, int channels, Bitmap originalImage) {
        Log.i("OCRPredictorNative", "begin to run image " + inputData.length + " " + width + " " + height);
        float[] dims = new float[]{1, channels, height, width};
        float[] rawResults = forward(nativePointer, inputData, dims, originalImage);
        ArrayList<OcrResultModel> results = postprocess(rawResults);
        return results;
    }

    public static class Config {
        public int cpuThreadNum;
        public String cpuPower;
        public String detModelFilename;
        public String recModelFilename;

    }

    protected native long init(String detModelPath, String recModelPath, int threadNum, String cpuMode);

    protected native float[] forward(long pointer, float[] buf, float[] ddims, Bitmap originalImage);

    protected native void destory(long pointer);

    private ArrayList<OcrResultModel> postprocess(float[] raw) {
        ArrayList<OcrResultModel> results = new ArrayList<OcrResultModel>();
        int begin = 0;

        while (begin < raw.length) {
            int point_num = Math.round(raw[begin]);
            int word_num = Math.round(raw[begin + 1]);
            OcrResultModel model = parse(raw, begin + 2, point_num, word_num);
            begin += 2 + 1 + point_num * 2 + word_num;
            results.add(model);
        }

        return results;
    }

    private OcrResultModel parse(float[] raw, int begin, int pointNum, int wordNum) {
        int current = begin;
        OcrResultModel model = new OcrResultModel();
        model.setConfidence(raw[current]);
        current++;
        for (int i = 0; i < pointNum; i++) {
            model.addPoints(Math.round(raw[current + i * 2]), Math.round(raw[current + i * 2 + 1]));
        }
        current += (pointNum * 2);
        for (int i = 0; i < wordNum; i++) {
            int index = Math.round(raw[current + i]);
            model.addWordIndex(index);
        }
        Log.i("OCRPredictorNative", "word finished " + wordNum);
        return model;
    }


}
