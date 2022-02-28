package com.baidu.paddle.lite.demo.ocr;

import android.graphics.Point;

import java.util.ArrayList;
import java.util.List;

public class OcrResultModel {
    private List<Point> points;
    private List<Integer> wordIndex;
    private String label;
    private float confidence;
    private float cls_idx;
    private String cls_label;
    private float cls_confidence;

    public OcrResultModel() {
        super();
        points = new ArrayList<>();
        wordIndex = new ArrayList<>();
    }

    public void addPoints(int x, int y) {
        Point point = new Point(x, y);
        points.add(point);
    }

    public void addWordIndex(int index) {
        wordIndex.add(index);
    }

    public List<Point> getPoints() {
        return points;
    }

    public List<Integer> getWordIndex() {
        return wordIndex;
    }

    public String getLabel() {
        return label;
    }

    public void setLabel(String label) {
        this.label = label;
    }

    public float getConfidence() {
        return confidence;
    }

    public void setConfidence(float confidence) {
        this.confidence = confidence;
    }

    public float getClsIdx() {
        return cls_idx;
    }

    public void setClsIdx(float idx) {
        this.cls_idx = idx;
    }

    public String getClsLabel() {
        return cls_label;
    }

    public void setClsLabel(String label) {
        this.cls_label = label;
    }

    public float getClsConfidence() {
        return cls_confidence;
    }

    public void setClsConfidence(float confidence) {
        this.cls_confidence = confidence;
    }
}
