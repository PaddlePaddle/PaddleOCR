/*
 * Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

declare module "@techstark/opencv-js" {
  export interface Mat {
    rows: number;
    cols: number;
    data: Uint8Array;
    data32S: Int32Array;
    channels(): number;
    copyTo(dst: Mat): void;
    clone(): Mat;
    delete(): void;
    roi(rect: Rect): Mat;
  }

  export interface MatVector {
    push_back(mat: Mat): void;
    size(): number;
    get(index: number): Mat;
    delete(): void;
  }

  export interface Size {
    width: number;
    height: number;
  }

  export interface Rect {
    x: number;
    y: number;
    width: number;
    height: number;
  }

  export interface Scalar {
    [index: number]: number;
  }

  export interface RotatedRect {
    center: { x: number; y: number };
    size: Size;
    angle: number;
  }

  export interface OpenCv {
    Mat: {
      new (): Mat;
      zeros(rows: number, cols: number, type: number): Mat;
    };
    MatVector: { new (): MatVector };
    Size: { new (width: number, height: number): Size };
    Rect: { new (x: number, y: number, width: number, height: number): Rect };
    Scalar: { new (v0?: number, v1?: number, v2?: number, v3?: number): Scalar };
    RotatedRect: {
      points(rect: RotatedRect): Array<{ x: number; y: number }>;
    };

    matFromArray(rows: number, cols: number, type: number, data: ArrayLike<number>): Mat;
    minAreaRect(points: Mat): RotatedRect;
    imread(canvas: HTMLCanvasElement): Mat;
    resize(src: Mat, dst: Mat, dsize: Size, fx?: number, fy?: number, interpolation?: number): void;
    cvtColor(src: Mat, dst: Mat, code: number): void;
    findContours(
      image: Mat,
      contours: MatVector,
      hierarchy: Mat,
      mode: number,
      method: number
    ): void;
    fillPoly(img: Mat, pts: MatVector, color: Scalar): void;
    mean(src: Mat, mask?: Mat): number[];
    getPerspectiveTransform(src: Mat, dst: Mat): Mat;
    warpPerspective(
      src: Mat,
      dst: Mat,
      M: Mat,
      dsize: Size,
      flags?: number,
      borderMode?: number,
      borderValue?: Scalar
    ): void;
    rotate(src: Mat, dst: Mat, rotateCode: number): void;

    // Mat type constants
    CV_32FC1: number;
    CV_32FC2: number;
    CV_32SC2: number;
    CV_8UC1: number;
    CV_8UC4: number;

    // Color conversion codes
    COLOR_RGBA2BGR: number;
    COLOR_GRAY2BGR: number;

    // Interpolation flags
    INTER_LINEAR: number;
    INTER_CUBIC: number;

    // Border types
    BORDER_REPLICATE: number;

    // Contour retrieval modes
    RETR_LIST: number;

    // Contour approximation methods
    CHAIN_APPROX_SIMPLE: number;

    // Rotation codes
    ROTATE_90_COUNTERCLOCKWISE: number;
  }

  export type CvModule = OpenCv & { onRuntimeInitialized?: () => void };

  const cv: CvModule | Promise<CvModule>;
  export default cv;
}
