// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import XCTest
@testable import PaddleOCRDemo

/// Verifies ``OCRParameterResolver``
final class OCRParameterResolverTests: XCTestCase {

    func testDetResizeMergePrefersAppDefaultsOverModelWhenUINil() throws {
        let dir = FileManager.default.temporaryDirectory
        let url = dir.appendingPathComponent("merge_test_det_\(UUID().uuidString).yaml")
        let yaml = """
        Global:
          model_name: PP-OCRv5_mobile_det
        PreProcess:
          transform_ops:
          - DetResizeForTest:
              resize_long: 960
        PostProcess:
          name: DBPostProcess
        """
        try yaml.write(to: url, atomically: true, encoding: .utf8)
        defer { try? FileManager.default.removeItem(at: url) }

        let det = try InferenceConfig.load(from: url.path)
        let m = det.detResizeFromModel
        XCTAssertNotNil(m)
        XCTAssertEqual(m?.limitSideLen, 960)

        let eff = OCRParameterResolver.effective(modelConfig: det, runtime: .noOverrides)
        let app = OCRParameterResolver.textDetLimitSideLenAppDefault
        XCTAssertEqual(eff.detResize.limitSideLen, app)
        XCTAssertEqual(eff.mergedLimitSideLen, app)
    }

    func testEffectiveDetPostprocessMergesThresholdsOnly() throws {
        let dir = FileManager.default.temporaryDirectory
        let url = dir.appendingPathComponent("merge_test_post_\(UUID().uuidString).yaml")
        let yaml = """
        Global:
          model_name: PP-OCRv5_mobile_det
        PreProcess:
          transform_ops:
          - DetResizeForTest:
              resize_long: 960
        PostProcess:
          name: DBPostProcess
          thresh: 0.99
          box_thresh: 0.99
          unclip_ratio: 9.9
          max_candidates: 7
        """
        try yaml.write(to: url, atomically: true, encoding: .utf8)
        defer { try? FileManager.default.removeItem(at: url) }

        let det = try InferenceConfig.load(from: url.path)
        let post = OCRParameterResolver.effective(modelConfig: det, runtime: .noOverrides)
        XCTAssertEqual(post.detThresh, 0.3, accuracy: 0.0001)
        XCTAssertEqual(post.detBoxThresh, 0.6, accuracy: 0.0001)
        XCTAssertEqual(post.detUnclipRatio, 1.5, accuracy: 0.0001)
        XCTAssertEqual(det.postProcess.maxCandidates, 7)
    }

    func testDecodeImageImgModeParsedAsRGB() throws {
        let dir = FileManager.default.temporaryDirectory
        let url = dir.appendingPathComponent("decode_rgb_\(UUID().uuidString).yaml")
        let yaml = """
        Global:
          model_name: PP-OCRv5_mobile_det
        PreProcess:
          transform_ops:
          - DecodeImage:
              channel_first: false
              img_mode: RGB
          - DetResizeForTest:
              resize_long: 960
        PostProcess:
          name: DBPostProcess
        """
        try yaml.write(to: url, atomically: true, encoding: .utf8)
        defer { try? FileManager.default.removeItem(at: url) }

        let cfg = try InferenceConfig.load(from: url.path)
        XCTAssertEqual(cfg.decodeImageChannelOrder, .rgb)
    }

    func testDecodeImageMissingDefaultsToBGR() throws {
        let dir = FileManager.default.temporaryDirectory
        let url = dir.appendingPathComponent("decode_default_bgr_\(UUID().uuidString).yaml")
        let yaml = """
        Global:
          model_name: PP-OCRv5_mobile_det
        PreProcess:
          transform_ops:
          - DetResizeForTest:
              resize_long: 960
        PostProcess:
          name: DBPostProcess
        """
        try yaml.write(to: url, atomically: true, encoding: .utf8)
        defer { try? FileManager.default.removeItem(at: url) }

        let cfg = try InferenceConfig.load(from: url.path)
        XCTAssertEqual(cfg.decodeImageChannelOrder, .bgr)
    }
}
