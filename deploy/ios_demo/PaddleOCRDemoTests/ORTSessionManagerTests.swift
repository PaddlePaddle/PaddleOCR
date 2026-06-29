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

final class ORTSessionManagerTests: XCTestCase {

    func testLoadModels() async throws {
        let manager = ORTSessionManager()
        try await manager.loadModels(executionProvider: .coreML)
    }

    func testLoadModelsCPU() async throws {
        let manager = ORTSessionManager()
        try await manager.loadModels(executionProvider: .cpu)
    }

    func testModelConfigResolution() throws {
        do {
            let detConfig = try ModelConfig.detection()
            XCTAssertFalse(detConfig.modelPath.isEmpty, "Det model path should not be empty")
            let expected = try InferenceConfig.load(from: detConfig.configPath).modelName
            XCTAssertEqual(detConfig.name, expected, "Display name should equal Global.model_name in the det model config")
        } catch let error as ModelConfigError {
            switch error {
            case .modelNotFound(let path):
                XCTAssertTrue(path.contains("det"), "Error should reference det model path")
            }
        }

        do {
            let recConfig = try ModelConfig.recognition()
            XCTAssertFalse(recConfig.modelPath.isEmpty, "Rec model path should not be empty")
            let expected = try InferenceConfig.load(from: recConfig.configPath).modelName
            XCTAssertEqual(recConfig.name, expected, "Display name should equal Global.model_name in the rec model config")
        } catch let error as ModelConfigError {
            switch error {
            case .modelNotFound(let path):
                XCTAssertTrue(path.contains("rec"), "Error should reference rec model path")
            }
        }
    }
}
