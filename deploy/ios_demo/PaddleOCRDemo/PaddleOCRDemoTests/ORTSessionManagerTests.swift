import XCTest
@testable import PaddleOCRDemo

final class ORTSessionManagerTests: XCTestCase {

    // MARK: - Model Loading Tests (INFER-02, INFER-03, INFER-04)

    /// Validates that both models load successfully with CoreML EP configured.
    /// Covers: INFER-02 (CoreML EP), INFER-03 (det loads), INFER-04 (rec loads)
    func testLoadModels() async throws {
        let manager = ORTSessionManager()
        // Should not throw -- both models load into ORTSessions
        try await manager.loadModels()
    }

    // MARK: - Detection Inference Tests (INFER-03)

    /// Validates detection model produces output from dummy input.
    /// Covers: INFER-03 (det runs inference on device)
    func testDetectionDummyInference() async throws {
        let manager = ORTSessionManager()
        try await manager.loadModels()

        let result = try await manager.validateDetModel()

        // Model name should be "det"
        XCTAssertEqual(result.modelName, "det")

        // Must have at least one input and one output
        XCTAssertFalse(result.inputNames.isEmpty, "Detection model should have input names")
        XCTAssertFalse(result.outputNames.isEmpty, "Detection model should have output names")

        // Output shapes must be non-empty (model produced tensors)
        XCTAssertFalse(result.outputShapes.isEmpty, "Detection model should produce output tensors")

        // No NaN values in output
        XCTAssertFalse(result.hasNaN, "Detection output should not contain NaN values")
    }

    // MARK: - Recognition Inference Tests (INFER-04)

    /// Validates recognition model produces output from dummy input.
    /// Covers: INFER-04 (rec runs inference on device)
    func testRecognitionDummyInference() async throws {
        let manager = ORTSessionManager()
        try await manager.loadModels()

        let result = try await manager.validateRecModel()

        // Model name should be "rec"
        XCTAssertEqual(result.modelName, "rec")

        // Must have at least one input and one output
        XCTAssertFalse(result.inputNames.isEmpty, "Recognition model should have input names")
        XCTAssertFalse(result.outputNames.isEmpty, "Recognition model should have output names")

        // Output shapes must be non-empty (model produced tensors)
        XCTAssertFalse(result.outputShapes.isEmpty, "Recognition model should produce output tensors")

        // No NaN values in output
        XCTAssertFalse(result.hasNaN, "Recognition output should not contain NaN values")
    }

    // MARK: - ModelConfig Tests

    /// Validates that ModelConfig resolves bundle paths correctly.
    /// If models are bundled, paths should resolve; if not, should throw ModelConfigError.
    func testModelConfigResolution() throws {
        // These should either resolve to valid paths or throw ModelConfigError.modelNotFound
        // (depending on whether models are in the test bundle).
        // We test the throwing path is the correct error type.
        do {
            let detConfig = try ModelConfig.detection()
            XCTAssertFalse(detConfig.modelPath.isEmpty, "Det model path should not be empty")
            XCTAssertEqual(detConfig.name, "PP-OCRv5 Mobile Det")
        } catch let error as ModelConfigError {
            // Acceptable if model file is not in the test host bundle
            switch error {
            case .modelNotFound(let path):
                XCTAssertTrue(path.contains("det"), "Error should reference det model path")
            }
        }

        do {
            let recConfig = try ModelConfig.recognition()
            XCTAssertFalse(recConfig.modelPath.isEmpty, "Rec model path should not be empty")
            XCTAssertEqual(recConfig.name, "PP-OCRv5 Mobile Rec")
        } catch let error as ModelConfigError {
            switch error {
            case .modelNotFound(let path):
                XCTAssertTrue(path.contains("rec"), "Error should reference rec model path")
            }
        }
    }
}
