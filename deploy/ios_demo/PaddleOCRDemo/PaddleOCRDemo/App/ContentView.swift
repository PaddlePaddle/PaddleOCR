import SwiftUI

struct ContentView: View {
    @StateObject private var viewModel = AppViewModel()

    var body: some View {
        VStack(spacing: 8) {
            switch viewModel.loadingState {
            case .idle:
                idleView
            case .loading:
                loadingView
            case .ready(let detResult, let recResult):
                readyView(detResult: detResult, recResult: recResult)
            case .failed(let message):
                failedView(message: message)
            }
        }
        .padding(.horizontal, 16)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Color(.systemBackground))
        .task {
            viewModel.loadModels()
        }
    }

    // MARK: - State Views

    private var idleView: some View {
        VStack(spacing: 4) {
            Text("PaddleOCR Demo")
                .font(.title3)
                .fontWeight(.semibold)
            Text("PP-OCRv5 Inference Engine")
                .font(.caption)
                .foregroundColor(Color(.secondaryLabel))
            Text("Initializing...")
                .font(.body)
                .foregroundColor(Color(.secondaryLabel))
                .padding(.top, 8)
        }
    }

    private var loadingView: some View {
        VStack(spacing: 8) {
            ProgressView()
                .scaleEffect(1.5)
                .padding(.bottom, 4)
            Text("Loading Models")
                .font(.title3)
                .fontWeight(.semibold)
            Text("Preparing detection and recognition models...")
                .font(.body)
                .foregroundColor(Color(.secondaryLabel))
                .multilineTextAlignment(.center)
        }
    }

    private func readyView(detResult: String, recResult: String) -> some View {
        VStack(spacing: 8) {
            Image(systemName: "checkmark.circle.fill")
                .font(.system(size: 48))
                .foregroundColor(.green)
            Text("Models Ready")
                .font(.title3)
                .fontWeight(.semibold)
            VStack(spacing: 4) {
                Text("Detection: PP-OCRv5 Mobile")
                    .font(.body)
                Text("Recognition: PP-OCRv5 Mobile")
                    .font(.body)
            }
            .foregroundColor(Color(.secondaryLabel))
            Text("Both models loaded successfully.")
                .font(.caption)
                .foregroundColor(Color(.secondaryLabel))
                .padding(.top, 4)
        }
    }

    private func failedView(message: String) -> some View {
        VStack(spacing: 8) {
            Image(systemName: "exclamationmark.triangle.fill")
                .font(.system(size: 48))
                .foregroundColor(.red)
            Text("Model Loading Failed")
                .font(.title3)
                .fontWeight(.semibold)
            Text(message)
                .font(.body)
                .foregroundColor(Color(.secondaryLabel))
                .multilineTextAlignment(.center)
            Button(action: {
                viewModel.loadModels()
            }) {
                Label("Retry", systemImage: "arrow.clockwise")
            }
            .buttonStyle(.borderedProminent)
            .padding(.top, 8)
        }
    }
}
