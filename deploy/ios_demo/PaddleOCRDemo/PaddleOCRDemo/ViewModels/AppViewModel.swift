import Foundation

enum LoadingState: Equatable {
    case idle
    case loading
    case ready(detResult: String, recResult: String)
    case failed(message: String)

    static func == (lhs: LoadingState, rhs: LoadingState) -> Bool {
        switch (lhs, rhs) {
        case (.idle, .idle), (.loading, .loading): return true
        case (.ready(let a1, let a2), .ready(let b1, let b2)): return a1 == b1 && a2 == b2
        case (.failed(let a), .failed(let b)): return a == b
        default: return false
        }
    }
}

@MainActor
class AppViewModel: ObservableObject {
    @Published var loadingState: LoadingState = .idle

    private let sessionManager = ORTSessionManager()

    func loadModels() {
        loadingState = .loading
        Task {
            do {
                try await sessionManager.loadModels()

                // Validate both models with dummy inference
                let detResult = try await sessionManager.validateDetModel()
                let recResult = try await sessionManager.validateRecModel()

                // Format results for display
                let detInfo = "Detection: \(detResult.modelName) -- outputs: \(detResult.outputNames.joined(separator: ", "))"
                let recInfo = "Recognition: \(recResult.modelName) -- outputs: \(recResult.outputNames.joined(separator: ", "))"

                loadingState = .ready(detResult: detInfo, recResult: recInfo)
            } catch {
                loadingState = .failed(message: error.localizedDescription)
            }
        }
    }
}
