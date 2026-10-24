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

import SwiftUI

private enum CoreMLComputeMode: String, CaseIterable, Identifiable {
    case automatic
    case cpuOnly
    case cpuAndGPU
    case devicesWithANE

    var id: String { rawValue }

    var segmentLabel: String {
        switch self {
        case .automatic: return "Auto"
        case .cpuOnly: return "CPU"
        case .cpuAndGPU: return "CPU+GPU"
        case .devicesWithANE: return "ANE"
        }
    }

    static func from(flags: Set<String>) -> CoreMLComputeMode {
        let cpuOnly = ORTCoreMLProviderOption.cpuOnly.rawValue
        let cpuAndGPU = ORTCoreMLProviderOption.cpuAndGPU.rawValue
        let aneOnly = ORTCoreMLProviderOption.aneOnly.rawValue
        if flags.contains(cpuOnly) { return .cpuOnly }
        if flags.contains(cpuAndGPU) { return .cpuAndGPU }
        if flags.contains(aneOnly) { return .devicesWithANE }
        return .automatic
    }

    func merge(into flags: inout Set<String>) {
        let cpuOnly = ORTCoreMLProviderOption.cpuOnly.rawValue
        let cpuAndGPU = ORTCoreMLProviderOption.cpuAndGPU.rawValue
        let aneOnly = ORTCoreMLProviderOption.aneOnly.rawValue
        flags.remove(cpuOnly)
        flags.remove(cpuAndGPU)
        flags.remove(aneOnly)
        switch self {
        case .automatic: break
        case .cpuOnly: flags.insert(cpuOnly)
        case .cpuAndGPU: flags.insert(cpuAndGPU)
        case .devicesWithANE: flags.insert(aneOnly)
        }
    }
}

/// ONNX Runtime: preferred execution provider (EP) preset plus session tuning. Card content only.
struct OnnxRuntimeSettingsPanel: View {
    private static let coreMLExtraToggleOptions: [ORTCoreMLProviderOption] =
        [.enableOnSubgraphs, .createMLProgram, .staticInputShapes]

    @Binding var primaryExecutionProvider: ORTPrimaryExecutionProvider
    @Binding var tuning: ORTSessionTuningOptions

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("All inference uses ONNX Runtime. Choose which EP is preferred first for this demo preset; unsupported operators still run on the CPU EP.")
                .font(.caption)
                .foregroundStyle(.secondary)

            Picker("Execution provider", selection: $primaryExecutionProvider) {
                ForEach(ORTPrimaryExecutionProvider.allCases) { provider in
                    Text(provider.displayTitle).tag(provider)
                }
            }
            .pickerStyle(.segmented)

            Divider()
                .padding(.vertical, 2)

            Text("Session options")
                .font(.subheadline.weight(.semibold))

            labeledStepper(
                title: "Intra-op threads",
                caption: "Session-wide CPU parallelism. Use 0 to leave ONNX Runtime’s default (no explicit limit).",
                value: intraOpThreadsBinding,
                range: 0...8
            )

            if primaryExecutionProvider == .xnnpack {
                labeledStepper(
                    title: "XNNPACK intra-op threads",
                    caption: "0 leaves the XNNPACK EP default.",
                    value: xnnpackThreadsBinding,
                    range: 0...8
                )
            }

            if primaryExecutionProvider == .coreML {
                VStack(alignment: .leading, spacing: 10) {
                    Text("Core ML EP options")
                        .font(.subheadline.weight(.semibold))
                        .padding(.top, 4)

                    VStack(alignment: .leading, spacing: 6) {
                        Text("Core ML placement")
                            .font(.body)
                        Picker("Core ML placement", selection: coreMLComputeModeBinding) {
                            ForEach(CoreMLComputeMode.allCases) { mode in
                                Text(mode.segmentLabel).tag(mode)
                            }
                        }
                        .pickerStyle(.segmented)
                        Text(coreMLComputeModeCaption)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }

                    ForEach(Self.coreMLExtraToggleOptions) { option in
                        Toggle(isOn: coreMLFlagBinding(option)) {
                            VStack(alignment: .leading, spacing: 2) {
                                Text(option.displayTitle)
                                Text(option.detailCaption)
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                            }
                        }
                    }
                }
            }

            HStack {
                Spacer()
                Button("Reset session options") {
                    tuning = .default
                }
                .font(.subheadline.weight(.medium))
            }
            .padding(.top, 4)
        }
    }

    private var coreMLComputeModeCaption: String {
        switch CoreMLComputeMode.from(flags: tuning.coreMLFlags) {
        case .automatic:
            return "ORT defaults — no cpuOnly/cpuAndGPU/device ANE restriction."
        case .cpuOnly:
            return "useCPUOnly — Core ML constrained to CPU."
        case .cpuAndGPU:
            return "useCPUAndGPU — CPU and GPU paths; excludes ANE in Core ML."
        case .devicesWithANE:
            return "Only enable Core ML EP on devices that have a Neural Engine."
        }
    }

    private var intraOpThreadsBinding: Binding<Int> {
        Binding(
            get: { tuning.intraOpThreads },
            set: { newValue in
                var next = tuning
                next.intraOpThreads = newValue
                tuning = next
            }
        )
    }

    private var xnnpackThreadsBinding: Binding<Int> {
        Binding(
            get: { tuning.xnnpackThreads },
            set: { newValue in
                var next = tuning
                next.xnnpackThreads = newValue
                tuning = next
            }
        )
    }

    private var coreMLComputeModeBinding: Binding<CoreMLComputeMode> {
        Binding(
            get: { CoreMLComputeMode.from(flags: tuning.coreMLFlags) },
            set: { mode in
                var next = tuning
                mode.merge(into: &next.coreMLFlags)
                tuning = next
            }
        )
    }

    private func coreMLFlagBinding(_ option: ORTCoreMLProviderOption) -> Binding<Bool> {
        let key = option.rawValue
        return Binding(
            get: { tuning.coreMLFlags.contains(key) },
            set: { on in
                var next = tuning
                if on {
                    next.coreMLFlags.insert(key)
                } else {
                    next.coreMLFlags.remove(key)
                }
                tuning = next
            }
        )
    }

    private func labeledStepper(title: String, caption: String, value: Binding<Int>, range: ClosedRange<Int>) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(alignment: .firstTextBaseline) {
                Text(title)
                Spacer()
                Stepper(value: value, in: range) {
                    Text("\(value.wrappedValue)")
                        .font(.body.monospacedDigit())
                        .frame(minWidth: 24, alignment: .trailing)
                }
            }
            Text(caption)
                .font(.caption)
                .foregroundStyle(.secondary)
        }
    }
}
