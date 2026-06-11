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

/// Displays the source image with bounding box polygon overlays drawn on a Canvas.
///
/// Polygon coordinates from `OCRResult.polygon` are in original image pixel space.
/// This view computes the scale factor and offset to map them to the displayed
/// image area, accounting for `.aspectFit` rendering within the available frame.
struct ResultImageView: View {
    let image: UIImage
    let results: [OCRResult]

    var body: some View {
        GeometryReader { geo in
            let imageSize = image.size
            let imageAspect = imageSize.width / imageSize.height
            let viewAspect = geo.size.width / geo.size.height

            // Compute the actual rendered image rect within the GeometryReader frame
            let fitSize: CGSize = imageAspect > viewAspect
                ? CGSize(width: geo.size.width, height: geo.size.width / imageAspect)
                : CGSize(width: geo.size.height * imageAspect, height: geo.size.height)
            let offset = CGPoint(
                x: (geo.size.width - fitSize.width) / 2,
                y: (geo.size.height - fitSize.height) / 2
            )
            let scale = fitSize.width / imageSize.width

            ZStack {
                Image(uiImage: image)
                    .resizable()
                    .aspectRatio(contentMode: .fit)

                Canvas { context, size in
                    for result in results {
                        let points = result.polygon.compactMap { coords -> CGPoint? in
                            guard coords.count >= 2 else { return nil }
                            return CGPoint(
                                x: CGFloat(coords[0]) * scale + offset.x,
                                y: CGFloat(coords[1]) * scale + offset.y
                            )
                        }
                        guard points.count >= 4 else { continue }

                        var path = Path()
                        path.move(to: points[0])
                        for pt in points.dropFirst() {
                            path.addLine(to: pt)
                        }
                        path.closeSubpath()

                        let strokeColor = Color.accentColor.opacity(0.9)
                        let fillColor = Color.accentColor.opacity(0.12)
                        context.stroke(path, with: .color(strokeColor), lineWidth: 2)
                        context.fill(path, with: .color(fillColor))
                    }
                }
            }
            .frame(width: geo.size.width, height: geo.size.height)
        }
        .aspectRatio(image.size.width / image.size.height, contentMode: .fit)
        .frame(maxHeight: 400)
        .accessibilityElement(children: .combine)
        .accessibilityLabel("Image preview with \(results.count) detected text regions")
    }
}
