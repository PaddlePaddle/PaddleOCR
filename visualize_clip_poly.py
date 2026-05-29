"""
可视化 clip_poly_to_rect 对不同多边形裁剪后的简化效果。
展示：原始多边形、裁剪后的交集多边形（可能>4点）、approxPolyDP简化结果（<=4点）。

Usage: python visualize_clip_poly.py
Output: clip_poly_visualization.png
"""

import numpy as np
import cv2
from shapely.geometry import Polygon, box as shapely_box
from shapely import intersection
from ppocr.data.imaug.random_crop_data import clip_poly_to_rect


def get_clipped_coords(poly, x, y, w, h):
    """获取裁剪后的原始交集坐标（未简化）"""
    poly_shape = Polygon(poly)
    crop_rect = shapely_box(x, y, x + w, y + h)
    clipped = intersection(poly_shape, crop_rect)
    if clipped.is_empty:
        return None
    if clipped.geom_type == "Polygon":
        return np.array(clipped.exterior.coords[:-1])
    return None


def draw_case(img, poly, x, y, w, h, title, offset_x, offset_y):
    """在图像上绘制一个案例：原始多边形、裁剪区域、交集、简化结果"""
    ox, oy = offset_x, offset_y
    scale = 2  # 放大便于观察

    def pt(p):
        return (int(p[0] * scale + ox), int(p[1] * scale + oy))

    # 绘制裁剪区域
    cv2.rectangle(img, pt([x, y]), pt([x + w, y + h]), (200, 200, 200), 1)

    # 绘制原始多边形
    pts_orig = np.array([pt(p) for p in poly])
    cv2.polylines(img, [pts_orig], True, (150, 150, 150), 1, cv2.LINE_AA)

    # 获取裁剪后交集
    clipped = get_clipped_coords(poly, x, y, w, h)
    if clipped is not None:
        pts_clip = np.array([pt(p) for p in clipped])
        cv2.polylines(img, [pts_clip], True, (0, 165, 255), 2, cv2.LINE_AA)
        for p in clipped:
            cv2.circle(img, pt(p), 4, (0, 165, 255), -1)

    # 获取简化结果
    result = clip_poly_to_rect(poly.astype(np.float32), x, y, w, h)
    if result is not None:
        pts_res = np.array([pt(p) for p in result])
        cv2.polylines(img, [pts_res], True, (0, 200, 0), 2, cv2.LINE_AA)
        for p in result:
            cv2.circle(img, pt(p), 5, (0, 200, 0), -1)

    # 标题和信息
    cv2.putText(img, title, (ox, oy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    n_clip = len(clipped) if clipped is not None else 0
    n_res = len(result) if result is not None else 0
    cv2.putText(img, f"clipped:{n_clip}pts -> simplified:{n_res}pts",
                (ox, oy + 220), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)


# 构造测试用例
cases = []
crop = (10, 10, 80, 80)

# Case 1: 三角形裁剪结果（3点，不需要简化）
cases.append(("3pts: triangle clip", np.array([[50, 0], [100, 80], [0, 80]]), crop))

# Case 2: 四边形裁剪结果（4点，直接通过）
cases.append(("4pts: quad clip", np.array([[20, 5], [85, 15], [80, 85], [15, 90]]), crop))

# Case 3: 五边形裁剪结果
cases.append(("5pts: pentagon clip", np.array([[50, 0], [105, 40], [85, 105], [15, 105], [-5, 40]]), crop))

# Case 4: 六边形裁剪结果
angles6 = np.linspace(0, 2 * np.pi, 7)[:-1]
hex_poly = np.stack([50 + 60 * np.cos(angles6), 50 + 60 * np.sin(angles6)], axis=1)
cases.append(("6pts: hexagon clip", hex_poly, crop))

# Case 5: 八边形裁剪结果
angles8 = np.linspace(0, 2 * np.pi, 9)[:-1]
oct_poly = np.stack([50 + 70 * np.cos(angles8), 50 + 70 * np.sin(angles8)], axis=1)
cases.append(("8pts: octagon clip", oct_poly, crop))

# Case 6: 星形裁剪结果（复杂形状）
angles10 = np.linspace(0, 2 * np.pi, 11)[:-1]
radii = np.array([65, 30] * 5)
star_poly = np.stack([50 + radii * np.cos(angles10), 50 + radii * np.sin(angles10)], axis=1)
cases.append(("star shape clip", star_poly, crop))

# 绘制
cols, rows = 3, 2
cell_w, cell_h = 250, 260
img = np.zeros((rows * cell_h + 40, cols * cell_w, 3), dtype=np.uint8)

# 图例
cv2.putText(img, "Gray=original  Orange=clipped intersection  Green=simplified(<=4pts)",
            (10, rows * cell_h + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

for i, (title, poly, (x, y, w, h)) in enumerate(cases):
    r, c = i // cols, i % cols
    draw_case(img, poly.astype(np.float64), x, y, w, h, title,
              c * cell_w + 20, r * cell_h + 30)

out_path = "clip_poly_visualization.png"
cv2.imwrite(out_path, img)
print(f"Saved to {out_path}")
