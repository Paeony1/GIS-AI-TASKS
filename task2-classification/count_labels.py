"""统计 labels.tif 中三个类别 (0, 1, 2) 的像素个数和占比，忽略 nodata。只打印结果。"""

import numpy as np
import rasterio

FILENAME = "labels.tif"
CLASSES = [0, 1, 2]


def main():
    with rasterio.open(FILENAME) as src:
        data = src.read(1)
        nodata = src.nodata

    # 构建有效像素掩膜：若定义了 nodata，则排除 nodata 像素
    if nodata is not None:
        valid_mask = data != nodata
    else:
        valid_mask = np.ones(data.shape, dtype=bool)

    valid = data[valid_mask]
    total_valid = valid.size

    print(f"文件: {FILENAME}")
    print(f"nodata 值: {nodata}")
    print(f"有效像素总数: {total_valid}")
    print("-" * 50)

    if total_valid == 0:
        print("没有有效像素。")
        return

    # 逐类别统计像素个数与占比
    for cls in CLASSES:
        count = int(np.count_nonzero(valid == cls))
        pct = count / total_valid * 100
        print(f"类别 {cls}: {count} 像素  ({pct:.2f}%)")

    # 提示：是否存在不在 [0,1,2] 之内的其他有效值
    other = int(np.count_nonzero(~np.isin(valid, CLASSES)))
    if other > 0:
        print("-" * 50)
        print(f"注意: 有 {other} 个有效像素的值不在 {CLASSES} 中")


if __name__ == "__main__":
    main()
