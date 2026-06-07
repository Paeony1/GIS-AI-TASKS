"""
NDVI 验证脚本
=================
随机选取 3 个非 nodata 像素，对比：
  - 原始 DN (Red, NIR)
  - 转换后的反射率 (Red_ref, NIR_ref)
  - 手工计算的 NDVI = (NIR_ref - Red_ref) / (NIR_ref + Red_ref)
  - 代码（ndvi.py）输出文件 ndvi.tif 中的 NDVI

如果两者一致，说明 ndvi.py 的计算正确。
"""

import numpy as np
import rasterio

INPUT_FILE = "input.tif"
NDVI_FILE = "ndvi.tif"

SCALE = 0.0000275
OFFSET = -0.2
DN_NODATA = 0

N_SAMPLES = 3
SEED = 42  # 固定随机种子，结果可复现


def main():
    # ------------------------------------------------------------
    # 1. 读取原始波段 + 代码输出的 NDVI
    # ------------------------------------------------------------
    with rasterio.open(INPUT_FILE) as src:
        red_dn = src.read(1)
        nir_dn = src.read(2)

    with rasterio.open(NDVI_FILE) as src:
        ndvi_code = src.read(1)

    # ------------------------------------------------------------
    # 2. 找出所有非 nodata 像素（两波段都 != 0）
    # ------------------------------------------------------------
    valid_mask = (red_dn != DN_NODATA) & (nir_dn != DN_NODATA)
    valid_rows, valid_cols = np.where(valid_mask)

    if len(valid_rows) == 0:
        print("没有找到有效像素。")
        return

    # ------------------------------------------------------------
    # 3. 随机选取 3 个有效像素
    # ------------------------------------------------------------
    rng = np.random.default_rng(SEED)
    n = min(N_SAMPLES, len(valid_rows))
    idx = rng.choice(len(valid_rows), size=n, replace=False)

    print(f"随机选取 {n} 个非 nodata 像素进行验证 (seed={SEED}):\n")

    # ------------------------------------------------------------
    # 4. 逐像素对比手工计算与代码计算
    # ------------------------------------------------------------
    for k, i in enumerate(idx, start=1):
        r, c = int(valid_rows[i]), int(valid_cols[i])

        # 原始 DN
        dn_red = int(red_dn[r, c])
        dn_nir = int(nir_dn[r, c])

        # 转换为反射率
        ref_red = dn_red * SCALE + OFFSET
        ref_nir = dn_nir * SCALE + OFFSET

        # 手工计算 NDVI
        denom = ref_nir + ref_red
        ndvi_manual = (ref_nir - ref_red) / denom if denom != 0 else np.nan

        # 代码计算 NDVI（从输出文件读取）
        ndvi_from_code = float(ndvi_code[r, c])

        # 两者之差
        diff = abs(ndvi_manual - ndvi_from_code)

        print(f"像素 {k}  (row={r}, col={c})")
        print(f"  原始 DN        : Red={dn_red}, NIR={dn_nir}")
        print(f"  反射率         : Red={ref_red:.6f}, NIR={ref_nir:.6f}")
        print(f"  手工 NDVI      : {ndvi_manual:.6f}")
        print(f"  代码 NDVI      : {ndvi_from_code:.6f}")
        print(f"  差值 |手工-代码|: {diff:.2e}")
        print()


if __name__ == "__main__":
    main()
