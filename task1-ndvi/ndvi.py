"""
NDVI 计算脚本
=================
输入：input.tif
  - 波段1 = Red
  - 波段2 = NIR
  - dtype: uint16，有效值 1~65535，0 表示 nodata
  - DN -> 反射率：reflectance = DN * 0.0000275 - 0.2

输出：ndvi.tif (float32, NaN 作为 nodata)
"""

import numpy as np
import rasterio

INPUT_FILE = "input.tif"
OUTPUT_FILE = "ndvi.tif"

# 反射率转换系数（USGS Landsat Collection 2 风格）
SCALE = 0.0000275
OFFSET = -0.2

# DN 的 nodata 值
DN_NODATA = 0


def main():
    # ------------------------------------------------------------
    # 1. 读取两个波段，并保留元数据（坐标、投影、transform）
    # ------------------------------------------------------------
    with rasterio.open(INPUT_FILE) as src:
        red_dn = src.read(1)            # 波段1 = Red
        nir_dn = src.read(2)            # 波段2 = NIR
        profile = src.profile.copy()    # 复制原始元数据
        print(f"已读取 {INPUT_FILE}: 尺寸 {src.height} x {src.width}, "
              f"波段数 {src.count}, dtype {src.dtypes[0]}")

    # ------------------------------------------------------------
    # 2. 构建有效像素掩膜（DN != 0 才是有效像素）
    #    只要任一波段为 0（nodata），该像素就视为无效
    # ------------------------------------------------------------
    valid_mask = (red_dn != DN_NODATA) & (nir_dn != DN_NODATA)
    print(f"有效像素数量: {valid_mask.sum()} / {valid_mask.size}")

    # ------------------------------------------------------------
    # 3. DN 转换为反射率（只对有效像素计算）
    #    无效像素先填为 NaN，避免参与后续计算
    # ------------------------------------------------------------
    red_ref = np.full(red_dn.shape, np.nan, dtype=np.float32)
    nir_ref = np.full(nir_dn.shape, np.nan, dtype=np.float32)

    # 用 float32 做转换，仅在有效像素位置赋值
    red_ref[valid_mask] = red_dn[valid_mask].astype(np.float32) * SCALE + OFFSET
    nir_ref[valid_mask] = nir_dn[valid_mask].astype(np.float32) * SCALE + OFFSET

    # ------------------------------------------------------------
    # 4. 计算 NDVI = (NIR - Red) / (NIR + Red)
    #    - 结果为 float32
    #    - 无效像素保持 NaN
    #    - 处理除零：NIR + Red == 0 时设为 NaN
    # ------------------------------------------------------------
    # 输出数组初始化为 NaN（这样无效区域、除零区域自然就是 NaN）
    ndvi = np.full(red_ref.shape, np.nan, dtype=np.float32)

    denom = nir_ref + red_ref                       # 分母
    numer = nir_ref - red_ref                       # 分子

    # 只在「有效像素」且「分母不为 0」的位置做除法
    compute_mask = valid_mask & (denom != 0)
    ndvi[compute_mask] = numer[compute_mask] / denom[compute_mask]

    # 显式统计因除零被排除的像素（有效但分母为 0）
    div_zero_count = int((valid_mask & (denom == 0)).sum())
    print(f"除零像素数量 (NIR+Red==0，已设为 NaN): {div_zero_count}")

    # ------------------------------------------------------------
    # 5. 写出 NDVI GeoTIFF，保留原坐标系/投影/transform
    #    nodata 用 NaN 表示
    # ------------------------------------------------------------
    profile.update(
        dtype="float32",
        count=1,            # 单波段输出
        nodata=np.nan,      # NaN 作为 nodata
    )

    with rasterio.open(OUTPUT_FILE, "w", **profile) as dst:
        dst.write(ndvi, 1)
        dst.set_band_description(1, "NDVI")
    print(f"已写出 {OUTPUT_FILE}")

    # ------------------------------------------------------------
    # 6. 打印 NDVI 统计（忽略 NaN）
    # ------------------------------------------------------------
    print("\nNDVI 统计 (忽略 NaN):")
    print(f"  最小值 (min):  {np.nanmin(ndvi):.6f}")
    print(f"  最大值 (max):  {np.nanmax(ndvi):.6f}")
    print(f"  均值   (mean): {np.nanmean(ndvi):.6f}")
    print(f"  标准差 (std):  {np.nanstd(ndvi):.6f}")


if __name__ == "__main__":
    main()
