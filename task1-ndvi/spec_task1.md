# 规格书 · 任务一：NDVI 计算工具

> 状态：已实现（与 `ndvi.py`、`validate_ndvi.py` 对齐）
> 最后更新：2026-06-07

---

## 1. 目标

读取单幅多波段 GeoTIFF（Red + NIR），逐像素计算 NDVI 并输出为保留地理参考的 float32 GeoTIFF。

---

## 2. 输入

| 项目 | 说明 |
|------|------|
| 文件 | `input.tif` |
| 格式 | GeoTIFF（rasterio 可读） |
| 波段 1 | Red（红光） |
| 波段 2 | NIR（近红外） |
| 数据类型 | `uint16` |
| 有效值范围 | `1 ~ 65535` |
| nodata | `0`（DN 为 0 视为无效像素） |
| 反射率转换公式 | `reflectance = DN * 0.0000275 + (-0.2)` |
| 缩放因子 (scale) | `0.0000275` |
| 偏移量 (offset) | `-0.2` |

说明：

- 转换公式采用 USGS Landsat Collection 2 Level-2 地表反射率的标准系数。
- 只要 Red 或 NIR 任一波段在某像素为 `0`，该像素即整体视为 nodata（含"半 nodata"像素）。

---

## 3. 输出

### 3.1 NDVI 栅格

| 项目 | 说明 |
|------|------|
| 文件 | `ndvi.tif` |
| 波段数 | 1（波段描述：`NDVI`） |
| 数据类型 | `float32` |
| nodata | `NaN` |
| 坐标系 / 投影 / transform | 继承自 `input.tif`（profile 复制） |
| 取值范围 | 理论 `[-1, 1]`；nodata 与除零像素为 `NaN` |

### 3.2 统计表（运行时打印到 stdout）

忽略 NaN 计算，包含：

| 指标 | 来源函数 |
|------|----------|
| 最小值 min | `np.nanmin` |
| 最大值 max | `np.nanmax` |
| 均值 mean | `np.nanmean` |
| 标准差 std | `np.nanstd` |

同时打印：有效像素数量、除零像素数量、读取尺寸/波段数/dtype。

---

## 4. 约束

- **依赖**：仅使用 `rasterio` 与 `numpy`，不引入额外 GIS 库。
- **nodata 处理**：以 `(red != 0) & (nir != 0)` 构建有效掩膜；无效像素在反射率与 NDVI 阶段均填 `NaN`，不参与任何计算。
- **除零处理**：当 `NIR + Red == 0` 时，对应 NDVI 设为 `NaN`（输出数组以 NaN 初始化，仅在"有效且分母≠0"处赋值），并统计除零像素数量。
- **计算精度**：反射率与 NDVI 全程以 `float32` 计算与存储。
- **地理参考**：输出 profile 由输入复制后仅修改 `dtype / count / nodata`，保证 CRS 与 transform 不变。

---

## 5. 验收标准

1. **手算验证**：`validate_ndvi.py` 随机抽取 3 个非 nodata 像素（固定 `seed=42`，可复现），手工计算 `(NIR_ref - Red_ref) / (NIR_ref + Red_ref)` 与 `ndvi.tif` 读取值逐像素对比，差值应 ≤ `1e-6`。
2. **范围检查**：所有非 NaN 的 NDVI 值落在 `[-1, 1]` 内。
3. **地理对齐**：`ndvi.tif` 的 CRS 与 transform 与 `input.tif` 完全一致；像素行列与原图逐一对应。
4. **nodata 一致性**：输入中任一波段为 0 的像素，在输出中为 `NaN`。
5. **植被区目视**：在 GIS（如 QGIS）中加载 `ndvi.tif`，植被覆盖区呈现较高 NDVI（偏正），水体/裸地/建筑偏低，符合常识。

---

## 6. 非目标

本工具**不**包含以下处理：

- 大气校正（输入已假定为可直接套用线性系数的 DN）。
- 云 / 云阴影掩膜。
- 多时相合成、镶嵌、重投影或重采样。
- 异常值剔除或 NDVI 平滑/滤波。
- 除 Red/NIR 外其他波段的指数（EVI、SAVI 等）。
- 数据下载或格式转换（仅处理本地 GeoTIFF）。

---

## 附录：相关文件

| 文件 | 用途 |
|------|------|
| `inspect_metadata.py` | 打印输入元数据（波段、dtype、nodata、CRS、transform、标签） |
| `ndvi.py` | NDVI 主计算脚本 |
| `validate_ndvi.py` | 随机像素手算 vs 代码结果验证 |
