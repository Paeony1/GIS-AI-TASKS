# GIS-AI 任务

遥感影像处理的两个练习任务：植被指数计算（NDVI）与地物分类（随机森林）。
每个任务都遵循"规格先行 → AI 写代码 → 逐步验证"的流程，重点不在"能跑出结果"，而在**能证明结果是对的，并理解每一行代码**。

环境：Python + `rasterio` + `numpy` + `scikit-learn`。

---

## 任务一：NDVI 植被分析（`任务1-NDVI/`）

读取多波段 GeoTIFF（波段1=Red、波段2=NIR），逐像素计算 NDVI = (NIR − Red) / (NIR + Red)，输出保留地理参考的 float32 GeoTIFF。

| 文件 | 说明 |
|------|------|
| `spec_task1.md` | 规格书（输入/输出/约束/验收/非目标） |
| `self_check_report.md` | **自检报告**：6 关验证逐条对照实测结果 |
| `inspect_metadata.py` | 读取影像元数据（确认波段顺序、定标系数、nodata，不靠假设） |
| `ndvi.py` | NDVI 主计算脚本 |
| `validate_ndvi.py` | 随机抽 3 像素，手算 vs 代码逐像素对拍 |
| `input.tif` / `ndvi.tif` | 输入影像 / NDVI 输出 |
| `overlay.png` / `ndvi_overlay_50pct.png` | NDVI 叠回原图的目视检查 |

运行：
```bash
cd 任务1-NDVI
python inspect_metadata.py   # 看元数据
python ndvi.py               # 算 NDVI -> ndvi.tif
python validate_ndvi.py      # 手算对拍验证
```

关键结论：手算与代码差值约 1e-8；NDVI 全部落在 [−1, 1] 内；nodata 与除零像素安全处理为 NaN；CRS/transform 完整保留。详见 `self_check_report.md`。

---

## 任务二：植被/水体分类（`任务2分类/`）

用随机森林对 4 波段影像（Red/NIR/Green/SWIR）做三类地物分类（0=其他/1=植被/2=水体）。
核心不是"分得准"，而是**当你无法肉眼判断对错时，怎么相信或不相信一个模型**。

| 文件 | 说明 |
|------|------|
| `spec_task2.md` | 规格书 |
| `evaluation_report.md` | **评估报告**：基线/混淆矩阵/各类 P-R/两种切分对比/泄漏受控实验 |
| `count_labels.py` | 统计各类占比（含多数类基线） |
| `classify.py` | 随机森林分类 + 两种切分对比 |
| `leakage_demo.py` | 数据泄漏成因的受控实验（两种噪声对照） |
| `scene.tif` / `labels.tif` / `prediction.tif` | 输入影像 / 真值标签 / 预测结果 |
| `result_summary.txt` | classify.py 输出的验收摘要 |

运行：
```bash
cd 任务2分类
python count_labels.py    # 各类占比 + 多数类基线
python classify.py        # 分类 + 两种切分对比 -> prediction.tif, result_summary.txt
python leakage_demo.py    # 数据泄漏成因受控实验
```

### 一句话结论：为什么一个准确率 99% 的模型可能毫无用处？

因为这个数字可能来自"测试集偷看了训练集的空间邻居"。遥感像素空间自相关极强，随机打散切分会让测试像素的近邻全留在训练集里，模型只是记住邻居而非学会泛化；只有用空间块切分把邻居隔开，才能看到真实的泛化能力。**所以光看 accuracy 无法判断模型好坏，必须先追问：这个数字是用哪种切分、在什么数据结构上得来的。**

### 本数据的特殊之处

本数据两种切分都得到 100%——**不是泄漏被掩盖，而是数据 per-pixel 完美可分、无信息可泄漏**（见 `leakage_demo.py` 可分性诊断）。为证明确实理解泄漏成因，`leakage_demo.py` 做了受控对照：i.i.d. 白噪声演示不出泄漏（差距为负），空间相关噪声能演示出泄漏（σ=0.30 时随机切分比空间块高 +0.0297）——说明**泄漏的根源是空间结构，而非噪声本身**。详见 `evaluation_report.md`。
