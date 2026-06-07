"""
classify.py — 任务二：随机森林地物分类（两种切分对比）
================================================================
严格遵循 spec_task2.md：
  - 模型：RandomForestClassifier(n_estimators=100)
  - 特征：scene.tif 的原始 4 波段（Red, NIR, Green, SWIR），已为反射率，不缩放
  - 标签：labels.tif，0=其他 / 1=植被 / 2=水体
  - 核心：对比「随机像素切分（错误，会数据泄漏）」与「空间块切分（正确）」
  - 两种切分：相同随机种子、相同模型参数，唯一变量是切分方式
输出：
  - 控制台：每种切分的混淆矩阵 / 分类报告 / 总体准确率 / 基线准确率
  - prediction.tif：空间块切分测试集预测（非测试像素 = -1），保留空间参考
  - result_summary.txt：两种切分准确率对比 + 三条验收结论
"""

import numpy as np
import rasterio
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# ============================================================
# 配置
# ============================================================
SCENE_FILE = "scene.tif"
LABELS_FILE = "labels.tif"
PRED_FILE = "prediction.tif"
SUMMARY_FILE = "result_summary.txt"

SEED = 42                    # 两种切分共用，保证可复现且公平对比
N_ESTIMATORS = 100           # 随机森林树数（不调参）
TEST_SIZE = 0.30             # 测试集比例（70% 训练 / 30% 测试）
GRID = 10                    # 空间块切分：10×10 网格 = 100 个块
MAJORITY_BASELINE = 0.734    # 多数类（植被）占比，作为 accuracy 基线
WATER_RECALL_MIN = 0.5       # 验收标准②：水体 recall 门槛
NODATA_PRED = -1             # 预测栅格中非测试像素的填充值

CLASS_LABELS = [0, 1, 2]
CLASS_NAMES = ["其他(0)", "植被(1)", "水体(2)"]
WATER_NAME = "水体(2)"


def log(msg=""):
    print(msg)

# ============================================================
# 1. 读取数据
#    40000 像素很小，直接全部读入内存，无需分块。
# ============================================================
def load_data():
    # 读取 4 波段影像，shape = (4, H, W)
    with rasterio.open(SCENE_FILE) as src:
        bands = src.read()                 # (4, H, W)
        profile = src.profile.copy()       # 保留空间参考用于写出预测
        height, width = src.height, src.width
        n_bands = src.count

    # 读取标签，shape = (H, W)
    with rasterio.open(LABELS_FILE) as src:
        labels = src.read(1)

    log(f"影像: {SCENE_FILE}  波段数={n_bands}  尺寸={height}x{width}")
    log(f"标签: {LABELS_FILE}  尺寸={labels.shape}")

    # 将 (4, H, W) 重排为 (H*W, 4)：每行一个像素，4 列为 4 个波段特征
    # transpose 到 (H, W, 4) 再 reshape，保证像素顺序与标签一致（行优先）
    X = bands.transpose(1, 2, 0).reshape(-1, n_bands).astype(np.float32)
    y = labels.reshape(-1)

    log(f"特征矩阵 X: {X.shape}   标签向量 y: {y.shape}")
    return X, y, profile, height, width


# ============================================================
# 2. 两种切分方式
#    返回训练/测试的「像素索引」，便于后续重建预测栅格。
# ============================================================
def random_split(n_pixels):
    """
    随机像素切分（错误做法，作为反面对照）。
    --------------------------------------------------------
    直接在全部像素上随机抽 30% 作测试。问题在于：遥感像素存在强空间
    自相关，相邻像素几乎重复。随机打散后，测试像素的空间邻居大量留在
    训练集里，模型相当于「见过答案的邻居」，导致数据泄漏、精度虚高。
    """
    all_idx = np.arange(n_pixels)
    train_idx, test_idx = train_test_split(
        all_idx, test_size=TEST_SIZE, random_state=SEED
    )
    return train_idx, test_idx


def spatial_block_split(height, width):
    """
    空间块切分（正确做法）。
    --------------------------------------------------------
    将影像按 10×10 网格切成 100 个空间块（每块约 20×20 像素）。
    按「块」做 70/30 划分：同一块内的所有像素整体进训练或整体进测试，
    使训练/测试像素在空间上彼此分离，避免相邻像素跨集泄漏。
    返回训练/测试的「像素索引」（展平后的一维索引）。
    """
    # 每个像素属于哪个网格块：先算它的块行号、块列号，再编码为块 ID
    # 用 linspace 边界保证即使 H/W 不能被 GRID 整除也能均匀分块
    row_idx = np.arange(height)
    col_idx = np.arange(width)
    # 像素行 -> 块行（0..GRID-1）
    block_row = np.minimum((row_idx * GRID) // height, GRID - 1)
    block_col = np.minimum((col_idx * GRID) // width, GRID - 1)

    # 生成 (H, W) 的块 ID 矩阵，块 ID = 块行 * GRID + 块列
    br = block_row[:, None]                 # (H, 1)
    bc = block_col[None, :]                 # (1, W)
    block_id_map = (br * GRID + bc).reshape(-1)   # (H*W,)

    # 对「块」而非「像素」做 70/30 划分
    unique_blocks = np.unique(block_id_map)
    train_blocks, test_blocks = train_test_split(
        unique_blocks, test_size=TEST_SIZE, random_state=SEED
    )

    # 像素索引：所属块在测试块集合里的为测试，其余为训练
    test_block_set = set(test_blocks.tolist())
    is_test = np.array([b in test_block_set for b in block_id_map])
    all_idx = np.arange(block_id_map.size)
    test_idx = all_idx[is_test]
    train_idx = all_idx[~is_test]

    log(f"空间块切分: 共 {len(unique_blocks)} 块  -> "
        f"训练 {len(train_blocks)} 块 / 测试 {len(test_blocks)} 块")
    return train_idx, test_idx


# ============================================================
# 3. 训练 + 评估
#    两种切分共用此函数，保证模型参数、随机种子完全一致。
# ============================================================
def train_and_evaluate(X, y, train_idx, test_idx, split_name):
    log("\n" + "=" * 60)
    log(f"切分方式: {split_name}")
    log("=" * 60)
    log(f"训练像素: {len(train_idx)}   测试像素: {len(test_idx)}")

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    # 相同的模型配置（n_estimators=100, 相同种子），唯一变量是切分方式
    clf = RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=SEED)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # --- 总体准确率 ---
    acc = accuracy_score(y_test, y_pred)

    # --- 混淆矩阵（固定类别顺序 0,1,2）---
    cm = confusion_matrix(y_test, y_pred, labels=CLASS_LABELS)

    # --- 每类 precision/recall/f1 ---
    report_text = classification_report(
        y_test, y_pred, labels=CLASS_LABELS,
        target_names=CLASS_NAMES, digits=4, zero_division=0
    )
    report_dict = classification_report(
        y_test, y_pred, labels=CLASS_LABELS,
        target_names=CLASS_NAMES, output_dict=True, zero_division=0
    )

    # 打印结果
    log(f"\n总体准确率 (overall accuracy): {acc:.4f}")
    log(f"基线准确率 (多数类=植被):       {MAJORITY_BASELINE:.4f}")
    log("\n混淆矩阵 (行=真实, 列=预测, 顺序 0/1/2):")
    log(str(cm))
    log("\n分类报告 (每类 precision / recall / f1):")
    log(report_text)

    # 取出水体类 recall 供验收使用
    water_recall = report_dict[WATER_NAME]["recall"]

    return {
        "split_name": split_name,
        "clf": clf,
        "acc": acc,
        "cm": cm,
        "report_text": report_text,
        "water_recall": water_recall,
        "y_pred": y_pred,
        "test_idx": test_idx,
    }


# ============================================================
# 4. 写出空间块切分的预测栅格 prediction.tif
#    非测试像素填 -1，保留原始空间参考。
# ============================================================
def write_prediction(result, profile, height, width):
    # 整幅初始化为 nodata(-1)，再把测试像素的预测值填回对应位置
    pred_full = np.full(height * width, NODATA_PRED, dtype=np.int16)
    pred_full[result["test_idx"]] = result["y_pred"].astype(np.int16)
    pred_map = pred_full.reshape(height, width)

    out_profile = profile.copy()
    out_profile.update(count=1, dtype="int16", nodata=NODATA_PRED)

    with rasterio.open(PRED_FILE, "w", **out_profile) as dst:
        dst.write(pred_map, 1)
        dst.set_band_description(1, "classification (spatial-block test set)")

    log(f"\n已写出预测栅格: {PRED_FILE} (非测试像素 = {NODATA_PRED})")


# ============================================================
# 5. 验收检查 + 写出 result_summary.txt
#    三条验收标准（针对正确做法=空间块切分）：
#      ① 总体准确率 > 73.4%
#      ② 水体(2) recall ≥ 0.5
#      ③ 空间块准确率「显著低于」随机切分（证明随机切分数据泄漏）
# ============================================================
def acceptance_and_summary(res_random, res_block):
    acc_random = res_random["acc"]
    acc_block = res_block["acc"]
    water_recall_block = res_block["water_recall"]
    acc_gap = acc_random - acc_block

    # ① 准确率超过基线（以正确做法=空间块为准）
    check1 = acc_block > MAJORITY_BASELINE
    # ② 水体 recall 达标（空间块）
    check2 = water_recall_block >= WATER_RECALL_MIN
    # ③ 空间块准确率显著低于随机切分。
    #    "显著"在此量化为：差距为正且 ≥ 0.02（2 个百分点）。
    check3 = acc_gap >= 0.02

    lines = []
    lines.append("=" * 60)
    lines.append("任务二 分类结果摘要 (classify.py)")
    lines.append("=" * 60)
    lines.append("")
    lines.append("【两种切分方式准确率对比】")
    lines.append(f"  随机像素切分 (错误,会数据泄漏): accuracy = {acc_random:.4f}")
    lines.append(f"  空间块切分   (正确)           : accuracy = {acc_block:.4f}")
    lines.append(f"  基线准确率   (多数类=植被)     : accuracy = {MAJORITY_BASELINE:.4f}")
    lines.append(f"  准确率差 (随机 - 空间块)       : {acc_gap:+.4f}")
    lines.append("")
    lines.append("【验收标准检查】(以正确做法=空间块切分为准)")
    lines.append(f"  ① 准确率 > 73.4% ?            "
                 f"{acc_block:.4f} > {MAJORITY_BASELINE}  -> {'通过' if check1 else '不通过'}")
    lines.append(f"  ② 水体(2) recall ≥ 0.5 ?     "
                 f"{water_recall_block:.4f} ≥ {WATER_RECALL_MIN}  -> {'通过' if check2 else '不通过'}")
    lines.append(f"  ③ 空间块准确率显著低于随机切分? "
                 f"差距 {acc_gap:+.4f} ≥ 0.02  -> {'通过' if check3 else '不通过'}")
    lines.append("")
    if check3:
        lines.append("  结论③解读: 空间块切分精度明显更低,说明随机像素切分因空间")
        lines.append("           自相关产生了数据泄漏,其高精度是虚高的。")
    else:
        lines.append("  结论③解读: 两种切分精度差距不明显,未能体现预期的数据泄漏现象,")
        lines.append("           建议检查数据空间结构或块划分。")
    lines.append("")
    all_pass = check1 and check2 and check3
    lines.append(f"【总体验收】: {'全部通过' if all_pass else '存在未通过项,请查看上方明细'}")
    lines.append("=" * 60)

    summary = "\n".join(lines)
    log("\n" + summary)

    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        f.write(summary + "\n")
    log(f"\n已写出摘要文件: {SUMMARY_FILE}")


# ============================================================
# 主流程
# ============================================================
def main():
    X, y, profile, height, width = load_data()
    n_pixels = X.shape[0]

    # --- 切分一：随机像素切分（错误对照）---
    tr_r, te_r = random_split(n_pixels)
    res_random = train_and_evaluate(X, y, tr_r, te_r, "随机像素切分 (random_split)")

    # --- 切分二：空间块切分（正确做法）---
    tr_b, te_b = spatial_block_split(height, width)
    res_block = train_and_evaluate(X, y, tr_b, te_b, "空间块切分 (spatial_block_split)")

    # --- 输出空间块切分的预测栅格 ---
    write_prediction(res_block, profile, height, width)

    # --- 验收检查 + 摘要文件 ---
    acceptance_and_summary(res_random, res_block)


if __name__ == "__main__":
    main()



