"""
leakage_demo.py — 受控演示：空间数据泄漏的成因（任务二第3关）
================================================================
背景：classify.py 在原始数据上两种切分都得到 100%。原因不是"泄漏被
      掩盖了"，而是 labels.tif 是 scene.tif 波段的确定性函数——每个
      像素靠自己的 4 个波段值即可被完美分类，模型根本用不到空间邻居，
      因此【无信息可泄漏】，两种切分都是 100% 是真实结果，不是虚高。

      泄漏要发生，必须同时满足两个条件：
        (1) 单个像素自身分不准（否则模型用不着邻居）；
        (2) 测试像素与其在训练集中的邻居几乎一模一样（模型记住邻居=偷看答案）。
      原始数据连条件(1)都不满足，故无泄漏可演示。

本脚本通过受控实验，证明"泄漏的根源是空间结构"，而非噪声本身：

  A. 可分性诊断：打印各类在 4 波段上的取值范围，验证"原始数据每像素可分"。

  B. 两种噪声对照（核心）：人为往波段掺噪声，破坏 per-pixel 可分性，
     再看两种切分会不会拉开差距：
       - i.i.d. 噪声（白噪声）：每像素噪声独立 -> 满足条件(1)但破坏(2)
         -> 相邻像素不再相似 -> 演示不出泄漏（差距≈0 或为负）。
       - 空间相关噪声（成片平滑）：相邻像素共享相近噪声 -> 同时满足(1)(2)
         -> 随机切分能靠训练邻居作弊而虚高 -> 差距为正，泄漏被演示出来。
     真实遥感影像的地物本就成片、有空间结构，故空间相关噪声正是对
     "真实影像特性"的模拟。

  ⚠️ 说明：B 中的噪声是【人为受控注入】，仅用于暴露切分方式的差异，
     不是原始数据的真实情况，不对原始数据做任何篡改。

  结论：同一个模型，仅因切分方式不同，"看起来的"准确率可以天差地别。
        所以光看 accuracy 无法判断模型好坏，必须先追问这个数字是用
        哪种切分得来的——这就是"为什么一个准确率 99% 的模型可能毫无用处"。
"""

import numpy as np
import rasterio
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

SCENE_FILE = "scene.tif"
LABELS_FILE = "labels.tif"
SEED = 42
N_ESTIMATORS = 100
TEST_SIZE = 0.30
GRID = 10
CLASS_LABELS = [0, 1, 2]
CLASS_NAMES = ["其他(0)", "植被(1)", "水体(2)"]
NOISE_LEVELS = [0.0, 0.10, 0.20, 0.30]   # 高斯噪声标准差（反射率单位）


def load():
    with rasterio.open(SCENE_FILE) as src:
        bands = src.read().astype(np.float32)      # (4, H, W)
        h, w = src.height, src.width
    with rasterio.open(LABELS_FILE) as src:
        labels = src.read(1)
    X = bands.transpose(1, 2, 0).reshape(-1, bands.shape[0])
    y = labels.reshape(-1)
    return X, y, h, w


# ---------- A. 可分性诊断 ----------
def diagnose(X, y):
    print("=" * 60)
    print("A. 可分性诊断：各类在 4 波段上的取值范围")
    print("=" * 60)
    names = ["Red", "NIR", "Green", "SWIR"]
    for c in CLASS_LABELS:
        m = y == c
        print(f"类别 {c} (n={m.sum()}):")
        for b, nm in enumerate(names):
            print(f"    {nm:5s}: [{X[m, b].min():.4f}, {X[m, b].max():.4f}]")
    print("\n说明：若各类区间互不重叠 -> 每个像素独立即可完美分类，")
    print("     原始数据无可泄漏信息，故两种切分都是 100%。\n")


# ---------- 切分 ----------
def random_idx(n):
    return train_test_split(np.arange(n), test_size=TEST_SIZE, random_state=SEED)


def block_idx(h, w):
    rb = np.minimum((np.arange(h) * GRID) // h, GRID - 1)
    cb = np.minimum((np.arange(w) * GRID) // w, GRID - 1)
    bid = (rb[:, None] * GRID + cb[None, :]).reshape(-1)
    blocks = np.unique(bid)
    tr_b, te_b = train_test_split(blocks, test_size=TEST_SIZE, random_state=SEED)
    te_set = set(te_b.tolist())
    is_te = np.array([b in te_set for b in bid])
    allidx = np.arange(bid.size)
    return allidx[~is_te], allidx[is_te]


def run_split(X, y, tr, te):
    clf = RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=SEED)
    clf.fit(X[tr], y[tr])
    pred = clf.predict(X[te])
    acc = accuracy_score(y[te], pred)
    rep = classification_report(y[te], pred, labels=CLASS_LABELS,
                                target_names=CLASS_NAMES, output_dict=True,
                                zero_division=0)
    water_recall = rep["水体(2)"]["recall"]
    return acc, water_recall


# ---------- 两种噪声的生成 ----------
def iid_noise(rng, shape, sigma):
    """i.i.d. 高斯白噪声：每个像素、每个波段彼此独立。
    破坏 per-pixel 可分性，但也破坏了相邻像素的相似性 -> 演示不出泄漏。"""
    return rng.normal(0, sigma, shape).astype(np.float32)


def _smooth_field(field, h, w, k=7, iters=3):
    """对一张 (h*w,) 的白噪声场做可分离盒式均值滤波（多次迭代），
    得到空间平滑（相邻像素相近）的噪声场。用累积和实现 O(N) 滤波。"""
    a = field.reshape(h, w).copy()
    for _ in range(iters):
        # 行方向均值
        c = np.cumsum(np.pad(a, ((k, k), (0, 0)), mode="edge"), axis=0)
        a = (c[2 * k:, :] - c[:-2 * k, :]) / (2 * k)
        # 列方向均值
        c = np.cumsum(np.pad(a, ((0, 0), (k, k)), mode="edge"), axis=1)
        a = (c[:, 2 * k:] - c[:, :-2 * k]) / (2 * k)
    return a.reshape(-1)


def spatial_noise(rng, h, w, n_bands, sigma):
    """空间相关噪声：先生成白噪声再做空间平滑，使相邻像素共享相近噪声。
    平滑会缩小方差，故乘以 sqrt(2*k*iters) 把幅度大致拉回 sigma 量级。
    每个波段独立生成一张噪声场。模拟真实遥感影像'地物成片'的空间结构。"""
    out = np.empty((h * w, n_bands), dtype=np.float32)
    amp = np.sqrt(2 * 7 * 3)  # 与 _smooth_field 的 k=7, iters=3 对应
    for b in range(n_bands):
        white = rng.normal(0, 1, h * w).astype(np.float32)
        out[:, b] = sigma * _smooth_field(white, h, w) * amp
    return out


# ---------- B. 两种噪声对照演示 ----------
def noise_demo(X, y, h, w):
    print("=" * 60)
    print("B. 两种噪声对照：证明'泄漏的根源是空间结构'，而非噪声本身")
    print("=" * 60)
    tr_r, te_r = random_idx(len(y))
    tr_b, te_b = block_idx(h, w)
    n_bands = X.shape[1]

    header = (f"{'噪声类型':>12} | {'σ':>5} | {'随机切分acc':>11} | "
              f"{'空间块acc':>10} | {'差距(泄漏)':>10} | {'块_水体recall':>12}")
    print(header)
    print("-" * len(header))

    rows = []
    for sigma in NOISE_LEVELS:
        # 每个 sigma 用独立的 rng，保证两种噪声类型起点一致、可复现
        if sigma == 0.0:
            # 原始数据：无噪声，作为基准（两种切分都应 ≈100%）
            acc_r, _ = run_split(X, y, tr_r, te_r)
            acc_b, wr_b = run_split(X, y, tr_b, te_b)
            rows.append(("原始(无噪)", sigma, acc_r, acc_b, acc_r - acc_b, wr_b))
            print(f"{'原始(无噪)':>12} | {sigma:>5.2f} | {acc_r:>11.4f} | "
                  f"{acc_b:>10.4f} | {acc_r - acc_b:>+10.4f} | {wr_b:>12.4f}")
            continue

        # i.i.d. 白噪声
        rng = np.random.default_rng(SEED)
        Xn = X + iid_noise(rng, X.shape, sigma)
        acc_r, _ = run_split(Xn, y, tr_r, te_r)
        acc_b, wr_b = run_split(Xn, y, tr_b, te_b)
        rows.append(("i.i.d.白噪声", sigma, acc_r, acc_b, acc_r - acc_b, wr_b))
        print(f"{'i.i.d.白噪声':>12} | {sigma:>5.2f} | {acc_r:>11.4f} | "
              f"{acc_b:>10.4f} | {acc_r - acc_b:>+10.4f} | {wr_b:>12.4f}")

        # 空间相关噪声
        rng = np.random.default_rng(SEED)
        Xs = X + spatial_noise(rng, h, w, n_bands, sigma)
        acc_r, _ = run_split(Xs, y, tr_r, te_r)
        acc_b, wr_b = run_split(Xs, y, tr_b, te_b)
        rows.append(("空间相关噪声", sigma, acc_r, acc_b, acc_r - acc_b, wr_b))
        print(f"{'空间相关噪声':>12} | {sigma:>5.2f} | {acc_r:>11.4f} | "
              f"{acc_b:>10.4f} | {acc_r - acc_b:>+10.4f} | {wr_b:>12.4f}")

    print("\n解读：")
    print("  · i.i.d. 白噪声：差距≈0 或为负。每像素噪声独立，相邻像素不再相似，")
    print("    随机切分无邻居可作弊 -> 演示不出泄漏。")
    print("  · 空间相关噪声：σ 增大后差距转为正值。相邻像素共享相近噪声，")
    print("    随机切分的测试像素在训练集里有近似副本 -> 精度虚高 = 数据泄漏。")
    print("  · 结论：泄漏来自'空间结构(相邻像素相似)'，不是噪声大小本身。")
    print("    真实遥感影像天然有此结构，故必须用空间块切分才能得到可信精度。")
    return rows


def main():
    X, y, h, w = load()
    diagnose(X, y)
    noise_demo(X, y, h, w)


if __name__ == "__main__":
    main()
