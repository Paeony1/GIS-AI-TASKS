"""打印 input.tif 的元数据，用于判断 Red / NIR 波段位置。只读取信息，不做任何计算。"""

import rasterio

FILENAME = "input.tif"


def main():
    with rasterio.open(FILENAME) as src:
        print(f"文件: {FILENAME}")
        print("=" * 60)

        # 波段数量
        print(f"波段数量 (count): {src.count}")
        print("-" * 60)

        # 数据类型（每个波段）
        print(f"数据类型 (dtypes): {src.dtypes}")
        print("-" * 60)

        # nodata 值
        print(f"nodata 值: {src.nodata}")
        print(f"每波段 nodata (nodatavals): {src.nodatavals}")
        print("-" * 60)

        # 坐标系与仿射变换
        print(f"坐标系 (CRS): {src.crs}")
        print(f"仿射变换 (transform):\n{src.transform}")
        print("-" * 60)

        # 数据集级别的标签
        print("数据集标签 (dataset tags):")
        for k, v in src.tags().items():
            print(f"  {k}: {v}")
        print("-" * 60)

        # 每个波段的描述与标签
        for i in range(1, src.count + 1):
            print(f"波段 {i}:")
            print(f"  描述 (description): {src.descriptions[i - 1]}")
            print(f"  颜色解释 (colorinterp): {src.colorinterp[i - 1]}")
            band_tags = src.tags(i)
            if band_tags:
                print("  标签 (tags):")
                for k, v in band_tags.items():
                    print(f"    {k}: {v}")
            else:
                print("  标签 (tags): (无)")
            print()


if __name__ == "__main__":
    main()
