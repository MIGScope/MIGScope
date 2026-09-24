import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

def draw_image_from_csv(input_path, output_path, width=4096, threshold=590, mode="imshow", point_size=1):
    """
    从CSV绘制图像，可以选择 imshow 或 scatter 模式。
    :param input_path: 输入CSV文件路径
    :param output_path: 输出图像文件路径
    :param width: 每行数据点数
    :param threshold: 二值化阈值
    :param mode: 'imshow' 或 'scatter'
    :param point_size: scatter模式下小方块大小
    """
    # 读CSV并展平成一维
    df = pd.read_csv(input_path, header=None)
    data = df.values.flatten().astype(int)

    # reshape
    height = len(data) // width
    data = data[:width * height]
    data_2d = data.reshape((height, width))

    # 阈值化
    data_bin = (data_2d > threshold).astype(int)

    fig, ax = plt.subplots(figsize=(8, 20))

    if mode == "imshow":
        # ✅ imshow 保真显示每个点
        ax.imshow(data_bin, cmap="black", interpolation="nearest", aspect="equal")
    elif mode == "scatter":
        # ✅ scatter 每个点渲染成小方块
        y, x = np.nonzero(data_bin)
        ax.scatter(x, y, c="black", s=point_size, marker="s")  # s 控制方块大小
        ax.set_xlim([0, width])
        ax.set_ylim([height, 0])  # 翻转y轴
    else:
        raise ValueError("mode must be 'imshow' or 'scatter'")

    # 去掉坐标轴
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(output_path, dpi=800)
    plt.close()


# 使用示例
if __name__ == "__main__":
    # imshow 模式
    #draw_image_from_csv("result/result.csv", "result.png", mode="imshow")

    # scatter 模式，方块大小为 2
    draw_image_from_csv("result/result.csv", "vgg16.png", mode="scatter", point_size=0.1)
