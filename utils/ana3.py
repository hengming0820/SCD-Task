import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import norm
from tqdm import tqdm
import csv
DATA_NAME="MD"

if DATA_NAME=="SD":
    ST_CLASSES = ['water', 'ground', 'low vegetation', 'tree', 'building', 'sports field']
    COLORS = [29, 128, 75, 149, 38, 76]
    folder_path = "D:\pycharm\data\socend\label1"
    color_map = {'water': '#0000FF', 'ground': '#808080', 'tree': '#00FF00', 'building': '#800000', 'sports field': '#FF0000',
                 'low vegetation': '#008000'}


elif DATA_NAME=="MD":
    ST_CLASSES = ['road', 'buillding', 'water', 'farm', 'vege', 'background']
    COLORS = [76,149, 178, 225, 29, 255]
    folder_path = "D:\pycharm\data\CZWZ\label1_rgb"
    color_map = {'road':'#FF0000', 'buillding':'#00FF00', 'water':'#00FFFF', 'farm':'#FFFF00', 'vege':'#0000FF', 'background':'#000000'}
def compute_object_areas(label_map,mode):
    """
    计算每个独立地物的面积、像素值、类别名称和颜色值

    Args:
        label_map (numpy.ndarray): 语义分割标签图像

    Returns:
        dict: 每个独立地物的面积、像素值、类别名称和颜色值,以字典形式返回
    """
    object_info = {}
    object_count = 1

    # 获取所有独立的像素值,排除背景像素值0和255
    unique_pixel_values = np.unique(label_map)
    if mode=="SD":
        unique_pixel_values = unique_pixel_values[(unique_pixel_values != 0) & (unique_pixel_values != 255)]
    elif mode=="MD":
        unique_pixel_values = unique_pixel_values[(unique_pixel_values != 0)]

    # 遍历每个独立的像素值
    for pixel_value in unique_pixel_values:
        # 获取该像素值对应的像素索引
        pixel_indices = np.where(label_map == pixel_value)

        # 对该像素值进行连通域分析
        pixel_map = np.zeros_like(label_map, dtype=np.uint8)
        pixel_map[pixel_indices] = 255
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(pixel_map)

        # 获取该像素值对应的类别名称和颜色值
        class_index = COLORS.index(pixel_value)
        class_name = 'unknown'
        color_value = 0
        if 0 <= class_index < len(ST_CLASSES):
            class_name = ST_CLASSES[class_index]
            color_value = COLORS[class_index]

        # 遍历每个连通域
        for i in range(1, num_labels):  # 忽略背景连通域
            area = stats[i, cv2.CC_STAT_AREA]
            object_id = object_count
            object_info[object_id] = {'area': area, 'pixel_value': pixel_value, 'class_name': class_name,
                                      'color_value': color_value}
            object_count += 1

    return object_info


def custom_scaling(counts, power=0.5):
    scaled_sizes = np.power(counts, power)
    scaled_sizes = scaled_sizes / np.max(scaled_sizes) * 100  # 缩放到[0, 100]的范围
    return scaled_sizes

def save_total_object_info_as_csv(total_object_info, csv_path):
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['Object ID', 'Class Name', 'Area']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for object_id, info in total_object_info.items():
            writer.writerow({
                'Object ID': object_id,
                'Class Name': info['class_name'],
                'Area': info['area']
            })

    print(f"Total object info saved as CSV: {csv_path}")


def process_folder(folder_path):
    small_area = 50000
    middle_area = 150000
    big_area = 250000
    fixed_y_values = [25000, 100000, 200000]
    y_stick=[0,small_area,middle_area,big_area]
    plt.figure(figsize=(10, 6),dpi=600)

    plt.rcParams['xtick.direction'] = 'in'  # 将x轴的刻度线方向设置向内
    plt.rcParams['ytick.direction'] = 'in'  # 将y轴的刻度方向设置向内
    total_object_info = {}
    i = 0

    print(
        "\n======================================================================================================================================")
    print(
        f"===================================================={DATA_NAME} datasets LOADING ====================================================")
    print(
        "========================================================================================================================================\n")

    for filename in tqdm(os.listdir(folder_path), total=len(os.listdir(folder_path)), colour="white", ncols=200):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path) and filename.endswith(('.png', '.jpg', '.jpeg')):
            label_map = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            object_info = compute_object_areas(label_map, mode=DATA_NAME)

            # 将当前图像的结果合并到总结果中
            for object_id, info in object_info.items():
                if info['area'] > 5:  # 只保留面积大于5的对象信息
                    total_object_info[i + object_id] = info
                    i += 1
    csv_path = f"{DATA_NAME}_total_object_info.csv"
    save_total_object_info_as_csv(total_object_info, csv_path)
    print(
        "\n=======================================================================================================================================")
    print(
        f"====================================================plotting {DATA_NAME} datasets fig===================================================")
    print(
        "=========================================================================================================================================\n")

    class_counts = {}
    class_sizes = {}

    for object_id, info in tqdm(total_object_info.items(), total=len(total_object_info), colour="green", ncols=200):
        area = info['area']
        class_name = info['class_name']

        if class_name not in class_counts:
            class_counts[class_name] = [0, 0, 0]  # [small_count, middle_count, big_count]
            class_sizes[class_name] = []  # List to store sizes for each class

        if 10<area < small_area:
            class_counts[class_name][0] += 1
            class_sizes[class_name].append('small')
        elif area < middle_area:
            class_counts[class_name][1] += 1
            class_sizes[class_name].append('middle')
        elif area>=middle_area:
            class_counts[class_name][2] += 1
            class_sizes[class_name].append('big')

    x_values = []
    y_values = []
    sizes = []
    alphas = []
    scale=[]
    unique_classes = list(class_counts.keys())
    num_classes = len(unique_classes)
    for class_name, counts in class_counts.items():

        x_values.extend([class_name] * 3)  # Repeat the class name three times (for small, middle, big)
        y_values.extend(fixed_y_values)
        sizes.extend([counts[0], counts[1], counts[2]])
        scale.append(counts/np.sum(counts))
        alphas.extend([0.3, 0.5, 0.7])

    # Normalize sizes for better visualization
    sizes = np.array(sizes)
    scale=np.array(scale).reshape(18,)*100
    scale=np.round(scale,decimals=2)
    sizes_scale = 20 * scale

    # Plot scatter plot
    for i, class_name in enumerate(x_values):
        plt.scatter(class_name, y_values[i], c=color_map[class_name], s=sizes_scale[i], alpha=alphas[i],edgecolors="black")
        plt.annotate(scale[i], (class_name, y_values[i]), xytext=(class_name, y_values[i]+10000), arrowprops=dict(arrowstyle="->"))
    # 添加标签和箭头线
    plt.xlabel('Class Name')
    plt.ylabel('area')
    plt.yticks(y_stick)
    plt.savefig(DATA_NAME + "_Counts.png")
    plt.show()

process_folder(folder_path)
