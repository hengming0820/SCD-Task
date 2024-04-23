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


def process_folder(folder_path):
    small_area=40000
    small_num=0
    middle_area=120000
    middle_num=0
    big_area=200000
    big_num=0
    plt.figure(figsize=(10, 6))
    y_stick=np.linspace(0,250000,8000)
    plt.rcParams['xtick.direction'] = 'in'  # 将x周的刻度线方向设置向内
    plt.rcParams['ytick.direction'] = 'in'  # 将y轴的刻度方向设置向内
    total_object_info = {}
    all_areas = []
    i=0
    print("\n======================================================================================================================================")
    print(f"===================================================={DATA_NAME} datasets LOADING ====================================================")
    print("========================================================================================================================================\n")
    for filename in tqdm(os.listdir(folder_path),total=len(os.listdir(folder_path)),colour="white",ncols=200)   :
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path) and filename.endswith(('.png', '.jpg', '.jpeg')):
            label_map = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            object_info = compute_object_areas(label_map,mode=DATA_NAME)

            # 将当前图像的结果合并到总结果中
            for object_id, info in object_info.items():
                total_object_info[i+object_id] = info
                i+=1
            # 收集所有独立地物的面积值
            all_areas.extend([info['area'] for info in object_info.values()])
    # 打印总结果
    #print("Overall statistics:")
    print("\n=======================================================================================================================================")
    print(f"====================================================plotting {DATA_NAME} datasets fig===================================================")
    print("=========================================================================================================================================\n")
    for object_id, info in tqdm(total_object_info.items(),total=len(total_object_info),colour="green",ncols=200):
        area = info['area']
        if 5< area <80000:
            small_num+=1
        elif 80000<=area<160000:
            middle_num+=1
        elif area>120000:
            big_num+=1
        size=small_num
        pixel_value = info['pixel_value']
        class_name = info['class_name']
        color_value = info['color_value']
        # print(
        #     f"Object {object_id} area: {area} pixels, pixel value: {pixel_value}, class name: {class_name}, color value: {color_value}")
        if area>=5:
            plt.scatter(class_name, area,s=size,c=color_map[class_name])

    #plt.title(f'Distribution of Object Areas (mean={mean:.2f}, std={std_dev:.2f})')
    plt.yticks(y_stick)
    plt.xlabel('Class_name')
    plt.ylabel('Area (pixels)')
    plt.savefig(DATA_NAME+"_Area.png")
    plt.show()

# 使用示例
process_folder(folder_path)
