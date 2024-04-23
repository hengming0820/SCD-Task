import cv2
import numpy as np

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

def compute_object_areas(label_map):
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
    unique_pixel_values = unique_pixel_values[(unique_pixel_values != 0) & (unique_pixel_values != 255)]

    # 遍历每个独立的像素值
    for pixel_value in unique_pixel_values:
        # 获取该像素值对应的像素索引
        pixel_indices = np.where(label_map == pixel_value)

        # 对该像素值进行连通域分析
        pixel_map = np.zeros_like(label_map, dtype=np.uint8)
        pixel_map[pixel_indices] = 255
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(pixel_map)

        # 获取该像素值对应的类别名称和颜色值
        class_index = COLORS.index(pixel_value)  # 假设像素值从1开始对应类别
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

# 假设你有一个语义分割标签图像 label_map
label_map = cv2.imread(r"D:\pycharm\data\CZWZ\label1_rgb\69.png", cv2.IMREAD_GRAYSCALE)

# 调用计算面积的函数
object_info = compute_object_areas(label_map)

# 打印每个独立地物的面积、像素值、编号和所属类别
# 打印每个独立地物的面积、像素值、类别名称和颜色值
for object_id, info in object_info.items():
    area = info['area']
    pixel_value = info['pixel_value']
    class_name = info['class_name']
    color_value = info['color_value']
    print(f"Object {object_id} area: {area} pixels, pixel value: {pixel_value}, class name: {class_name}, color value: {color_value}")