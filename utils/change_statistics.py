import os
import numpy as np
import cv2

# 定义类别映射
category_mapping = {
    0: '无变化',
    76: '道路',
    149: '建筑',
    178: '水体',
    225: '耕地',
    29: '植被',
    255: '背景'
}
# category_mapping = {
#     255: '无变化',
#     29: '水',
#     38: '建筑',
#     75: '灌木',
#     76: '操场',
#     128: '裸地',
#     149: '树'
# }
# category_mapping = {
#     173: '沙漠',
#     97: '建筑',
#     90: '农田',
#     127: '水体',
#     255: '无变化'
# }

# category_mapping = {
#     0: '无变化',
#     1: '水体',
#     2: '草地',
#     3: '建筑',
#     4: '大鹏',
#     5: '道路',
#     6: '桥梁',
#     7: '其他',
#     8: '裸地',
#     9: '树林',
# }

# 文件夹路径
time1_folder = r"D:\ZMH\net\PRED_DIR\MD\SCDNet\im1_rgb"# 第一个时相的标签文件夹路径
time2_folder = r"D:\ZMH\net\PRED_DIR\MD\SCDNet\im2_rgb" # 第二个时相的标签文件夹路径

# 初始化变化方向统计字典
change_directions = {}

# 遍历第一个时相的文件夹
for filename in os.listdir(time1_folder):
    if filename.endswith(".png"):  # 假设标签是以.png文件存储的
        time1_file_path = os.path.join(time1_folder, filename)
        time2_file_path = os.path.join(time2_folder, filename)

        # 读取标签数据
        label_time1 = cv2.imread(time1_file_path, cv2.IMREAD_GRAYSCALE)
        label_time2 = cv2.imread(time2_file_path, cv2.IMREAD_GRAYSCALE)

        # 统计变化方向
        for i in range(label_time1.shape[0]):
            for j in range(label_time1.shape[1]):
                label1 = label_time1[i, j]
                label2 = label_time2[i, j]

                # 判断变化方向
                if label1 != label2:
                    direction = f"T1{category_mapping[label1]}变T2{category_mapping[label2]}"
                    change_directions[direction] = change_directions.get(direction, 0) + 1

# 计算比例
total_changes = sum(change_directions.values())
change_directions_percentages = {direction: count / total_changes for direction, count in change_directions.items()}

print("变化方向及对应比例：")
for direction, percentage in change_directions_percentages.items():
    print(f"{direction}: {percentage}")
