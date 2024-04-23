import cv2
import numpy as np
import os

# 定义图像文件夹路径
image_folder =r"D:\ZMH\net\data\Hi_UCD\train\im1"

# 获取图像文件夹中的所有图像文件
image_files = os.listdir(image_folder)

# 初始化各通道像素均值和标准差列表
b_mean_values = []
g_mean_values = []
r_mean_values = []
b_std_values = []
g_std_values = []
r_std_values = []

# 遍历图像文件
for image_file in image_files:
    # 构建图像文件的完整路径
    image_path = os.path.join(image_folder, image_file)

    # 读取图像
    image = cv2.imread(image_path)

    # 分离B、G、R通道
    b, g, r = cv2.split(image)

    # 计算各通道的像素均值和标准差
    b_mean = np.mean(b)
    g_mean = np.mean(g)
    r_mean = np.mean(r)
    b_std = np.std(b)
    g_std = np.std(g)
    r_std = np.std(r)

    # 将各通道的像素均值和标准差添加到列表中
    b_mean_values.append(b_mean)
    g_mean_values.append(g_mean)
    r_mean_values.append(r_mean)
    b_std_values.append(b_std)
    g_std_values.append(g_std)
    r_std_values.append(r_std)

# 计算所有图像的B、G、R通道像素均值和标准差的平均值
average_b_mean = np.mean(b_mean_values)
average_g_mean = np.mean(g_mean_values)
average_r_mean = np.mean(r_mean_values)
average_b_std = np.mean(b_std_values)
average_g_std = np.mean(g_std_values)
average_r_std = np.mean(r_std_values)

# 输出结果
print("所有图像的B通道像素均值的平均值:", average_b_mean)
print("所有图像的G通道像素均值的平均值:", average_g_mean)
print("所有图像的R通道像素均值的平均值:", average_r_mean)
print("RGB",[np.round(average_r_mean,2),np.round(average_g_mean,2),np.round(average_b_mean,2)])
print("所有图像的B通道像素标准差的平均值:", average_b_std)
print("所有图像的G通道像素标准差的平均值:", average_g_std)
print("所有图像的R通道像素标准差的平均值:", average_r_std)
print("RGB",[np.round(average_r_std,2),np.round(average_g_std,2),np.round(average_b_std,2)])