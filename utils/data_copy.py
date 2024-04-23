import os
import shutil

# 源文件夹路径
source_folder = r"D:\ZMH\net\data\merge\label1"
# 目标文件夹路径
destination_folder = r"D:\ZMH\net\data\merge\label1_aug"
# 指定要复制的文件列表
list_folder=r"D:\ZMH\net\data\merge\data_list"

def _read_images_list(mode, data_path):
    images_list_file = os.path.join(data_path, mode + ".txt")
    with open(images_list_file, "r") as f:
        return f.readlines()
# 遍历指定文件列表

list=_read_images_list("train",list_folder)
for i,num in enumerate(list):
    file_name = num.rstrip('\n')
    source_file = os.path.join(source_folder, file_name)
# 构建目标文件路径
    destination_file = os.path.join(destination_folder, file_name)
# 执行文件复制操作
    shutil.copy2(source_file, destination_file)