from PIL import Image
import os

def convert_png_to_jpg(folder_path):
    # 检查文件夹是否存在
    if not os.path.isdir(folder_path):
        print("文件夹不存在。")
        return

    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # 检查文件是否是PNG文件
        if os.path.isfile(file_path) and filename.lower().endswith('.png'):
            try:
                # 打开PNG文件
                image = Image.open(file_path)

                # 创建目标文件的文件名（将文件扩展名从.png改为.jpg）
                new_filename = os.path.splitext(filename)[0] + '.jpg'
                new_file_path = os.path.join(folder_path, new_filename)

                # 将PNG文件保存为JPG文件
                image.save(new_file_path, 'JPEG')

                # 关闭源图像
                image.close()

                # 删除源PNG文件
                os.remove(file_path)

                print(f"转换成功: {filename} -> {new_filename}")

            except Exception as e:
                print(f"转换失败: {filename} ({e})")

    print("转换完成。")

# 指定包含PNG文件的文件夹路径
folder_path = r"D:\ZMH\net\data\merge\im2"

# 调用函数进行转换
convert_png_to_jpg(folder_path)