import os
from PIL import Image

def calculate_pixel_statistics(folder_path):
    pixel_counts = {}
    total_pixels = 0

    for filename in os.listdir(folder_path):
        if filename.endswith(('.jpg', '.jpeg', '.png','.tif')):
            file_path = os.path.join(folder_path, filename)
            image = Image.open(file_path)
            pixels = image.load()

            width, height = image.size
            total_pixels += width * height

            for x in range(width):
                for y in range(height):
                    pixel = pixels[x, y]
                    if pixel in pixel_counts:
                        pixel_counts[pixel] += 1
                    else:
                        pixel_counts[pixel] = 1

    pixel_statistics = {}
    for pixel, count in pixel_counts.items():
        ratio = count / total_pixels
        pixel_statistics[pixel] = {
            'count': count,
            'ratio': ratio
        }

    return pixel_statistics

# 指定文件夹路径
folder_path =r"D:\ZMH\net\socend\label1"

result = calculate_pixel_statistics(folder_path)
for pixel, statistics in result.items():
    print(f"Pixel: {pixel}, Count: {statistics['count']}, Ratio: {statistics['ratio']}")