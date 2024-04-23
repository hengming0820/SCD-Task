import numpy as np
import os
import cv2
from PIL import Image
def mask_to_image(mask):#白色背景类没有改变
    colors = [
        [0, 0, 0],
        [0, 0, 255],#road
        [0, 255, 0],#building
        [255, 255, 0],#water
        [0, 255, 255],#farm
        [255, 0, 0],#vege
        [255,255,255]
    ]#符合原始数据标签,崇州乌镇
    # colors=[
    #     [255,255,255],
    #     [0,0,255],
    #     [128,0,0],
    #     [0,255,0],
    #     [255,0,0],
    #     [0,0,0]
    # ]
    #CNAM-CD
    # colors=[
    #     [255,255,255],
    #     [0,165,255],
    #     [100,30,230],
    #     [0,140,70],
    #     [214,112,218],
    #     [240,170,0],
    #     [170,235,127],
    #     [0,80,230],
    #     [57,220,205],
    #     [32,165,218]
    # ]
    labels = [0,
              76,#road
              149,#building
              178,#water
              225,#farm
              29,#vege
              255
              ]
    # labels = [0,
    #           1,#road
    #           2,
    #           3,#building
    #           4,#water
    #           5,#farm
    #           ]
    # labels=[
    #     0,
    #      1,
    #      2,
    #      3,
    #      4,
    #      5,
    #      6,
    #      7,
    #      8,
    #      9
    # ]
    assert len(mask.shape) == 2
    img=(mask).astype(np.uint8)
    dst = np.repeat(np.expand_dims(img, axis=2), 3, axis=2)
    for i in range(len(colors)):
        dst[img == labels[i]] = colors[i]

    #return cv2.cvtColor(dst,cv2.COLOR_BGR2RGB)
    return dst
if __name__ == '__main__':
    dst_path = r"D:\ZMH\net\data\merge\label2_new"
    path=r"D:\ZMH\net\data\merge\label2_rgb"
    for name in os.listdir(dst_path):
        if name.endswith(('.jpg', '.jpeg', '.png','.tif')):
            img_path=os.path.join(path,name)
            color_path=os.path.join(dst_path,name)
            # print(color_path)
            # img=cv2.imread(img_path, 0)
            # img_color=mask_to_image(img)
            # print(img_color.shape)
            # cv2.imwrite(color_path,img_color)
            color = cv2.imread(color_path,cv2.IMREAD_UNCHANGED)
            #img = color2class(color)
            img = mask_to_image(color)
            cv2.imwrite(img_path, img)