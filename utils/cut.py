# -- coding: utf-8 --
import cv2
import numpy as np
import os
def cut_image(src_image1_path,src_image2_path,dst_image_path,dst_label_path,image_size,k=0):
    """
    输入大图原图和标签，得到切割后的小图存入对应路径
    :param src_imgge_path: 三通道图路径
    :param src_label_path: 标签路径
    :param dst_image_path: 小图保存路径
    :param dst_label_path: 小图标签路径
    :param image_size: 切割大小 整数
    :k: 命名数字的起点
    :return:
    """
    def cut_(img_path,dst_path,img_size,img_type,img_tail,k):
        num=1
        fileanme=os.path.basename(img_path)
        fileanme=fileanme.split('.')[0]
        if img_type=='label':
            #gray_label
            img = np.expand_dims(cv2.imread(img_path,0),axis=2)
            # #img=np.expand_dims(cv2.resize(img,(15360,15360)),axis=2)
            # img=np.expand_dims(cv2.resize(img,(8000,8000)),axis=2)
            #channels = 1
            #bgr_label
            #img=cv2.imread(img_path)
            #img = cv2.resize(img, (4096, 6144))
            #img = cv2.resize(img, (15360, 15360))
            channels = 1
        else:
            img = cv2.imread(img_path)

            #img = cv2.imread(img_path,im)
            #img = cv2.resize(img, (4096, 6144))
            #img = cv2.resize(img, (15360, 15360))
            channels = 3

            # 检查图像深度
            depth = img.dtype

            if depth == 'float64':
                # 将图像深度转换为8位无符号整数
                img = cv2.convertScaleAbs(img)

                # 继续使用转换后的图像进行后续处理
                # ...

                # 显示或保存转换后的图像
        # num_x = img.shape[0] // img_size+1
        # num_y = img.shape[1] // img_size+1
        num_y=4
        num_x=4

        img_paded = np.zeros(shape=[num_x * img_size,num_y*image_size,channels])
        print(img_paded.shape)
        img_paded[:img.shape[0],:img.shape[1],:]=img
        for i in range(num_x):
            for j in range(num_y):
                roi = img_paded[i * img_size:(i + 1) * img_size, j * img_size:(j + 1) * img_size]
                # 把 256*256 resize成 512*512
                # roi = cv2.resize(roi, (0, 0), fx=2, fy=2)
                # if img_type=='label':
                #     roi[roi == 1] = 255


                cv2.imwrite(dst_path+'\\'+f'{fileanme}_{num}.'+img_tail , roi)
                k += 1
                num+=1

    cut_(src_image1_path,dst_image_path,image_size,'label','png',k)
    cut_(src_image2_path,dst_label_path,image_size,'label','png',k)

    # cut_(src_image1_path,dst_image_path,image_size,'im','png',k)
    # cut_(src_image2_path,dst_label_path,image_size,'im','png',k)


if __name__ == '__main__':
    # src_image_path=r'E:\siyu\projects\gf_contest\test_submit\austin1_.tif'
    # src_label_path = r'E:\siyu\projects\gf_contest\test_submit\austin1.tif'
    #
    # dst_image_path=r'E:\siyu\projects\gf_contest\test_submit\test_cut\src'
    # dst_label_path=r'E:\siyu\projects\gf_contest\test_submit\test_cut\label'
    #
    #
    # image_size=256
    # k=0
    # #切图
    # cut_image(src_image_path,src_label_path,dst_image_path,dst_label_path,image_size,k)
    # print(k)

    src_paths=r"D:\ZMH\net\data\Hi_UCD\val\labelA"
    label_paths=r"D:\ZMH\net\data\Hi_UCD\val\labelB"
    dst_src_path=r"D:\ZMH\net\data\Hi_UCD\val256\labelA"
    dst_label_path=r"D:\ZMH\net\data\Hi_UCD\val256\labelB"
    os.makedirs(dst_src_path, exist_ok=True)
    os.makedirs(dst_label_path, exist_ok=True)
    img_size=256
    k=1
    n=1
    for name in os.listdir(src_paths):

        src_dir=os.path.join(src_paths,name)
        label_dir=os.path.join(label_paths,name)
        cut_image(src_dir,label_dir,dst_src_path,dst_label_path,img_size,k)
        n+=1
        k+=16
        if n>300:
            break


