import os
import random

import albumentations as A
import cv2
import numpy as np
import torch
from skimage import io
from torch.utils import data
import utils.transform as transform
import matplotlib.pyplot as plt
from torchvision.transforms import functional as F

num_classes = 7
ST_COLORMAP=[[0,0,0], [255,0,0], [0,255,0], [0,255,255], [255,255,0], [0,0,255], [255,255,255]]#RGB
ST_CLASSES = ['unchanged', 'road', 'buillding', 'water', 'farm', 'vege', 'background']
#乌镇
# MEAN_A = np.array([63.67 ,64.86 ,62.83])
# STD_A  = np.array([29.20 ,31.33 ,34.68])
# MEAN_B = np.array([79.26 ,86.94 ,77.32])
# STD_B  = np.array([32.08 ,34.54 ,39.81])
#崇州
# MEAN_A = np.array([105.72 ,111.11 ,106.58])
# STD_A  = np.array([33.97 ,38.39 ,39.80])
# MEAN_B = np.array([80.49 ,81.74 ,70.25])
# STD_B  = np.array([32.73 ,35.46 ,37.60])
#merge
MEAN_A = np.array([86.60 ,90.09 ,86.69])
STD_A  = np.array([34.80 ,35.18 ,37.47])
MEAN_B = np.array([79.93 ,84.11 ,73.46])
STD_B  = np.array([ 32.44 ,35.04 ,38.61])
#aug
# MEAN_A = np.array([82.72, 86.1, 82.8])
# STD_A  = np.array([40.2, 38.65, 35.36])
# MEAN_B = np.array([70.2, 80.47, 76.19])
# STD_B  = np.array([40.39, 38.11, 35.52])
root = r"D:\pycharm\data\CZWZ"

colormap2label = np.zeros(256 ** 3)
for i, cm in enumerate(ST_COLORMAP):
    colormap2label[(cm[0] * 256 + cm[1]) * 256 + cm[2]] = i

def Colorls2Index(ColorLabels):
    IndexLabels = []
    for i, data in enumerate(ColorLabels):
        IndexMap = Color2Index(data)
        IndexLabels.append(IndexMap)
    return IndexLabels

def Color2Index(ColorLabel):
    data = ColorLabel.astype(np.int32)
    idx = (data[:, :, 0] * 256 + data[:, :, 1]) * 256 + data[:, :, 2]
    IndexMap = colormap2label[idx]
    #IndexMap = 2*(IndexMap > 1) + 1 * (IndexMap <= 1)
    IndexMap = IndexMap * (IndexMap < num_classes)
    return IndexMap

def Index2Color(pred):
    colormap = np.asarray(ST_COLORMAP, dtype='uint8')
    x = np.asarray(pred, dtype='int32')
    return colormap[x, :]

def showIMG(img):
    plt.imshow(img)
    plt.show()
    return 0

def _read_images_list(mode, data_path):
    images_list_file = os.path.join(data_path, 'data_list', mode + ".txt")
    with open(images_list_file, "r") as f:
        return f.readlines()

def normalize_image(im, time='A'):
    assert time in ['A', 'B']
    if time=='A':
        im = (im - MEAN_A) / STD_A
    else:
        im = (im - MEAN_B) / STD_B
    return im

def tensor2int(im, time='A'):
    assert time in ['A', 'B']
    if time=='A':
        im = im * STD_A + MEAN_A
    else:
        im = im * STD_B + MEAN_B
    return im.astype(np.uint8)

def normalize_images(imgs, time='A'):
    for i, im in enumerate(imgs):
        imgs[i] = normalize_image(im, time)
    return imgs
def label_process(x_label):
    a=np.shape(x_label)
    prediction=np.zeros(shape=a)
    mask1 = (x_label ==0)#无变化
    mask2 = (x_label == 76)#road
    mask3 = (x_label ==149)#building
    mask4 = (x_label == 178)#water
    mask5 = (x_label == 225)#farm
    mask6 = (x_label == 29)#vege
    mask7 = (x_label == 255)#背景
    prediction[mask1] = 0
    prediction[mask2] = 1
    prediction[mask3] = 2
    prediction[mask4] = 3
    prediction[mask5] = 4
    prediction[mask6] = 5
    prediction[mask7] = 6
    return prediction

# def label_process(x_label):
#     a=np.shape(x_label)
#     prediction=np.zeros(shape=a)
#     mask1 = (x_label ==0)#无变化
#     mask2 = (x_label == 76)#road
#     mask3 = (x_label ==149)#building
#     mask4 = (x_label == 178)#water
#     mask5 = (x_label == 225)#farm
#     mask6 = (x_label == 29)#vege
#     mask7 = (x_label == 255)#背景
#     prediction[mask1] = 0
#     prediction[mask2] = 1
#     prediction[mask3] = 1
#     prediction[mask4] = 2
#     prediction[mask5] = 3
#     prediction[mask6] = 4
#     prediction[mask7] = 5
#     return prediction
# def read_RSimages(mode):
#     #assert mode in ['train', 'val', 'test']
#     img_A_dir = os.path.join(root,  'im1')
#     img_B_dir = os.path.join(root,  'im2')
#     label_A_dir = os.path.join(root, 'label1')
#     label_B_dir = os.path.join(root,  'label2')
#     #label_A_dir = os.path.join(root, mode, 'label1_rgb')
#     #label_B_dir = os.path.join(root, mode, 'label2_rgb')
#
#     #data_list = os.listdir(img_A_dir)
#     data_list=_read_images_list(mode,root)
#     imgs_list_A, imgs_list_B, labels_A, labels_B = [], [], [], []
#     count = 0
#     for idx, it in enumerate(data_list):
#         # print(it)
#         if (it[-4:]=='.jpg'):
#             img_A_path = os.path.join(img_A_dir, it)
#             img_B_path = os.path.join(img_B_dir, it)
#             label_A_path = os.path.join(label_A_dir, it)
#             label_B_path = os.path.join(label_B_dir, it)
#
#             #print(img_B_path)
#             imgs_list_A.append(img_A_path)
#             imgs_list_B.append(img_B_path)
#
#             # label_A = io.imread(label_A_path)
#             # label_B = io.imread(label_B_path)
#             label_A=cv2.imread(label_A_path)
#             label_B=cv2.imread(label_B_path)
#             labels_A.append(label_A)
#             labels_B.append(label_B)
#         if not idx%500: print('%d/%d images loaded.'%(idx, len(data_list)))
#         if idx>10: break
#
#     print(labels_A[0].shape)
#     print(str(len(imgs_list_A)) + ' ' + mode + ' images' + ' loaded.')
#
#     return imgs_list_A, imgs_list_B, labels_A, labels_B
def rand_TemporalFlip(im1,im2,label1,label2,prob):
    randomseed=random.random()
    if randomseed>prob:
        n_im1,n_im2,n_label1,n_label2=im2,im1,label2,label1
        return n_im1,n_im2,n_label1,n_label2
    else:
        return im1,im2,label1,label2


class Data(data.Dataset):
    def __init__(self, mode, random_flip =False):
        self.random_flip = random_flip

        self.imgs_A=os.path.join(root,"im1")
        self.imgs_B=os.path.join(root,"im2")
        self.labels_A=os.path.join(root,"label1_rgb")
        self.labels_B =os.path.join(root,"label2_rgb")
        self._list_images = _read_images_list(mode,root)

    def get_mask_name(self, idx):
        mask_name = os.path.split(self._list_images[idx])[-1]
        return mask_name


    def __getitem__(self, idx):
        labelname=self._list_images[idx].rstrip('\n')
        imgname = os.path.splitext(labelname)[0]+'.jpg'
        path=os.path.join(self.imgs_A, imgname)
        img_A = cv2.cvtColor(cv2.imread(os.path.join(self.imgs_A, imgname)),cv2.COLOR_BGR2RGB).astype(float)
        #img_A = cv2.cvtColor(cv2.imread(path),cv2.COLOR_BGR2RGB).astype(float)
        #img_A=cv2.imread(os.path.join(self.imgs_A, imgname)).astype(float)
        img_A = normalize_image(img_A, 'A')
        img_B = cv2.cvtColor(cv2.imread(os.path.join(self.imgs_B, imgname)),cv2.COLOR_BGR2RGB).astype(float)
        img_B = normalize_image(img_B, 'B')
        label_A = cv2.imread(os.path.join(self.labels_A, labelname),0)
        label_A=label_process(label_A)
        label_B = cv2.imread(os.path.join(self.labels_B, labelname),0)
        label_B=label_process(label_B)
        if self.random_flip:
             img_A, img_B, label_A, label_B = transform.rand_rot90_flip_SCD(img_A, img_B, label_A, label_B)
             img_A, img_B, label_A, label_B = rand_TemporalFlip(img_A, img_B, label_A, label_B,0.4)

        return F.to_tensor(img_A), F.to_tensor(img_B), torch.from_numpy(label_A), torch.from_numpy(label_B)

    def __len__(self):
        return len(self._list_images)

#with aug
# class Data(data.Dataset):
#     def __init__(self, mode, random_flip = False):
#         self.random_flip = random_flip
#         self.mode=mode
#         self.imgs_A=os.path.join(root,"im1")
#         self.imgs_B=os.path.join(root,"im2")
#         self.labels_A=os.path.join(root,"label1")
#         self.labels_B =os.path.join(root,"label2")
#         self._list_images = _read_images_list(mode,root)
#         self.image_transforms = A.Compose([A.OneOf([A.MotionBlur(p=1),
#                                         A.GaussianBlur(blur_limit=3, p=1),
#                                         A.Blur(blur_limit=3, p=1)])],additional_targets={'image2': 'image'})
#         self.label_transforms = A.Compose([
#             A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=1),
#             #A.Rotate(limit=90, p=1),
#             A.HorizontalFlip(p=0.5),
#             A.VerticalFlip(p=0.5),
#             A.RandomSizedCrop(min_max_height=[256,512],height=512,width=512,w2h_ratio=1.0,interpolation=1,always_apply=False,p=0.5)
#         ], additional_targets={'image2': 'image', 'mask2': "mask"})
#
#
#     def get_mask_name(self, idx):
#         mask_name = os.path.split(self._list_images[idx])[-1]
#         return mask_name
#
#     def __getitem__(self, idx):
#         labelname=self._list_images[idx].rstrip('\n')
#         imgname=os.path.splitext(labelname)[0]+".png"
#         img_A = cv2.cvtColor(cv2.imread(os.path.join(self.imgs_A, imgname)),cv2.COLOR_BGR2RGB).astype(float)
#         img_A = normalize_image(img_A, 'A')
#         img_B = cv2.cvtColor(cv2.imread(os.path.join(self.imgs_B, imgname)),cv2.COLOR_BGR2RGB).astype(float)
#         img_B = normalize_image(img_B, 'B')
#         label_A = cv2.imread(os.path.join(self.labels_A, labelname),0)
#         label_A=label_process(label_A)
#         label_B = cv2.imread(os.path.join(self.labels_B, labelname),0)
#         label_B=label_process(label_B)
#         if self.mode=="train":
#         #if self.random_flip:
#             augmented1 = self.image_transforms(image=img_A, image2=img_B)
#             augmented1_image1 = augmented1['image']
#             augmented1_image2 = augmented1['image2']
#             augmented2 = self.label_transforms(image=augmented1_image1, image2=augmented1_image2, mask=label_A ,mask2=label_B)
#             # augmented_image1 = augmented2['image']
#             # augmented_image2 = augmented2['image2']
#             # augmented_label1 = augmented2['mask']
#             # augmented_label2 = augmented2['mask2']
#             img_A = augmented2['image']
#             img_B = augmented2['image2']
#             label_A = augmented2['mask']
#             label_B = augmented2['mask2']
#             # 显示原始图像和增强后的图像进行对比
#             # fig, axes = plt.subplots(2, 2, figsize=(10, 10))
#             # axes[0, 0].imshow(img_A)
#             # axes[0, 0].set_title('orig 1')
#             #
#             # axes[0, 1].imshow(augmented_image1)
#             # axes[0, 1].set_title('aug 1')
#             #
#             # axes[1, 0].imshow(img_B)
#             # axes[1, 0].set_title('orig 2')
#             #
#             # axes[1, 1].imshow(augmented_image2)
#             # axes[1, 1].set_title('aug 2')
#             #
#             # plt.tight_layout()
#             # plt.show()
#             #
#             # # 显示原始标签和增强后的标签进行对比
#             # fig, axes = plt.subplots(2, 2, figsize=(10, 10))
#             # axes[0, 0].imshow(label_A, cmap='gray')
#             # axes[0, 0].set_title('orig 1')
#             #
#             # axes[0, 1].imshow(augmented_label1, cmap='gray')
#             # axes[0, 1].set_title('aug 1')
#             #
#             # axes[1, 0].imshow(label_B, cmap='gray')
#             # axes[1, 0].set_title('orig 2')
#             #
#             # axes[1, 1].imshow(augmented_label2, cmap='gray')
#             # axes[1, 1].set_title('aug 2')
#             #
#             # plt.tight_layout()
#             # plt.show()
#             #img_A, img_B, label_A, label_B = transform.rand_rot90_flip_SCD(img_A, img_B, label_A, label_B)
#         return F.to_tensor(img_A), F.to_tensor(img_B), torch.from_numpy(label_A), torch.from_numpy(label_B)
#
#
#     def __len__(self):
#         return len(self._list_images)

class Data_test(data.Dataset):
    def __init__(self, mode,test_dir):
        self.imgs_A = os.path.join(test_dir, 'im1')
        self.imgs_B= os.path.join(test_dir, 'im2')
        self._list_images = _read_images_list(mode, test_dir)
        #self.mask_name_list = []

    def get_mask_name(self, idx):
        return self.mask_name_list[idx]

    def __getitem__(self, idx):
        labelname = self._list_images[idx].rstrip('\n')
        imgname=os.path.splitext(labelname)[0]+".png"
        img_A = cv2.cvtColor(cv2.imread(os.path.join(self.imgs_A, imgname)),cv2.COLOR_BGR2RGB).astype(float)
        img_A = normalize_image(img_A, 'A')
        img_B = cv2.cvtColor(cv2.imread(os.path.join(self.imgs_B, imgname)),cv2.COLOR_BGR2RGB).astype(float)
        img_B = normalize_image(img_B, 'B')
        return F.to_tensor(img_A), F.to_tensor(img_B),labelname

    def __len__(self):
        return len(self._list_images)

