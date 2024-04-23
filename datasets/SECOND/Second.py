import os
import albumentations as A
import cv2
import numpy as np
import torch
from skimage import io
from torch.utils import data
import utils.transform as transform
import matplotlib.pyplot as plt
from torchvision.transforms import functional as F
import torchvision.transforms as transforms



num_classes = 7
ST_COLORMAP = [[255,255,255], [0,0,255], [128,128,128], [0,128,0], [0,255,0], [128,0,0], [255,0,0]]
ST_CLASSES = ['unchanged', 'water', 'ground', 'low vegetation', 'tree', 'building', 'sports field']

MEAN_A = np.array([113.40, 114.08, 116.45])
STD_A  = np.array([48.30,  46.27,  48.14])
MEAN_B = np.array([111.07, 114.04, 118.18])
STD_B  = np.array([49.41,  47.01,  47.94])

root = r"D:\pycharm\data\socend"

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
    mask1 = (x_label ==29)#水
    mask2 = (x_label == 38)#建筑
    mask3 = (x_label == 75)#低矮植被
    mask4 = (x_label == 76)#操场
    mask5 = (x_label == 128)#地面
    mask6 = (x_label == 149)#树
    mask7 = (x_label == 255)#无变化
    prediction[mask7] = 0
    prediction[mask1] = 1
    prediction[mask5] = 2
    prediction[mask3] = 3
    prediction[mask6] = 4
    prediction[mask2] = 5
    prediction[mask4] = 6
    return prediction

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
#with aug
class Data(data.Dataset):
    def __init__(self, mode, random_flip = False):
        self.random_flip = random_flip
        self.mode=mode
        self.imgs_A=os.path.join(root,"im1")
        self.imgs_B=os.path.join(root,"im2")
        self.labels_A=os.path.join(root,"label1")
        self.labels_B =os.path.join(root,"label2")
        self._list_images = _read_images_list(mode,root)
        # self.image_transforms = A.Compose([A.OneOf([A.MotionBlur(p=1),
        #                                 A.GaussianBlur(blur_limit=3, p=1),
        #                                 A.Blur(blur_limit=3, p=1)])],additional_targets={'image2': 'image'})
        # self.label_transforms = A.Compose([
        #     A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=1),
        #     #A.Rotate(limit=90, p=1),
        #     A.HorizontalFlip(p=0.5),
        #     A.VerticalFlip(p=0.5),
        #     A.RandomSizedCrop(min_max_height=[256,512],height=512,width=512,w2h_ratio=1.0,interpolation=1,always_apply=False,p=0.5)
        # ], additional_targets={'image2': 'image', 'mask2': "mask"})


    def get_mask_name(self, idx):
        mask_name = os.path.split(self._list_images[idx])[-1]
        return mask_name

    def __getitem__(self, idx):
        labelname=self._list_images[idx].rstrip('\n')
        imgname=os.path.splitext(labelname)[0]+".png"
        img_A = io.imread(os.path.join(self.imgs_A, imgname))
        #img_A = cv2.cvtColor(cv2.imread(os.path.join(self.imgs_A, imgname)),cv2.COLOR_BGR2RGB).astype(float)
        img_A = normalize_image(img_A, 'A')
        #img_B = cv2.cvtColor(cv2.imread(os.path.join(self.imgs_B, imgname)),cv2.COLOR_BGR2RGB).astype(float)
        img_B = io.imread(os.path.join(self.imgs_B, imgname))
        img_B = normalize_image(img_B, 'B')
        label_A = cv2.imread(os.path.join(self.labels_A, labelname),0)
        #label_A=io.imread(self.labels_A)
        label_A=label_process(label_A)
        label_B = cv2.imread(os.path.join(self.labels_B, labelname),0)
        label_B=label_process(label_B)
        if self.mode=="train":
            img_A, img_B, label_A, label_B = transform.rand_rot90_flip_SCD(img_A, img_B, label_A, label_B)
            img_A, img_B, label_A, label_B = transform.rand_TemporalFlip(img_A, img_B, label_A, label_B,prob=0.5)
        return F.to_tensor(img_A), F.to_tensor(img_B), torch.from_numpy(label_A), torch.from_numpy(label_B)


    def __len__(self):
        return len(self._list_images)
class Data_5z(data.Dataset):
    def __init__(self, mode, random_flip = False):
        self.random_flip = random_flip
        self.mode=mode
        self.imgs_A=os.path.join(root,"im1")
        self.imgs_B=os.path.join(root,"im2")
        self.labels_A=os.path.join(root,"label1")
        self.labels_B =os.path.join(root,"label2")
        self._list_images = _read_images_list(mode,root)
        self.image_transforms = A.Compose([A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)]
                                          ,additional_targets={'image2': 'image'})
        self.label_transforms = A.Compose([
            A.Rotate(limit=90, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5)
        ], additional_targets={'image2': 'image', 'mask2': "mask"})


    def get_mask_name(self, idx):
        mask_name = os.path.split(self._list_images[idx])[-1]
        return mask_name

    def __getitem__(self, idx):
        labelname=self._list_images[idx].rstrip('\n')
        imgname=os.path.splitext(labelname)[0]+".png"
        img_A = io.imread(os.path.join(self.imgs_A, imgname))
        #img_A = cv2.cvtColor(cv2.imread(os.path.join(self.imgs_A, imgname)),cv2.COLOR_BGR2RGB).astype(float)

        #img_B = cv2.cvtColor(cv2.imread(os.path.join(self.imgs_B, imgname)),cv2.COLOR_BGR2RGB).astype(float)
        img_B = io.imread(os.path.join(self.imgs_B, imgname))

        label_A = cv2.imread(os.path.join(self.labels_A, labelname),0)
        #label_A=io.imread(self.labels_A)
        label_A=label_process(label_A)
        label_B = cv2.imread(os.path.join(self.labels_B, labelname),0)
        label_B=label_process(label_B)
        if self.mode=="train":
            augmented1 = self.image_transforms(image=img_A, image2=img_B)
            augmented1_image1 = augmented1['image']
            augmented1_image2 = augmented1['image2']
            augmented2 = self.label_transforms(image=augmented1_image1, image2=augmented1_image2, mask=label_A ,mask2=label_B)
            # augmented_image1 = augmented2['image']
            # augmented_image2 = augmented2['image2']
            # augmented_label1 = augmented2['mask']
            # augmented_label2 = augmented2['mask2']
            img_A = augmented2['image']
            img_B = augmented2['image2']
            label_A = augmented2['mask']
            label_B = augmented2['mask2']
            img_A = normalize_image(img_A, 'A')
            img_B = normalize_image(img_B, 'B')
        return F.to_tensor(img_A), F.to_tensor(img_B), torch.from_numpy(label_A), torch.from_numpy(label_B)


    def __len__(self):
        return len(self._list_images)

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
        img_A = io.imread(os.path.join(self.imgs_A, imgname))
        #img_A = cv2.cvtColor(cv2.imread(os.path.join(self.imgs_A, imgname)),cv2.COLOR_BGR2RGB).astype(float)
        img_A = normalize_image(img_A, 'A')
        img_B = io.imread(os.path.join(self.imgs_B, imgname))
        #img_B = cv2.cvtColor(cv2.imread(os.path.join(self.imgs_B, imgname)),cv2.COLOR_BGR2RGB).astype(float)
        img_B = normalize_image(img_B, 'B')
        return F.to_tensor(img_A), F.to_tensor(img_B),labelname

    def __len__(self):
        return len(self._list_images)

