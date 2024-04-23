import os
import sys
import time
import argparse
import warnings

import cv2
import numpy as np
import torch.autograd

from matplotlib import pyplot as plt
from skimage import io, exposure
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

#################################
# from datasets import RS_ST as RS
#from datasets.my_data import my_datasets as MD
from datasets.SECOND import Second as MD
# from models.BiSRNet import BiSRNet as Net
# from models.SCanNet import SCanNet as Net
#from models.SCanNet_hrnet import SCanNet as Net
#from models.BiSRNet import BiSRNet as Net
#from models.ECGv1 import ECGNet_NO2 as Net
#from models.Daudt.HRSCD4 import HRSCD4 as Net
#from models.SCDNet import SCDNet as Net
#from models.ECGv1 import ECGNet_dp as Net
#from models.TBFFNet import TBFFNet as Net
from models.CG_SPNet import CG_SPNet as Net

DATA_NAME = r'WZ'
#################################
# DATA_DIR="chongzhou_test"
Net_NAME = r"/CG_SPNet"
DATA_DIR = r"SD"
Test_BS=sys.argv[1:]
working_path = os.path.dirname(os.path.abspath(__file__))
NET_NAME = 'CG_SPNet'
DATA_NAME = 'SD'
args = {
    'batch_size': 1,
    'gpu': True,
    'pred_dir': Test_BS[2],
    'chkpt_dir': Test_BS[0],
    'data_dir': Test_BS[1]
}



def compare_models(model_1, model_2):
    models_differ = 0
    for key_item_1, key_item_2 in zip(model_1.state_dict().items(), model_2.state_dict().items()):
        if torch.equal(key_item_1[1], key_item_2[1]):
            pass
        else:
            models_differ += 1
            if (key_item_1[0] == key_item_2[0]):
                print('Mismtach found at', key_item_1[0])
            else:
                raise Exception
    if models_differ == 0:
        print('Models match perfectly! :)')


def main():
    begin_time = time.time()
    opt = PredOptions().parse()
    net = Net(3,MD.num_classes).cuda()
    #net = Net(False).cuda()
    net.load_state_dict(torch.load(opt.chkpt_path),True)
    print("model loaded")
    sys.stdout.flush()
    net.eval()

    test_set = MD.Data_test("full", opt.test_dir)
    test_loader = DataLoader(test_set, batch_size=opt.pred_batch_size)
    predict(net, test_set, test_loader, opt.pred_dir, flip=False, index_map=True, intermediate=False)
    # predict_direct(net, test_set, test_loader, opt.pred_dir, flip=False, index_map=True)
    time_use = time.time() - begin_time
    print('Total time: %.2fs' % time_use)


# For models with 3 outputs: 1 change map + 2 semantic maps.
# Parameters: flip->test time augmentation     index_map->"False" means rgb results      intermediate->whether to outputs the intermediate maps
def predict(net, pred_set, pred_loader, pred_dir, flip=False, index_map=False, intermediate=False):
    all_iters=pred_loader.__len__();

    warnings.filterwarnings("ignore", category=UserWarning)

    pred_A_dir_rgb = os.path.join(pred_dir, 'im1_rgb')
    pred_B_dir_rgb = os.path.join(pred_dir, 'im2_rgb')
    if not os.path.exists(pred_A_dir_rgb): os.makedirs(pred_A_dir_rgb)
    if not os.path.exists(pred_B_dir_rgb): os.makedirs(pred_B_dir_rgb)
    if index_map:
        pred_A_dir = os.path.join(pred_dir, 'im1')
        pred_B_dir = os.path.join(pred_dir, 'im2')
        if not os.path.exists(pred_A_dir): os.makedirs(pred_A_dir)
        if not os.path.exists(pred_B_dir): os.makedirs(pred_B_dir)
    if intermediate:
        pred_mA_dir = os.path.join(pred_dir, 'im1_semantic')
        pred_mB_dir = os.path.join(pred_dir, 'im2_semantic')
        pred_change_dir = os.path.join(pred_dir, 'change')
        if not os.path.exists(pred_mA_dir): os.makedirs(pred_mA_dir)
        if not os.path.exists(pred_mB_dir): os.makedirs(pred_mB_dir)
        if not os.path.exists(pred_change_dir): os.makedirs(pred_change_dir)
    tbar = tqdm(pred_loader)
    print("\n===========================================================================================")
    print("----------------------------------------pred  processing-------------------------------------")
    print("===========================================================================================\n")
    for vi, data in enumerate(tbar):
        imgs_A, imgs_B, mask_name = data
        # imgs = torch.cat([imgs_A, imgs_B], 1)
        imgs_A = imgs_A.cuda().float()
        imgs_B = imgs_B.cuda().float()
        with torch.no_grad():
            #outputs_A, outputs_B = net(torch.cat([imgs_A,imgs_B],dim=1))  # ,aux
            out_change, outputs_A, outputs_B = net(torch.cat([imgs_A,imgs_B],dim=1))  # ,aux
            out_change = F.sigmoid(out_change)
            # vision_change = np.array((torch.squeeze(out_change)).cpu())
            # label_flat = np.concatenate([arr.flatten() for arr in vision_change]).reshape(-1, 1)
            # reducer = umap.UMAP(n_components=2)
            # embedding = reducer.fit_transform(label_flat)
            #
            # # 绘制可视化图形
            # plt.scatter(embedding[:, 0], embedding[:, 1], c=vision_change, cmap='viridis')
            # plt.colorbar()
            # plt.show()
            # plt.savefig('example.png')
        if flip:
            outputs_A = F.softmax(outputs_A, dim=1)
            outputs_B = F.softmax(outputs_B, dim=1)

            imgs_A_v = torch.flip(imgs_A, [2])
            imgs_B_v = torch.flip(imgs_B, [2])
            out_change_v, outputs_A_v, outputs_B_v = net(imgs_A_v, imgs_B_v)
            outputs_A_v = torch.flip(outputs_A_v, [2])
            outputs_B_v = torch.flip(outputs_B_v, [2])
            out_change_v = torch.flip(out_change_v, [2])
            outputs_A += F.softmax(outputs_A_v, dim=1)
            outputs_B += F.softmax(outputs_B_v, dim=1)
            out_change += F.sigmoid(out_change_v)

            imgs_A_h = torch.flip(imgs_A, [3])
            imgs_B_h = torch.flip(imgs_B, [3])
            out_change_h, outputs_A_h, outputs_B_h = net(imgs_A_h, imgs_B_h)
            outputs_A_h = torch.flip(outputs_A_h, [3])
            outputs_B_h = torch.flip(outputs_B_h, [3])
            out_change_h = torch.flip(out_change_h, [3])
            outputs_A += F.softmax(outputs_A_h, dim=1)
            outputs_B += F.softmax(outputs_B_h, dim=1)
            out_change += F.sigmoid(out_change_h)

            imgs_A_hv = torch.flip(imgs_A, [2, 3])
            imgs_B_hv = torch.flip(imgs_B, [2, 3])
            out_change_hv, outputs_A_hv, outputs_B_hv = net(imgs_A_hv, imgs_B_hv)
            outputs_A_hv = torch.flip(outputs_A_hv, [2, 3])
            outputs_B_hv = torch.flip(outputs_B_hv, [2, 3])
            out_change_hv = torch.flip(out_change_hv, [2, 3])
            outputs_A += F.softmax(outputs_A_hv, dim=1)
            outputs_B += F.softmax(outputs_B_hv, dim=1)
            out_change += F.sigmoid(out_change_hv)
            out_change = out_change / 4

        outputs_A = outputs_A.cpu().detach()
        outputs_B = outputs_B.cpu().detach()
        change_mask = out_change.cpu().detach() > 0.5
        change_mask = change_mask.squeeze()
        pred_A = torch.argmax(outputs_A, dim=1).squeeze()
        pred_B = torch.argmax(outputs_B, dim=1).squeeze()

        if intermediate:
            pred_A_path = os.path.join(pred_mA_dir, mask_name)
            pred_B_path = os.path.join(pred_mB_dir, mask_name)
            pred_change_path = os.path.join(pred_change_dir, mask_name)
            io.imsave(pred_A_path, MD.Index2Color(pred_A.numpy()))
            io.imsave(pred_B_path, MD.Index2Color(pred_B.numpy()))
            change_map = exposure.rescale_intensity(change_mask.numpy(), 'image', 'dtype')
            io.imsave(pred_change_path, change_map)
        pred_A = (pred_A * change_mask.long()).numpy()
        pred_B = (pred_B * change_mask.long()).numpy()
        pred_A_path = os.path.join(pred_A_dir_rgb, mask_name[0])
        pred_B_path = os.path.join(pred_B_dir_rgb, mask_name[0])
        io.imsave(pred_A_path, MD.Index2Color(pred_A))
        io.imsave(pred_B_path, MD.Index2Color(pred_B))
        tqdm.set_description(tbar,pred_A_dir)
        if index_map:
            pred_A_path = os.path.join(pred_A_dir, mask_name[0])
            pred_B_path = os.path.join(pred_B_dir, mask_name[0])
            io.imsave(pred_A_path, pred_A.astype(np.uint8))
            io.imsave(pred_B_path, pred_B.astype(np.uint8))
        per=round(100*vi/all_iters,2)
        print(f"test_percentage:{per}%")
        sys.stdout.flush()

        # For models that directly produce 2 SCD maps.


# Parameters: flip->test time augmentation     index_map->"False" means rgb results
def predict_direct(net, pred_set, pred_loader, pred_dir, flip=False, index_map=False, ):
    pred_A_dir_rgb = os.path.join(pred_dir, 'im1_rgb')
    pred_B_dir_rgb = os.path.join(pred_dir, 'im2_rgb')
    if not os.path.exists(pred_A_dir_rgb): os.makedirs(pred_A_dir_rgb)
    if not os.path.exists(pred_B_dir_rgb): os.makedirs(pred_B_dir_rgb)
    if index_map:
        pred_A_dir = os.path.join(pred_dir, 'im1')
        pred_B_dir = os.path.join(pred_dir, 'im2')
        if not os.path.exists(pred_A_dir): os.makedirs(pred_A_dir)
        if not os.path.exists(pred_B_dir): os.makedirs(pred_B_dir)

    for vi, data in enumerate(pred_loader):
        imgs_A, imgs_B = data
        # imgs = torch.cat([imgs_A, imgs_B], 1)
        imgs_A = imgs_A.cuda().float()
        imgs_B = imgs_B.cuda().float()
        mask_name = pred_set.get_mask_name(vi)
        with torch.no_grad():
            outputs_A, outputs_B = net(imgs_A, imgs_B)  # ,aux
        if flip:
            outputs_A = F.softmax(outputs_A, dim=1)
            outputs_B = F.softmax(outputs_B, dim=1)

            imgs_A_v = torch.flip(imgs_A, [2])
            imgs_B_v = torch.flip(imgs_B, [2])
            outputs_A_v, outputs_B_v = net(imgs_A_v, imgs_B_v)
            outputs_A_v = torch.flip(outputs_A_v, [2])
            outputs_B_v = torch.flip(outputs_B_v, [2])
            outputs_A += F.softmax(outputs_A_v, dim=1)
            outputs_B += F.softmax(outputs_B_v, dim=1)

            imgs_A_h = torch.flip(imgs_A, [3])
            imgs_B_h = torch.flip(imgs_B, [3])
            outputs_A_h, outputs_B_h = net(imgs_A_h, imgs_B_h)
            outputs_A_h = torch.flip(outputs_A_h, [3])
            outputs_B_h = torch.flip(outputs_B_h, [3])
            outputs_A += F.softmax(outputs_A_h, dim=1)
            outputs_B += F.softmax(outputs_B_h, dim=1)

            imgs_A_hv = torch.flip(imgs_A, [2, 3])
            imgs_B_hv = torch.flip(imgs_B, [2, 3])
            outputs_A_hv, outputs_B_hv = net(imgs_A_hv, imgs_B_hv)
            outputs_A_hv = torch.flip(outputs_A_hv, [2, 3])
            outputs_B_hv = torch.flip(outputs_B_hv, [2, 3])
            outputs_A += F.softmax(outputs_A_hv, dim=1)
            outputs_B += F.softmax(outputs_B_hv, dim=1)

        outputs_A = outputs_A.cpu().detach()
        outputs_B = outputs_B.cpu().detach()
        pred_A = torch.argmax(outputs_A, dim=1)
        pred_B = torch.argmax(outputs_B, dim=1)
        pred_A = pred_A.squeeze().numpy().astype(np.uint8)
        pred_B = pred_B.squeeze().numpy().astype(np.uint8)

        pred_A_path = os.path.join(pred_A_dir_rgb, mask_name)
        pred_B_path = os.path.join(pred_B_dir_rgb, mask_name)
        io.imsave(pred_A_path, MD.Index2Color(pred_A))
        io.imsave(pred_B_path, MD.Index2Color(pred_B))
        print(pred_A_path)
        if index_map:
            pred_A_path = os.path.join(pred_A_dir, mask_name)
            pred_B_path = os.path.join(pred_B_dir, mask_name)
            io.imsave(pred_A_path, pred_A.astype(np.uint8))
            io.imsave(pred_B_path, pred_B.astype(np.uint8))
        '''
        change_path = os.path.join(pred_dir, 'change', mask_name)
        io.imsave(change_path, (change_mask*255).astype(np.uint8))'''


if __name__ == '__main__':
    #main()
    begin_time = time.time()
    #opt = PredOptions().parse()
    net = Net(3,MD.num_classes).cuda()
    #net = Net(False).cuda()
    net.load_state_dict(torch.load(args["chkpt_dir"]),True)
    print("model loaded")
    sys.stdout.flush()
    net.eval()

    test_set = MD.Data_test("full", args["data_dir"])
    test_loader = DataLoader(test_set, batch_size=args["batch_size"])
    predict(net, test_set, test_loader, args["pred_dir"], flip=False, index_map=True, intermediate=False)
    # predict_direct(net, test_set, test_loader, opt.pred_dir, flip=False, index_map=True)
    time_use = time.time() - begin_time
    print('Total time: %.2fs' % time_use)
