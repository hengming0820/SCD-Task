# -*- coding:utf-8 -*-
import seaborn as sns

import torch.autograd

import torch.nn.functional as F
from matplotlib import pyplot as plt
import datetime
from torch.utils.data import DataLoader

from utils.loss import CrossEntropyLoss2d, weighted_BCE_logits, ChangeSimilarity
from utils.utils import accuracy, SCDD_eval_all, AverageMeter, get_hist
import torch
import numpy as np
import os
from tqdm import tqdm
from datasets.my_data import my_datasets as SD
#from datasets.SECOND import Second as SD
#from datasets.LandsatSCD import landsat as SD
#from datasets.my_data import my_datasets as SD

# from datasets.Hi_UCD import Hi_UCD as SD
import time
from models.SSCDl import SSCDl_HR as Net
#from models.CG_SPNet import CG_SPNet as Net
import sys
Test_BS=sys.argv[1:]
working_path = os.path.dirname(os.path.abspath(__file__))
NET_NAME = 'SSCDl_HR'
DATA_NAME = 'MD'

args = {
    'train_batch_size': 1,
    'val_batch_size': 4,
    'lr': 0.0003,
    'epochs': 50,
    'gpu': True,
    'psd_TTA': True,
    'pred_dir': os.path.join(working_path, 'results', DATA_NAME),
    'eval_dir': os.path.join(working_path, 'confusion_matrix', DATA_NAME,NET_NAME),
    'chkpt_dir': os.path.join(working_path, 'checkpoints', DATA_NAME),
    'log_dir': os.path.join(working_path, 'logs', DATA_NAME, NET_NAME),
    'load_path': os.path.join(working_path, 'checkpoints', DATA_NAME,
                              'SSCDl_HR_20e_mIoU67.20_Sek19.64_Fscd56.74_OA79.12.pth')
}

if not os.path.exists(args['log_dir']): os.makedirs(args['log_dir'])
if not os.path.exists(args['pred_dir']): os.makedirs(args['pred_dir'])
if not os.path.exists(args['chkpt_dir']): os.makedirs(args['chkpt_dir'])


def hist_lcm(preds, labels, num_class):
    hist = np.zeros((num_class, num_class))
    for pred, label in zip(preds, labels):
        infer_array = np.array(pred)
        unique_set = set(np.unique(infer_array))
        assert unique_set.issubset(set([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])), "unrecognized label number"
        label_array = np.array(label)
        assert infer_array.shape == label_array.shape, "The size of prediction and target must be the same"
        hist += get_hist(infer_array, label_array, num_class)
    hist_all = hist
    return hist_all


def plot_confusion_matrix(confusion_matrix, class_names, save_path):
    plt.figure(figsize=(15, 15))
    sns.set(font_scale=2.5)  # 设置字体大小

    # 计算准确率的混淆矩阵
    total_samples = np.sum(confusion_matrix, axis=1)  # 每个类别的总样本数
    accuracy_matrix = confusion_matrix / total_samples[:, np.newaxis]  # 每个元素除以总样本数
    color=""
    # 绘制热力图
    if DATA_NAME=="SD":
        color="Oranges"
    elif DATA_NAME=="MD":
        color="Blues"
    ax = sns.heatmap(accuracy_matrix, annot=True, fmt=".2f", cmap=color,cbar=False,
                     xticklabels=class_names, yticklabels=class_names, square=True)
    # 获取混淆矩阵的高度
    box = ax.get_position()
    matrix_height = box.height

    # 调整颜色条的位置和大小
    cbar_ax = plt.gcf().add_axes([box.x1 + 0.01, box.y0, 0.03, matrix_height])
    cbar = plt.colorbar(ax.collections[0], cax=cbar_ax)
    cbar.ax.tick_params(labelsize=30)  # 设置颜色条刻度标签的字体大小
    # plt.xlabel('Predicted Labels')
    # plt.ylabel('True Labels')
    # plt.title('Accuracy Matrix')
    # # 设置xticklabels保持水平
    # plt.xticks(rotation=30)
    # plt.yticks(rotation=0)

    ax.set_xlabel('Predicted Labels', fontsize=25)  # 设置x轴标签
    ax.set_ylabel('True Labels', fontsize=25)  # 设置y轴标签
    #ax.set_title('Accuracy Matrix', fontsize=30)  # 设置标题
    ax.tick_params(axis='y', labelsize=25, rotation=30)
    ax.tick_params(axis='x', labelsize=25)
    plt.subplots_adjust(left=0.15, right=0.9, bottom=0.1, top=0.9)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def calc_conf(softmap):
    b, c, h, w = softmap.size()
    conf, index = torch.max(softmap, dim=1)
    index_onehot = F.one_hot(index.long(), num_classes=SD.num_classes).permute((0, 3, 1, 2))
    masked_softmap = index_onehot * softmap
    threds = np.zeros(c)
    for idx in range(c):
        masked_softmap_i = torch.flatten(masked_softmap[:, idx])
        masked_softmap_i = masked_softmap_i[masked_softmap_i.nonzero()]
        len = masked_softmap_i.size(0)
        if len:
            masked_softmap_i, _ = torch.sort(masked_softmap_i, descending=True)
            mid_val = masked_softmap_i[len // 2]
            threds[idx] = mid_val.cpu().detach().numpy()  # *args['pseudo_thred']
        else:
            threds[idx] = 0.5
    threds[threds > 0.9] = 0.9
    threds = torch.from_numpy(threds).unsqueeze(1).unsqueeze(2).cuda()
    thred_onehot = index_onehot * threds
    thredmap, _ = torch.max(thred_onehot, dim=1)
    conf = torch.ge(conf, thredmap)
    return conf, index


def Eval(val_loader, net, criterion):
    # the following code is written assuming that batch size is 1
    net.eval()
    torch.cuda.empty_cache()
    start = time.time()

    val_loss = AverageMeter()
    acc_meter = AverageMeter()

    preds_all = []
    predsA_all = []
    predsB_all = []
    labels_all = []
    labelsA_all = []
    labelsB_all = []
    all_iters=val_loader.__len__();
    for vi, data in tqdm(enumerate(val_loader), total=val_loader.__len__(), colour="white", ncols=80,
                         desc="-----val  processing-----"):
        imgs_A, imgs_B, labels_A, labels_B = data
        if args['gpu']:
            imgs_A = imgs_A.cuda().float()
            imgs_B = imgs_B.cuda().float()
            labels_A = labels_A.cuda().long()
            labels_B = labels_B.cuda().long()

        with torch.no_grad():
            input = torch.cat([imgs_A, imgs_B], dim=1)
            # input=torch.cat([imgs_B,imgs_A],dim=1)
            out_change, outputs_A, outputs_B = net(input)
            #pred1, pred2, pred3, outputs_A,outputs_B = net(torch.cat([imgs_A, imgs_B],dim=1))
            loss_A = criterion(outputs_A, labels_A)
            loss_B = criterion(outputs_B, labels_B)
            loss = loss_A * 0.5 + loss_B * 0.5
        val_loss.update(loss.cpu().detach().numpy())

        labels_A = labels_A.cpu().detach().numpy()
        labels_B = labels_B.cpu().detach().numpy()
        outputs_A = outputs_A.cpu().detach()
        outputs_B = outputs_B.cpu().detach()
        change_mask = F.sigmoid(out_change).cpu().detach() > 0.5
        preds_A = torch.argmax(outputs_A, dim=1)
        preds_B = torch.argmax(outputs_B, dim=1)

        # preds_A = torch.argmax(outputs_B, dim=1)#flip
        # preds_B = torch.argmax(outputs_A, dim=1)#flip

        # preds_A = (preds_A * change_mask.squeeze().long()).numpy()
        # preds_B = (preds_B * change_mask.squeeze().long()).numpy()
        preds_A = preds_A * change_mask.squeeze().long()
        preds_B = preds_B * change_mask.squeeze().long()
        # SCD
        # preds_A = (preds_A .squeeze().long()).numpy()
        # preds_B = (preds_B .squeeze().long()).numpy()
        # preds_A = (preds_A .squeeze().long())
        # preds_B = (preds_B .squeeze().long())
        # if preds_A.shape!=labels_A.shape:
        #     preds_A=torch.unsqueeze(preds_A,dim=0)
        #     preds_B=torch.unsqueeze(preds_B,dim=0)
        preds_A=np.array(preds_A)
        preds_B=np.array(preds_B)
        for (pred_A, pred_B, label_A, label_B) in zip(preds_A, preds_B, labels_A, labels_B):
            acc_A, valid_sum_A = accuracy(pred_A, label_A)
            acc_B, valid_sum_B = accuracy(pred_B, label_B)
            preds_all.append(pred_A)
            preds_all.append(pred_B)

            predsA_all.append(pred_A)
            predsB_all.append(pred_B)

            labels_all.append(label_A)
            labels_all.append(label_B)
            labelsA_all.append(label_A)
            labelsB_all.append(label_B)
            acc = (acc_A + acc_B) * 0.5
            acc_meter.update(acc)
        per=round(100*vi/all_iters,2)
        print(f"test_percentage:{per}%")
        sys.stdout.flush()
        # if curr_epoch % args['predict_step'] == 0 and vi == 0:
        #     pred_A_color = SD.Index2Color(preds_A[0])
        #     pred_B_color = SD.Index2Color(preds_B[0])
        #     #io.imsave(os.path.join(args['pred_dir'], NET_NAME + '_A.png'), pred_A_color)
        #     cv2.imwrite(os.path.join(args['pred_dir'], NET_NAME + '_A.png'), pred_A_color)
        #    # io.imsave(os.path.join(args['pred_dir'], NET_NAME + '_B.png'), pred_B_color)
        #     cv2.imwrite(os.path.join(args['pred_dir'], NET_NAME + '_B.png'), pred_B_color)
        #     print('Prediction saved!')

    hist_A = hist_lcm(predsA_all, labelsA_all, SD.num_classes)
    hist_B = hist_lcm(predsB_all, labelsB_all, SD.num_classes)

    Fscd, IoU_mean, Sek, iu1, iu2, kappa, class_iou, hist = SCDD_eval_all(preds_all, labels_all, SD.num_classes)
    Fscd_A, IoU_mean_A, Sek_A, iu1_A, iu2_A, kappa_A, class_iou_A, hist_A = SCDD_eval_all(predsA_all, labelsA_all,
                                                                                          SD.num_classes)
    Fscd_B, IoU_mean_B, Sek_B, iu1_B, iu2_B, kappa_B, class_iou_B, hist_B = SCDD_eval_all(predsB_all, labelsB_all,
                                                                                          SD.num_classes)

    # class_names = ['no_change', 'water', 'N.vg.surface', 'low vegetation', 'tree', 'building', 'playground']
    #class_names = ['no_change', 'water', 'ground', 'low veg', 'tree', 'building', 'playground']
    if DATA_NAME=="SD":
        class_names = ['no_change', 'water', 'ground', 'low veg', 'tree', 'building', 'playground']
    elif DATA_NAME=="MD":
        class_names = ['unchanged', 'road', 'buillding', 'water', 'farm', 'vege', 'background']

    curr_time = time.time() - start
    print('%.1fs Val loss: %.2f Fscd: %.2f IoU: %.2f Sek: %.2f Accuracy: %.2f' \
          % (curr_time, val_loss.average(), Fscd * 100, IoU_mean * 100, Sek * 100, acc_meter.average() * 100))
    print("iu1: %.2f; iu2: %.2f; kappa: %.2f" % (iu1 * 100, iu2 * 100, kappa * 100))
    print(
        "no_change: %.2f; water: %.2f; N.vg.surface: %.2f; low vegetation: %.2f; tree: %.2f; building: %.2f; playground: %.2f" %
        (class_iou[0] * 100, class_iou[1] * 100, class_iou[2] * 100, class_iou[3] * 100, class_iou[4] * 100,
         class_iou[5] * 100, class_iou[6] * 100,))
    print("==================A==================")
    print('%.1fs Val loss: %.2f Fscd: %.2f IoU: %.2f Sek: %.2f Accuracy: %.2f' \
          % (curr_time, val_loss.average(), Fscd_A * 100, IoU_mean_A * 100, Sek_A * 100, acc_meter.average() * 100))
    print("iu1: %.2f; iu2: %.2f; kappa: %.2f" % (iu1_A * 100, iu2_A * 100, kappa_A * 100))
    print(
        "no_change: %.2f; water: %.2f; N.vg.surface: %.2f; low vegetation: %.2f; tree: %.2f; building: %.2f; playground: %.2f" %
        (class_iou_A[0] * 100, class_iou_A[1] * 100, class_iou_A[2] * 100, class_iou_A[3] * 100, class_iou_A[4] * 100,
         class_iou_A[5] * 100, class_iou_A[6] * 100,))

    print("==================B=================")
    print('%.1fs Val loss: %.2f Fscd: %.2f IoU: %.2f Sek: %.2f Accuracy: %.2f' \
          % (curr_time, val_loss.average(), Fscd_B * 100, IoU_mean_B * 100, Sek_B * 100, acc_meter.average() * 100))
    print("iu1: %.2f; iu2: %.2f; kappa: %.2f" % (iu1_B * 100, iu2_B * 100, kappa_B * 100))
    print(
        "no_change: %.2f; water: %.2f; N.vg.surface: %.2f; low vegetation: %.2f; tree: %.2f; building: %.2f; playground: %.2f" %
        (class_iou_B[0] * 100, class_iou_B[1] * 100, class_iou_B[2] * 100, class_iou_B[3] * 100, class_iou_B[4] * 100,
         class_iou_B[5] * 100, class_iou_B[6] * 100,))
    # print(class_iou)

    if not os.path.exists(args['eval_dir']): os.makedirs(args['eval_dir'])
    save_path1 = os.path.join(args['eval_dir'], "confusion_matrixA.png")
    save_path2 = os.path.join(args['eval_dir'], "confusion_matrixB.png")
    save_path3 = os.path.join(args['eval_dir'], "confusion_matrixALL.png")
    plot_confusion_matrix(hist_A, class_names, save_path1)
    plot_confusion_matrix(hist_B, class_names, save_path2)
    plot_confusion_matrix(hist, class_names, save_path3)

    return Fscd, IoU_mean, Sek, acc_meter.avg, val_loss.avg


if __name__ == '__main__':
    num_class = 7
    val_set = SD.Data('val')
    val_loader = DataLoader(val_set, batch_size=args['val_batch_size'], num_workers=0, shuffle=False)
    net = Net(3, num_classes=SD.num_classes).cuda()
    #net = Net(use_DS=True).cuda()
    net.load_state_dict(torch.load(args['load_path']), strict=True)
    print("model loaded")
    sys.stdout.flush()
    criterion = CrossEntropyLoss2d(ignore_index=0).cuda()
    print(datetime.datetime.now())
    time_start = time.time()
    Fscd_v, mIoU_v, Sek_v, acc_v, loss_v = Eval(val_loader, net, criterion)
    time_end = time.time()
    times = time_end - time_start
    print('Best rec: Val Fscd %.2f acc %.2f loss %.2f MIOU %.2f sek: %.2f' % (
        Fscd_v * 100, acc_v * 100, loss_v, mIoU_v * 100, Sek_v) * 100)
    print("Pred times: {} seconds".format(times))
    print("Per image Pred times: {} s".format(times / 2968))