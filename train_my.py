import os
import time
import copy
import random
import threading
import cv2
import numpy as np
import torch.nn as nn
import torch.autograd
#from skimage import io
from torch import optim
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
from thop import profile
from utils.train_info_save import ConsoleLogger,save_dict_to_txt
working_path = os.path.dirname(os.path.abspath(__file__))

from utils.loss import CrossEntropyLoss2d, weighted_BCE_logits, ChangeSimilarity
from utils.PersonLoss import PearsonLoss3
from utils.utils import accuracy, SCDD_eval_all, AverageMeter
from ptflops import get_model_complexity_info
# Data and model choose
###############################################
#from datasets.my_data import my_datasets as SD
from datasets.SECOND import Second as SD
from models.CG_SPNet import CG_SPNet as Net
import sys
train_BS=sys.argv[1:]
#train_BS=['','','10','0.001']
#train_BS=["50", "0.1", "8", "0", "D:/pycharm/CG_SPNet/checkpoints/SD/CG_SPNet_7e_mIoU73.72_Sek23.84_Fscd63.05_OA87.84.pth", "D:/pycharm/data/socend", "D:/pycharm/CG_SPNet/results", "D:/pycharm/CG_SPNet/logs", "D:/pycharm/CG_SPNet/checkpoints/SD", ""]
NET_NAME = 'CG_SPNet'
#DATA_NAME = 'CZ'
#DATA_NAME = 'MD'
DATA_NAME = 'SD'
#DATA_NAME = 'WZ'
###############################################
# Training options
###############################################
#0.00001
args = {
    'train_batch_size': int(train_BS[2]),
    'val_batch_size':int(train_BS[2]),
    'lr': float(train_BS[1]),#0.03
    'epochs':int(train_BS[0]),
    'gpu': True,
    'psd_TTA': True,
    'lr_decay_power': 1.5,#0.9
    'train_crop_size': False,
    'weight_decay': 5e-4,#1e4
    'momentum': 0.9,
    'print_freq': 10,
    'predict_step': 100,
    'pseudo_thred': 0.8,
    'pred_dir': train_BS[6],
    'chkpt_dir': train_BS[8],
    'log_dir': train_BS[7],
    'load_path': train_BS[4]
}
###############################################

if not os.path.exists(args['log_dir']): os.makedirs(args['log_dir'])
if not os.path.exists(args['pred_dir']): os.makedirs(args['pred_dir'])
if not os.path.exists(args['chkpt_dir']): os.makedirs(args['chkpt_dir'])
writer = SummaryWriter(args['log_dir'])
num='_stage1_with_init_'

def calc_conf(softmap):
    b, c, h, w = softmap.size()
    conf, index = torch.max(softmap, dim=1)
    index_onehot = F.one_hot(index.long(), num_classes=SD.num_classes).permute((0,3,1,2))
    masked_softmap = index_onehot*softmap
    threds = np.zeros(c)
    for idx in range(c):
        masked_softmap_i = torch.flatten(masked_softmap[:, idx])
        masked_softmap_i = masked_softmap_i[masked_softmap_i.nonzero()]
        len = masked_softmap_i.size(0)
        if len:
            masked_softmap_i, _ = torch.sort(masked_softmap_i, descending=True)
            mid_val = masked_softmap_i[len//2]
            threds[idx] = mid_val.cpu().detach().numpy() #*args['pseudo_thred']
        else:
            threds[idx] = 0.5
    threds[threds>0.9]=0.9
    threds = torch.from_numpy(threds).unsqueeze(1).unsqueeze(2).cuda()
    thred_onehot = index_onehot*threds
    thredmap, _ = torch.max(thred_onehot, dim=1)
    conf = torch.ge(conf, thredmap)
    return conf, index
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    #m_seed = 42
    # 设置seed
    # torch.manual_seed(m_seed)
    # torch.cuda.manual_seed_all(m_seed)
    #set_seed(m_seed)

    #os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    save_dict_to_txt(args,os.path.join(args['chkpt_dir'],NET_NAME+num+"_train_info.txt"))

    net = Net(3, SD.num_classes).cuda()
    #net = Net(use_DS=False).cuda()

    #net = Net(False).cuda()
    #net.load_state_dict(torch.load(args["load_path"]),True)
    print(args)
    # x2 = torch.randn((1, 3, 512, 512))
    # x1 = torch.randn((1, 3, 512, 512))

   #获取浮点数总数
    # flops, params = profile(net, inputs=torch.cat([x1.cuda(),x2.cuda()],dim=1))
    # print("\nTotal parameters: {:.2f}Mb".format(params / 1e6))
    # print("\nTotal flops: {:.2f}Gbps".format(flops / 1e9))
    #net.load_state_dict(torch.load(args['load_path']), strict=False)
    #freeze_model(net.FCN)

    train_set = SD.Data('lab', random_flip=True)
    #train_set = SD.Data_5z('train', random_flip=True)
    train_loader = DataLoader(train_set, batch_size=args['train_batch_size'], num_workers=0, shuffle=True)
    val_set = SD.Data('lab')
    val_loader = DataLoader(val_set, batch_size=args['val_batch_size'], num_workers=0, shuffle=False)

    params_num=sum(p.numel()for p in net.parameters())
    print("\nParams: %1fM"%(params_num/1e6))
    sys.stdout.flush()

    # flops, params = get_model_complexity_info(net, (3, 256, 256), as_strings=True,
    #                                           print_per_layer_stat=True)  # 不用写batch_size大小，默认batch_size=1
    # print('\nFlops:  ' + flops)
    # print('\nParams: ' + params)
    #criterion = CrossEntropyLoss2d(ignore_index=0).cuda()
    criterion = CrossEntropyLoss2d(ignore_index=0).cuda()
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args['lr'],
                          weight_decay=args['weight_decay'], momentum=args['momentum'], nesterov=True)
    torch.autograd.set_detect_anomaly(True)
    train(train_loader, net, criterion, optimizer, val_loader)
    writer.close()
    print('Training finished.')
    sys.stdout.flush()


def train(train_loader, net, criterion, optimizer, val_loader):
    net_psd = copy.deepcopy(net)
    net_psd.eval()
    bestaccV=0.0
    bestFscdV=0.0
    bestaccT = 0
    bestmIou=0.0
    bestscoreV = 0.0
    bestloss = 1.0
    patience=20
    count=0
    begin_time = time.time()
    all_iters = float(len(train_loader) * args['epochs'])
    criterion_sc = ChangeSimilarity().cuda()
    #criterion_sc = PearsonLoss3().cuda()
    curr_epoch = 0
    while True:
        torch.cuda.empty_cache()
        net.train()

        #freeze_model(net.FCN)
       # # # freeze_model(net.hr)
       #  freeze_model(net.resCD)
       #  freeze_model(net.classifierCD)
       #  freeze_model(net.classifier1)
       #  freeze_model(net.classifier2)
       #  freeze_model(net.fusion)
       #  freeze_model(net.transit)
       # #
       #  freeze_model(net.TF)
       #  freeze_model(net.down)
       #  freeze_model(net.down_sample)
       #  #
       #  freeze_model(net.siam_FPR)
       #  freeze_model(net.CGFE)

        # freeze_model(net.Siam_SR)
        # freeze_model(net.CotSR)

        #freeze_model(net.transformer)
        # freeze_model(net.DecCD)
        # freeze_model(net.Dec1)
        # freeze_model(net.Dec2)

        start = time.time()
        acc_meter = AverageMeter()
        train_seg_loss = AverageMeter()
        train_bn_loss = AverageMeter()
        train_sc_loss = AverageMeter()

        curr_iter = curr_epoch * len(train_loader)
        for i, data in tqdm(enumerate(train_loader),total=train_loader.__len__(),colour="white",ncols=200,desc="-----train  processing-----"):
            running_iter = curr_iter + i + 1

            adjust_lr(optimizer, running_iter, all_iters)
            imgs_A, imgs_B, labels_A, labels_B = data
            if args['gpu']:
                imgs_A = imgs_A.cuda().float()
                imgs_B = imgs_B.cuda().float()
                labels_bn = (labels_A > 0).unsqueeze(1).cuda().float()
                labels_A = labels_A.cuda().long()
                labels_B = labels_B.cuda().long()

            optimizer.zero_grad()

           # out_change, outputs_A, outputs_B = net(imgs_A, imgs_B)
            out_change, outputs_A, outputs_B = net(torch.cat([imgs_A,imgs_B],dim=1))

            assert outputs_A.size()[1] == SD.num_classes

            loss_seg = criterion(outputs_A, labels_A) + criterion(outputs_B, labels_B)
            loss_bn = weighted_BCE_logits(out_change, labels_bn)
            loss_sc = criterion_sc(outputs_A[:, 1:], outputs_B[:, 1:], labels_bn)
            loss = loss_seg*0.5 + loss_bn + loss_sc
            #loss = loss_seg*0.5  + loss_sc
            if True: #curr_epoch: #bestscoreV>0.3:
                with torch.no_grad():
                    out_change_psd, outputsA_psd, outputsB_psd = net_psd(torch.cat([imgs_A,imgs_B],dim=1))
                    softmap_A = F.softmax(outputsA_psd, dim=1)
                    softmap_B = F.softmax(outputsB_psd, dim=1)
                    out_change_psd = F.sigmoid(out_change_psd)
                    if args['psd_TTA']:
                        imgs_A_v = torch.flip(imgs_A, [2])
                        imgs_B_v = torch.flip(imgs_B, [2])
                        out_change_v, outputs_A_v, outputs_B_v = net_psd(torch.cat([imgs_A_v, imgs_B_v],dim=1))
                        outputs_A_v = torch.flip(outputs_A_v, [2])
                        outputs_B_v = torch.flip(outputs_B_v, [2])
                        out_change_v = torch.flip(out_change_v, [2])
                        softmap_A += F.softmax(outputs_A_v, dim=1)
                        softmap_B += F.softmax(outputs_B_v, dim=1)
                        out_change_psd += F.sigmoid(out_change_v)

                        imgs_A_h = torch.flip(imgs_A, [3])
                        imgs_B_h = torch.flip(imgs_B, [3])
                        out_change_h, outputs_A_h, outputs_B_h = net_psd(torch.cat([imgs_A_h, imgs_B_h],dim=1))
                        outputs_A_h = torch.flip(outputs_A_h, [3])
                        outputs_B_h = torch.flip(outputs_B_h, [3])
                        out_change_h = torch.flip(out_change_h, [3])
                        softmap_A += F.softmax(outputs_A_h, dim=1)
                        softmap_B += F.softmax(outputs_B_h, dim=1)
                        out_change_psd += F.sigmoid(out_change_h)

                        imgs_A_hv = torch.flip(imgs_A, [2, 3])
                        imgs_B_hv = torch.flip(imgs_B, [2, 3])
                        out_change_hv, outputs_A_hv, outputs_B_hv = net_psd(torch.cat([imgs_A_hv, imgs_B_hv],dim=1))
                        outputs_A_hv = torch.flip(outputs_A_hv, [2, 3])
                        outputs_B_hv = torch.flip(outputs_B_hv, [2, 3])
                        out_change_hv = torch.flip(out_change_hv, [2, 3])
                        softmap_A += F.softmax(outputs_A_hv, dim=1)
                        softmap_B += F.softmax(outputs_B_hv, dim=1)
                        out_change_psd += F.sigmoid(out_change_hv)

                        softmap_A = softmap_A / 4
                        softmap_B = softmap_B / 4
                        out_change_psd = out_change_psd / 4
                b, c, h, w = outputsA_psd.shape
                confA, A_index = calc_conf(softmap_A)
                confB, B_index = calc_conf(softmap_B)
                confAB = torch.logical_and(confA, confB)
                AB_same = torch.eq(A_index, B_index)
                confAB_same = torch.logical_and(confAB, AB_same)
                labels_unchange = torch.logical_not(labels_bn).squeeze()
                pseudo_unchange = (confAB_same*A_index*labels_unchange).long()
                loss_psd = criterion(outputs_A, pseudo_unchange) + criterion(outputs_B, pseudo_unchange)
                loss += loss_psd*0.5
            loss.backward()
            optimizer.step()
            #time.sleep(0.1)
            percentage=running_iter/all_iters
            percentage = format(percentage*100, ".4f")
            print(f"\ntrain_percentage: {percentage}%,sc_loss: {loss_sc}%,seg_loss:{loss_seg}%,bcd_loss:{loss_bn}%")
            sys.stdout.flush()



            labels_A = labels_A.cpu().detach().numpy()
            labels_B = labels_B.cpu().detach().numpy()
            outputs_A = outputs_A.cpu().detach()
            outputs_B = outputs_B.cpu().detach()
            change_mask = F.sigmoid(out_change).cpu().detach() > 0.5
            preds_A = torch.argmax(outputs_A, dim=1)
            preds_B = torch.argmax(outputs_B, dim=1)
            preds_A = (preds_A * change_mask.squeeze().long()).numpy()
            preds_B = (preds_B * change_mask.squeeze().long()).numpy()
            # batch_valid_sum = 0
            acc_curr_meter = AverageMeter()
            for (pred_A, pred_B, label_A, label_B) in zip(preds_A, preds_B, labels_A, labels_B):
                acc_A, valid_sum_A = accuracy(pred_A, label_A)
                acc_B, valid_sum_B = accuracy(pred_B, label_B)
                acc = (acc_A + acc_B) * 0.5
                acc_curr_meter.update(acc)
            acc_meter.update(acc_curr_meter.avg)
            train_seg_loss.update(loss_seg.cpu().detach().numpy())
            train_bn_loss.update(loss_bn.cpu().detach().numpy())
            train_sc_loss.update(loss_sc.cpu().detach().numpy())

            curr_time = time.time() - start
            if (i + 1) % args['print_freq'] == 0:
                print('[epoch: %d] [iter: %d / total:%d %.1fs] [lr %f] [train seg_loss %.4f bn_loss %.4f acc %.2f]' % (
                    curr_epoch, i + 1, len(train_loader), curr_time, optimizer.param_groups[0]['lr'],
                    train_seg_loss.val, train_bn_loss.val, acc_meter.val * 100))  # sc_loss %.4f, train_sc_loss.val,
                #sys.stdout.flush()
                writer.add_scalar('train seg_loss', train_seg_loss.val, running_iter)
                writer.add_scalar('train sc_loss', train_sc_loss.val, running_iter)
                writer.add_scalar('train accuracy', acc_meter.val, running_iter)
                writer.add_scalar('lr', optimizer.param_groups[0]['lr'], running_iter)

        Fscd_v, mIoU_v, Sek_v, acc_v, loss_v = validate(val_loader, net, criterion,criterion_sc, curr_epoch)
        if acc_meter.avg > bestaccT: bestaccT = acc_meter.avg
        if Fscd_v>bestFscdV or mIoU_v>bestmIou:
            bestFscdV=Fscd_v
            bestaccV=acc_v
            bestloss=loss_v
            bestmIou=mIoU_v
            count=0
            torch.save(net.state_dict(), os.path.join(args['chkpt_dir'], NET_NAME+'_%de_mIoU%.2f_Sek%.2f_Fscd%.2f_OA%.2f.pth'\
                %(curr_epoch, mIoU_v*100, Sek_v*100, Fscd_v*100, acc_v*100)) )
        else:
            count+=1
        # if count>patience:
        #     print("early stopping!")
        #     print('Total time: %.1fs Best rec: Train acc %.2f, Val Fscd %.2f acc %.2f loss %.4f miou %.4f' % (
        #     time.time() - begin_time, bestaccT * 100, bestFscdV * 100, bestaccV * 100, bestloss,bestmIou))
        #     break
        #('Valid_result: Total time: %.1fs Best rec: Train acc %.2f, Val mIou: %.2f  Fscd: %.2f acc %.2f loss: %.4f Sek: %.2f' %(time.time()-begin_time, bestaccT*100, bestmIou*100,bestFscdV*100, bestaccV*100, bestloss,Sek_v*100))
        print(f'Valid_result: Total time:{round(time.time()-begin_time,2)}%s Best rec: Train acc:{round(bestaccT*100,2)}%,val_acc:{round(bestaccV*100,2)}% loss:{round(bestloss,4)}%,Val mIou:{round(bestmIou*100,2)}%, Fscd:{round(bestFscdV*100,2)}%,Sek:{round(Sek_v*100,2)}%')

        sys.stdout.flush()
        curr_epoch += 1
        if curr_epoch >= args['epochs']:
            return

def validate(val_loader, net, criterion,criterion_sc, curr_epoch):
    # the following code is written assuming that batch size is 1
    net.eval()
    torch.cuda.empty_cache()
    start = time.time()

    val_sc_loss=AverageMeter()
    val_loss = AverageMeter()
    acc_meter = AverageMeter()

    preds_all = []
    labels_all = []
    all_iter=len(val_loader)
    for vi, data in tqdm(enumerate(val_loader),total=val_loader.__len__(),colour="green",ncols=100,desc="-----val  processing-----"):
        imgs_A, imgs_B, labels_A, labels_B = data
        if args['gpu']:
            imgs_A = imgs_A.cuda().float()
            imgs_B = imgs_B.cuda().float()
            labels_A = labels_A.cuda().long()
            labels_B = labels_B.cuda().long()
            labels_bn = (labels_A > 0).unsqueeze(1).cuda().float()

        with torch.no_grad():
            out_change, outputs_A, outputs_B = net(torch.cat([imgs_A,imgs_B],dim=1))
            loss_A = criterion(outputs_A, labels_A)
            loss_B = criterion(outputs_B, labels_B)
            loss = loss_A * 0.5 + loss_B * 0.5
            TEMP=outputs_A[:, 1:]
            loss_sc = criterion_sc(outputs_A[:, 1:], outputs_B[:, 1:], labels_bn)
        val_loss.update(loss.cpu().detach().numpy())
        val_sc_loss.update(loss_sc.cpu().detach().numpy())
        labels_A = labels_A.cpu().detach().numpy()
        labels_B = labels_B.cpu().detach().numpy()
        outputs_A = outputs_A.cpu().detach()
        outputs_B = outputs_B.cpu().detach()
        change_mask = F.sigmoid(out_change).cpu().detach() > 0.5
        preds_A = torch.argmax(outputs_A, dim=1)
        preds_B = torch.argmax(outputs_B, dim=1)
        preds_A = (preds_A * change_mask.squeeze().long()).numpy()
        preds_B = (preds_B * change_mask.squeeze().long()).numpy()
        print(f"val_percentage:{100*(vi/all_iter)}%")
        sys.stdout.flush()
        for (pred_A, pred_B, label_A, label_B) in zip(preds_A, preds_B, labels_A, labels_B):
            acc_A, valid_sum_A = accuracy(pred_A, label_A)
            acc_B, valid_sum_B = accuracy(pred_B, label_B)
            preds_all.append(pred_A)
            preds_all.append(pred_B)
            labels_all.append(label_A)
            labels_all.append(label_B)
            acc = (acc_A + acc_B) * 0.5
            acc_meter.update(acc)

        if curr_epoch % args['predict_step'] == 0 and vi == 0:
            pred_A_color = SD.Index2Color(preds_A[0])
            pred_B_color = SD.Index2Color(preds_B[0])
            #io.imsave(os.path.join(args['pred_dir'], NET_NAME + '_A.png'), pred_A_color)
            cv2.imwrite(os.path.join(args['pred_dir'], NET_NAME + '_A.png'), pred_A_color)
           # io.imsave(os.path.join(args['pred_dir'], NET_NAME + '_B.png'), pred_B_color)
            cv2.imwrite(os.path.join(args['pred_dir'], NET_NAME + '_B.png'), pred_B_color)
            print('Prediction saved!')

    Fscd, IoU_mean, Sek ,_,_,_,_,_= SCDD_eval_all(preds_all, labels_all, SD.num_classes)

    curr_time = time.time() - start
    with ConsoleLogger(args['chkpt_dir'],NET_NAME,num=num) as logger :
        print('epoch: %d, times: %.1fs Val seg loss: %.2f Val sc loss: %.2f Fscd: %.2f IoU: %.2f Sek: %.2f Accuracy: %.2f'\
        %(curr_epoch,curr_time, val_loss.average(),val_sc_loss.average(), Fscd*100, IoU_mean*100, Sek*100, acc_meter.average()*100))
    print('epoch: %d, times: %.1fs Val seg loss: %.2f Val sc loss: %.2f Fscd: %.2f IoU: %.2f Sek: %.2f Accuracy: %.2f'\
    %(curr_epoch,curr_time, val_loss.average(),val_sc_loss.average(), Fscd*100, IoU_mean*100, Sek*100, acc_meter.average()*100))
    writer.add_scalar('val_loss', val_loss.average(), curr_epoch)
    writer.add_scalar('val_sc_loss', val_sc_loss.average(), curr_epoch)
    writer.add_scalar('miou', IoU_mean, curr_epoch)
    writer.add_scalar('val_Fscd', Fscd, curr_epoch)
    writer.add_scalar('Sek', Sek, curr_epoch)
    writer.add_scalar('val_Accuracy', acc_meter.average(), curr_epoch)

    return Fscd, IoU_mean, Sek, acc_meter.avg, val_loss.avg


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()


def adjust_lr(optimizer, curr_iter, all_iter, init_lr=args['lr']):
    scale_running_lr = ((1. - float(curr_iter) / all_iter) ** args['lr_decay_power'])
    running_lr = init_lr * scale_running_lr

    for param_group in optimizer.param_groups:
        param_group['lr'] = running_lr


if __name__ == '__main__':
    main()
