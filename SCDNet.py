# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.MS_Attention.attention import ChannelAttention, SpatialAttention, SCAttention, ChannelAttention1, \
    PAM_Module, CAM_Module,BAM_Att,ChannelAttentionHL
from models.MS_Attention.attention import SCSEBlock
from models.block.dropblock import DropBlock2D,LinearScheduler
import numpy as np
from functools import partial
import logging
logger = logging.getLogger('base')

class REBNCONV(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, dirate=1):
        super(REBNCONV, self).__init__()

        self.conv_s1 = nn.Conv2d(in_ch, out_ch, 3, padding=1 * dirate, dilation=1 * dirate)
        self.bn_s1 = nn.BatchNorm2d(out_ch)
        self.relu_s1 = nn.ReLU(inplace=True)

    def forward(self, x):
        hx = x
        xout = self.relu_s1(self.bn_s1(self.conv_s1(hx)))

        return xout
class ResBlock(nn.Module):
    def __init__(
        self, in_ch, out_ch,
        ):

        super(ResBlock, self).__init__()

        self.conv_main = nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, 1, 1),
                                      nn.BatchNorm2d(out_ch),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(out_ch, out_ch, 3, 1, 1),
                                      nn.BatchNorm2d(out_ch),
                                      nn.ReLU(inplace=True),#not use?
                                      )
        self.conv_short=nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, 1, 1),
                                      nn.BatchNorm2d(out_ch))
    def forward(self, x):
        x_main=self.conv_main(x)
        x_short=self.conv_short(x)

        return x_main+x_short

class ASPP(nn.Module):
    def __init__(self, in_channel=512):
        super(ASPP, self).__init__()
        # global average pooling : init nn.AdaptiveAvgPool2d ;also forward torch.mean(,,keep_dim=True)
        '''
        ASPP中的rate=6，12，18是针对于输出stride=16的情况。
        输出stride=8时，rate再乘以2，以维持相同的感受野大小。rate的取值在实验中也进行调参比较。
        https://zhuanlan.zhihu.com/p/147822276
        stride=32==>[3,6,9]?
        '''
        depth = int(in_channel)
        self.mean = nn.AdaptiveAvgPool2d((1, 1))
        self.conv = nn.Conv2d(in_channel, depth, 1, 1)
        # k=1 s=1 no pad
        self.atrous_block1 = nn.Conv2d(in_channel, depth, 1, 1)# default dilation=1
        self.atrous_block2 = nn.Conv2d(in_channel, depth, 3, 1, padding=3, dilation=3)
        self.atrous_block4 = nn.Conv2d(in_channel, depth, 3, 1, padding=7, dilation=7)
        self.atrous_block8 = nn.Conv2d(in_channel, depth, 3, 1, padding=15, dilation=15)

        self.conv_1x1_output = nn.Conv2d(depth * 5, depth, 1, 1)
        self.ReLU = nn.ReLU(inplace=True)
        self.att = False
        if self.att:
            self.pam = PAM_Module(depth)

    def forward(self, x):
        '''
        using sigle conv , thus the ROF is 1, 7, 15,31 for d=1,3,7,15
        :param x:
        :return:
        '''
        size = x.shape[2:]

        image_features = self.mean(x)
        image_features = self.conv(image_features)
        image_features = F.upsample(image_features, size=size, mode='bilinear')

        atrous_block1 = self.ReLU(self.atrous_block1(x))

        atrous_block2 = self.ReLU(self.atrous_block2(x))

        atrous_block4 = self.ReLU(self.atrous_block4(x))

        atrous_block8 = self.ReLU(self.atrous_block8(x))
        # if self.att:
        #     image_features = self.pam(image_features) * image_features
        #     astrous_block1 = self.pam(atrous_block1) * atrous_block1
        #     astrous_block2 = self.pam(atrous_block2) * atrous_block2
        #     astrous_block4 = self.pam(atrous_block4) * atrous_block4
        #     astrous_block8 = self.pam(atrous_block8) * atrous_block8

        net = self.conv_1x1_output(torch.cat([image_features, atrous_block1, atrous_block2,
                                              atrous_block4, atrous_block8], dim=1))
        return net


class Dblock(nn.Module):
    def __init__(self, channel,refine=False):
        super(Dblock, self).__init__()
        # self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        # self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=2, padding=2)
        # self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=4, padding=4)
        # self.dilate4 = nn.Conv2d(channel, channel, kernel_size=3, dilation=8, padding=8)

        self.dilate1 = REBNCONV(channel, channel, dirate=1)
        self.dilate2 = REBNCONV(channel, channel, dirate=2)
        self.dilate3 = REBNCONV(channel, channel, dirate=4)
        self.dilate4 = REBNCONV(channel, channel, dirate=6)

        #self.conv_final=nn.Conv2d(channel, channel, kernel_size=3, padding=1)

        self.refine=refine
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        # dilate1_out = nonlinearity(self.dilate1(x))
        # dilate2_out = nonlinearity(self.dilate2(dilate1_out))
        # dilate3_out = nonlinearity(self.dilate3(dilate2_out))
        # dilate4_out = nonlinearity(self.dilate4(dilate3_out))
        '''
        using 4-layer conv, thus the ROF is 1, 7, 15,31
        :param x:
        :return:
        '''
        dilate1_out = self.dilate1(x)
        dilate2_out = self.dilate2(dilate1_out)
        dilate3_out = self.dilate3(dilate2_out)
        #dilate4_out = self.dilate4(dilate3_out)

        out = x+dilate1_out + dilate2_out + dilate3_out# + dilate4_out  # + dilate5_out


        return out

class unetUp2(nn.Module):
    def __init__(self, in_size, out_size, is_deconv=True,use_att=False,act_mode='relu'):
        super(unetUp2, self).__init__()
        if act_mode=='relu':
            conv_act=nn.ReLU(inplace=True)
        else:
            conv_act=nn.LeakyReLU(0.2, inplace=True)

        if is_deconv:
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=4, stride=2, padding=1)
        else:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)

        self.use_att=use_att
        if self.use_att:
            self.att = ChannelAttentionHL(in_size, out_size)#methods1
            #self.att=ChannelAttentionHL(out_size*3,out_size)#method2

        self.conv_cat = ResBlock(out_size * 2, out_size)

    def forward(self, inputs1, inputs2):
        outputs1 = self.up(inputs1)
        if self.use_att:
            #=========methods1======
            att_map=self.att(inputs1)
            inputs2=att_map*inputs2#diff
            # inputs3=att_map*inputs3#feat1
            # inputs4 = att_map * inputs4#feat2
            #===========method2==================
            # outputs123=torch.cat([outputs1,inputs2,inputs3], 1)
            # att_map=self.att(outputs123)
            # inputs2=inputs2*att_map
            # inputs3=inputs3*att_map


        return self.conv_cat(torch.cat([outputs1,inputs2], 1))

class unetUp3(nn.Module):
    def __init__(self, in_size, out_size, is_deconv=True,use_att=False,act_mode='relu',use_se=False,se_block='SCSE',drop_block=None):
        super(unetUp3, self).__init__()
        if act_mode=='relu':
            conv_act=nn.ReLU(inplace=True)
        else:
            conv_act=nn.LeakyReLU(0.2, inplace=True)
        if use_se:
            if se_block=='SCSE':
                self.conv = nn.Sequential(
                    # drop_block,#use drop after cat
                    nn.Conv2d(out_size * 3, out_size, 3, 1, 1),
                    nn.BatchNorm2d(out_size),
                    conv_act,
                    nn.Conv2d(out_size, out_size, 3, 1, 1),
                    nn.BatchNorm2d(out_size),
                    conv_act,
                    SCSEBlock(out_size)  # seem to worse on the test
                )
            else:
                self.conv = nn.Sequential(
                    # drop_block,#use drop after cat
                    nn.Conv2d(out_size * 3, out_size, 3, 1, 1),
                    nn.BatchNorm2d(out_size),
                    conv_act,
                    nn.Conv2d(out_size, out_size, 3, 1, 1),
                    nn.BatchNorm2d(out_size),
                    conv_act,
                    BAM_Att(out_size)  # seem to worse on the test
                )

        else:
            self.conv = nn.Sequential(nn.Conv2d(out_size * 3, out_size, 3, 1, 1),
                                      nn.BatchNorm2d(out_size),
                                      conv_act,
                                      nn.Conv2d(out_size, out_size, 3, 1, 1),
                                      nn.BatchNorm2d(out_size),
                                      conv_act
                                      )




        if is_deconv:
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=4, stride=2, padding=1)
        else:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)

        self.use_att=use_att
        if self.use_att:
            self.att = ChannelAttentionHL(in_size, out_size)#methods1
            #self.att=ChannelAttentionHL(out_size*3,out_size)#method2


    def forward(self, inputs1, inputs2,inputs3):
        outputs1 = self.up(inputs1)
        if self.use_att:
            #=========methods1======
            att_map=self.att(inputs1)
            inputs2=att_map*inputs2
            inputs3=att_map*inputs3
            #===========method2==================
            # outputs123=torch.cat([outputs1,inputs2,inputs3], 1)
            # att_map=self.att(outputs123)
            # inputs2=inputs2*att_map
            # inputs3=inputs3*att_map


        return self.conv(torch.cat([outputs1,inputs2,inputs3], 1))

class unetUp3_Res(nn.Module):
    def __init__(self, in_size, out_size, is_deconv=True,use_att=False,act_mode='relu',use_se=False):
        super(unetUp3_Res, self).__init__()
        if act_mode=='relu':
            conv_act=nn.ReLU(inplace=True)
        else:
            conv_act=nn.LeakyReLU(0.2, inplace=True)
        # if use_se:
        #     self.conv = nn.Sequential(nn.Conv2d(out_size * 3, out_size, 3, 1, 1),
        #                               nn.BatchNorm2d(out_size),
        #                               conv_act,
        #                               # nn.Conv2d(out_size, out_size, 3, 1, 1),
        #                               # nn.BatchNorm2d(out_size),
        #                               # conv_act
        #                               #SCSEBlock(out_size)
        #                               )
        # else:
        #     self.conv = nn.Sequential(nn.Conv2d(out_size * 3, out_size, 3, 1, 1),
        #                               nn.BatchNorm2d(out_size),
        #                               conv_act,
        #                               nn.Conv2d(out_size, out_size, 3, 1, 1),
        #                               nn.BatchNorm2d(out_size),
        #                               conv_act
        #                               )
        self.SEBlock=SCSEBlock(out_size)
        # self.conv_cat=nn.Sequential(nn.Conv2d(out_size*3, out_size, 3, 1, 1),
        #                               nn.BatchNorm2d(out_size),
        #                               conv_act)
        # self.conv2=nn.Sequential(nn.Conv2d(out_size, out_size, 3, 1, 1),
        #                               nn.BatchNorm2d(out_size),
        #                               #conv_act
        #                          )
        self.conv_cat=ResBlock(out_size*3,out_size)


        if is_deconv:
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=4, stride=2, padding=1)
        else:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)

        self.use_att=use_att
        if self.use_att:
            self.att = ChannelAttentionHL(in_size, out_size)#methods1
            #self.att=ChannelAttentionHL(out_size*3,out_size)#method2


    def forward(self, inputs1, inputs2,inputs3):
        outputs1 = self.up(inputs1)
        if self.use_att:
            #=========methods1======
            att_map=self.att(inputs1)
            inputs2=att_map*inputs2
            inputs3=att_map*inputs3

        conv_out=self.conv_cat(torch.cat([outputs1,inputs2,inputs3], 1))



        return conv_out

class unet_2D_Encoder(nn.Module):
    def __init__(self, feature_scale=4, n_classes=1, is_deconv=True, in_channels=3, is_batchnorm=True, use_att=False,
                 use_res=False,
                 dblock_type='AS',use_se=False):
        super(unet_2D_Encoder, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        self.n_class = n_classes
        self.use_att = use_att
        self.use_dblock=False

        if dblock_type=='AS':
            logger.info("using AS for center...")
            self.dblock=Dblock(512)
            self.use_dblock=True
        elif dblock_type=='ASPP':
            logger.info("using ASPP for center...")
            self.dblock=ASPP(in_channel=512)
            self.use_dblock = True

        else:
            logger.info("using No for center...")
            self.dblock = None
            self.use_dblock = False

        # downsampling
        from torchvision import models
        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        if use_se:
            logger.info("using se for encoder...")
            self.encoder1 = nn.Sequential(resnet.layer1,
                                          SCSEBlock(filters[0]))
            self.encoder2 = nn.Sequential(resnet.layer2,
                                          SCSEBlock(filters[1]))
            self.encoder3 = nn.Sequential(resnet.layer3,
                                          SCSEBlock(filters[2]))
            self.encoder4 = nn.Sequential(resnet.layer4,
                                          SCSEBlock(filters[3]))
        else:
            self.encoder1 = resnet.layer1
            self.encoder2 = resnet.layer2
            self.encoder3 = resnet.layer3
            self.encoder4 = resnet.layer4

        #self.dblock = Dblock(512)

    def forward(self, inputs):

        x = self.firstconv(inputs)  # [2,64,128,128]
        x = self.firstbn(x)
        x_conv = self.firstrelu(x)
        x1 = self.firstmaxpool(x_conv)  # [2,64,64,64]
        e1 = self.encoder1(x1)  ##[2,64,64,64]
        e2 = self.encoder2(e1)  # [2,128,32,32]
        e3 = self.encoder3(e2)  # [2,256,16,16]
        e4 = self.encoder4(e3)  # [2,512,8,8]

        # # Center
        if self.use_dblock:
            e4 = self.dblock(e4)  # [2,512,8,8]   in decoder
        return  x_conv,e1,e2,e3,e4


#decoder
class EDCls_UNet2_New2(nn.Module):

   def __init__(self, in_c=3, use_att=False,diff_mode='diff', training_mode=True,att_mode='BAM',act_mode='relu',nf=64,out_c=7,net_feat=None,use_DS=True,use_CatOut=False,cat_mode='cat_feat',use_se=False,drop_rate=0.1,se_block='BAM'):
        super(EDCls_UNet2_New2, self).__init__()
        self.feat_Extactor=net_feat
        filters=[64,128,256,512]
        if act_mode=='relu':
            conv_act=nn.ReLU(inplace=True)
        else:
            conv_act=nn.LeakyReLU(0.2, inplace=True)

        self.use_att = use_att
        self.use_CatOut=use_CatOut
        self.cat_mode = cat_mode
        self.training_mode=training_mode


        if use_att:#=====can be removed
            # message="using att_mode is {}".format(att_mode)
            # logger.info(message)
            # if att_mode == 'BAM':
            #     # self.AttFunc1 = BAM(filters[0], ds=8)
            #     # self.AttFunc2 = BAM(filters[1], ds=4)
            #     # self.AttFunc3 = BAM(filters[2], ds=2)
            #
            #     self.AttFunc4 = BAM(filters[-1], ds=1)#using multiple PAM for feat1,feat2,feat3,feat4?
            # elif att_mode=='PCAM':
            #     self.AttFunc4=PCAM(filters[-1])
            # else:
            #     self.AttFunc4 = PAM(in_channels=filters[-1], out_channels=filters[-1], sizes=[1, 2, 4, 8], ds=1)
            deconv_att = True
            logger.info("use att for decoder...")
        else:
            deconv_att=False
            logger.info("no att for decoder...")

        #se_block=se_block




        self.use_drop = False
        if drop_rate>0:
            self.use_drop=True
            message="using dropblock with rate {:.3f} ".format(drop_rate)
            logger.info(message)

            self.dropblock = LinearScheduler(
                DropBlock2D(drop_prob=drop_rate, block_size=7),
                start_value=0.,
                stop_value=drop_rate,
                nr_steps=1e4
            )
        logger.info("use residual for decoder...")
        self.decoder4 = unetUp3_Res(filters[2] * 2, filters[2], use_att=deconv_att, act_mode=act_mode,
                                use_se=use_se)
        self.decoder3 = unetUp3_Res(filters[1] * 2, filters[1], use_att=deconv_att, act_mode=act_mode,
                                use_se=use_se)
        self.decoder2 = unetUp3_Res(filters[0] * 2, filters[0], use_att=deconv_att, act_mode=act_mode,
                                use_se=use_se)
        self.decoder1 = unetUp3_Res(filters[0], filters[0], use_att=deconv_att, act_mode=act_mode,
                                use_se=use_se)

        # self.decoder4 = unetUp3(filters[2] * 2, filters[2], use_att=deconv_att, act_mode=act_mode,
        #                         use_se=use_se, drop_block=self.dropblock,se_block=se_block)
        # self.decoder3 = unetUp3(filters[1] * 2, filters[1], use_att=deconv_att, act_mode=act_mode,
        #                         use_se=use_se, drop_block=self.dropblock,se_block=se_block)
        # self.decoder2 = unetUp3(filters[0] * 2, filters[0], use_att=deconv_att, act_mode=act_mode,
        #                         use_se=use_se, drop_block=self.dropblock,se_block=se_block)
        # self.decoder1 = unetUp3(filters[0], filters[0], use_att=deconv_att, act_mode=act_mode,
        #                         use_se=use_se, drop_block=self.dropblock,se_block=se_block)



        self.class_conv = nn.Sequential(nn.ConvTranspose2d(64, 32, 4, 2, 1),  # size*2
                                        conv_act,
                                        nn.Conv2d(32, 32, 3, padding=1),
                                        conv_act,
                                        self.dropblock,
                                        nn.Conv2d(32, out_c, 3, padding=1)
                                        )

        self.diff_mode=diff_mode
        logger.info("diff mode is {}".format(diff_mode))
        if diff_mode=='cat':
            self.conv_cat01=nn.Conv2d(filters[0]*2,filters[0],1)
            self.conv_cat2=nn.Conv2d(filters[1]*2,filters[1],1)
            self.conv_cat3 = nn.Conv2d(filters[2] * 2, filters[2], 1)
            self.conv_cat4 = nn.Conv2d(filters[3] * 2, filters[3], 1)
        self.use_DS=use_DS


        if self.use_DS:
            logger.info("using deep supervsion...")
            if self.use_drop:
                self.class_conv1 = nn.Sequential(
                                                 nn.Conv2d(filters[2], 32, 3, 1, 1),  # 256
                                                 conv_act,
                                                 nn.Conv2d(32, 32, 3, padding=1),
                                                 conv_act,
                                                 self.dropblock,
                                                 nn.Conv2d(32, out_c, 3, padding=1)
                                                 )
                self.class_conv2 = nn.Sequential(nn.Conv2d(filters[1], 32, 3, 1, 1),  # 128
                                                 conv_act,
                                                 nn.Conv2d(32, 32, 3, padding=1),
                                                 conv_act,
                                                 self.dropblock,
                                                 nn.Conv2d(32, out_c, 3, padding=1)
                                                 )
                self.class_conv3 = nn.Sequential(nn.Conv2d(filters[0], 32, 3, 1, 1),  # 64
                                                 conv_act,
                                                 nn.Conv2d(32, 32, 3, padding=1),
                                                 conv_act,
                                                 self.dropblock,
                                                 nn.Conv2d(32, out_c, 3, padding=1)
                                                 )
            else:
                self.classifier1 = nn.Sequential(nn.Conv2d(filters[2], 32, 3, 1, 1),  # 256
                                                 conv_act,
                                                 nn.Conv2d(32, 32, 3, padding=1),
                                                 conv_act,
                                                 nn.Conv2d(32, out_c, 3, padding=1))
                self.classifier2 = nn.Sequential(nn.Conv2d(filters[1], 32, 3, 1, 1),  # 128
                                                 conv_act,
                                                 nn.Conv2d(32, 32, 3, padding=1),
                                                 conv_act,
                                                 nn.Conv2d(32, out_c, 3, padding=1))
                self.classifier3 = nn.Sequential(nn.Conv2d(filters[0], 32, 3, 1, 1),  # 64
                                                 conv_act,
                                                 nn.Conv2d(32, 32, 3, padding=1),
                                                 conv_act,
                                                 nn.Conv2d(32, out_c, 3, padding=1))

        else:
            logger.info("using no deep supervsion...")

   def forward(self, x1, x2):
        if self.use_drop:
            self.dropblock.step()
        feat1_0,feat1_1,feat1_2,feat1_3,feat1_4=self.feat_Extactor(x1)
        feat2_0, feat2_1, feat2_2, feat2_3, feat2_4=self.feat_Extactor(x2)
        # if self.use_att:
        #
        #     height = feat1_4.shape[3]
        #     feat12_4 = torch.cat((feat1_4, feat2_4), 3)  # 2[1,64,16,16]==>[1,64,16,32]
        #     feat12_4 = self.AttFunc4(feat12_4)
        #     feat1_4, feat2_4 = feat12_4[:, :, :, 0:height], feat12_4[:, :, :, height:]


        if self.diff_mode=='diff':
            feat12_0 = torch.abs(feat1_0 - feat2_0)  # [4,64,256,256]
            feat12_1 = torch.abs(feat1_1 - feat2_1)  # [4,64,128,128]
            feat12_2 = torch.abs(feat1_2 - feat2_2)  # [4,128,64,64]
            feat12_3 = torch.abs(feat1_3 - feat2_3)  # [4,256,32,32]
            feat12_4 = torch.abs(feat1_4 - feat2_4)  # [4,512,16,16]
        else:
            feat12_0=self.conv_cat01(torch.cat([feat1_0,feat2_0],dim=1))
            feat12_1 = self.conv_cat01(torch.cat([feat1_1, feat2_1], dim=1))
            feat12_2 = self.conv_cat2(torch.cat([feat1_2, feat2_2], dim=1))
            feat12_3 = self.conv_cat3(torch.cat([feat1_3, feat2_3], dim=1))
            feat12_4 = self.conv_cat4(torch.cat([feat1_4, feat2_4], dim=1))


        d4_1 = self.decoder4(feat12_4,feat12_3,feat1_3)  # #[4,256,16,16] 16
        d3_1 = self.decoder3(d4_1,feat12_2,feat1_2)  # [4,128,32,32] 32
        d2_1 = self.decoder2(d3_1,feat12_1,feat1_1) # [4,64,64,64]64
        d1_1 = self.decoder1(d2_1,feat12_0,feat1_0)  # 128


        d4_2 = self.decoder4(feat12_4, feat12_3, feat2_3)  # 16
        d3_2 = self.decoder3(d4_2, feat12_2, feat2_2)  # 32
        d2_2 = self.decoder2(d3_2, feat12_1, feat2_1)  # 64
        d1_2 = self.decoder1(d2_2, feat12_0, feat2_0)  # 128

        pred1 = self.class_conv(d1_1)
        pred2 = self.class_conv(d1_2)
        if self.training_mode:
            if self.use_DS:

                pred1_0, pred2_0 = self.class_conv1(d4_1), self.class_conv1(d4_2)
                pred1_1, pred2_1 = self.class_conv2(d3_1), self.class_conv2(d3_2)
                pred1_2, pred2_2 = self.class_conv3(d2_1), self.class_conv3(d2_2)

                return (pred1_0, pred2_0), (pred1_1, pred2_1), (pred1_2, pred2_2), pred1, pred2

        return  pred1,pred2


class EDCls_UNet2_MC7Bin(nn.Module):

   def __init__(self, in_c=3, use_att=False,diff_mode='diff', training_mode=True,att_mode='BAM',act_mode='relu',nf=64,out_c=7,net_feat=None,use_DS=False,use_CatOut=False,cat_mode='cat_feat',use_se=False,drop_rate=0.1,se_block='BAM'):
        super(EDCls_UNet2_MC7Bin, self).__init__()
        self.feat_Extactor=net_feat
        filters=[64,128,256,512]
        if act_mode=='relu':
            conv_act=nn.ReLU(inplace=True)
        else:
            conv_act=nn.LeakyReLU(0.2, inplace=True)

        self.use_att = use_att
        self.use_CatOut=use_CatOut
        self.cat_mode = cat_mode
        self.training_mode=training_mode


        if use_att:#=====can be removed

            deconv_att=True
            logger.info("use att for decoder...")
        else:
            deconv_att=False
            logger.info("no att for decoder...")

        self.use_drop = False
        if drop_rate>0:
            self.use_drop=True
            message="using dropblock with rate {:.3f} ".format(drop_rate)
            logger.info(message)

            self.dropblock = LinearScheduler(
                DropBlock2D(drop_prob=drop_rate, block_size=7),
                start_value=0,
                stop_value=drop_rate,
                nr_steps=10000
            )
        # logger.info("use residual for decoder...")
        # self.decoder4 = unetUp3_Res(filters[2] * 2, filters[2], use_att=deconv_att, act_mode=act_mode,
        #                         use_se=use_se)
        # self.decoder3 = unetUp3_Res(filters[1] * 2, filters[1], use_att=deconv_att, act_mode=act_mode,
        #                         use_se=use_se)
        # self.decoder2 = unetUp3_Res(filters[0] * 2, filters[0], use_att=deconv_att, act_mode=act_mode,
        #                         use_se=use_se)
        # self.decoder1 = unetUp3_Res(filters[0], filters[0], use_att=deconv_att, act_mode=act_mode,
        #                         use_se=use_se)
        #self.decoder4 = unetUp2(filters[4], filters[3], use_att=deconv_att, act_mode=act_mode)
        #==========================================================
        logger.info("use new residual for decoder...")
        self.decoder4 = unetUp3(filters[2] * 2, filters[2], use_att=deconv_att, act_mode=act_mode,
                                        use_se=use_se)
        self.decoder3 = unetUp3(filters[1] * 2, filters[1], use_att=deconv_att, act_mode=act_mode,
                                        use_se=use_se)
        self.decoder2 = unetUp3(filters[0] * 2, filters[0], use_att=deconv_att, act_mode=act_mode,
                                        use_se=use_se)
        self.decoder1 = unetUp3(filters[0], filters[0], use_att=deconv_att, act_mode=act_mode,
                                        use_se=use_se)
        self.decoder4_diff = unetUp2(filters[2] * 2, filters[2], use_att=deconv_att, act_mode=act_mode)
        self.decoder3_diff = unetUp2(filters[1] * 2, filters[1], use_att=deconv_att, act_mode=act_mode)
        self.decoder2_diff = unetUp2(filters[0] * 2, filters[0], use_att=deconv_att, act_mode=act_mode)
        self.decoder1_diff = unetUp2(filters[0], filters[0], use_att=deconv_att, act_mode=act_mode)

        self.class_conv = nn.Sequential(nn.ConvTranspose2d(64, 32, 4, 2, 1),  # size*2
                                        conv_act,
                                        nn.Conv2d(32, 32, 3, padding=1),
                                        conv_act,
                                        self.dropblock,
                                        nn.Conv2d(32, out_c, 3, padding=1)
                                        )

        self.class_conv_bin = nn.Sequential(nn.ConvTranspose2d(64, 32, 4, 2, 1),  # size*2
                                        conv_act,
                                        nn.Conv2d(32, 32, 3, padding=1),
                                        conv_act,
                                        self.dropblock,
                                        nn.Conv2d(32, 1, 3, padding=1),
                                        nn.Sigmoid()
                                        )

        self.diff_mode=diff_mode
        logger.info("diff mode is {}".format(diff_mode))
        if diff_mode=='cat':
            self.conv_cat01=nn.Conv2d(filters[0]*2,filters[0],1)
            self.conv_cat2=nn.Conv2d(filters[1]*2,filters[1],1)
            self.conv_cat3 = nn.Conv2d(filters[2] * 2, filters[2], 1)
            self.conv_cat4 = nn.Conv2d(filters[3] * 2, filters[3], 1)
        self.use_DS=use_DS
        if self.use_DS:
            logger.info("using deep supervsion...")
            if self.use_drop:
                self.class_conv1 = nn.Sequential(
                                                 nn.Conv2d(filters[2], 32, 3, 1, 1),  # 256
                                                 conv_act,
                                                 nn.Conv2d(32, 32, 3, padding=1),
                                                 conv_act,
                                                 self.dropblock,
                                                 nn.Conv2d(32, out_c, 3, padding=1)
                                                 )
                self.class_conv2 = nn.Sequential(nn.Conv2d(filters[1], 32, 3, 1, 1),  # 128
                                                 conv_act,
                                                 nn.Conv2d(32, 32, 3, padding=1),
                                                 conv_act,
                                                 self.dropblock,
                                                 nn.Conv2d(32, out_c, 3, padding=1)
                                                 )
                self.class_conv3 = nn.Sequential(nn.Conv2d(filters[0], 32, 3, 1, 1),  # 64
                                                 conv_act,
                                                 nn.Conv2d(32, 32, 3, padding=1),
                                                 conv_act,
                                                 self.dropblock,
                                                 nn.Conv2d(32, out_c, 3, padding=1)
                                                 )

                self.class_conv1_bin = nn.Sequential(
                    nn.Conv2d(filters[2], 32, 3, 1, 1),  # 256
                    conv_act,
                    nn.Conv2d(32, 32, 3, padding=1),
                    conv_act,
                    self.dropblock,
                    nn.Conv2d(32, 1, 3, padding=1),
                    nn.Sigmoid()
                )
                self.class_conv2_bin = nn.Sequential(nn.Conv2d(filters[1], 32, 3, 1, 1),  # 128
                                                 conv_act,
                                                 nn.Conv2d(32, 32, 3, padding=1),
                                                 conv_act,
                                                 self.dropblock,
                                                 nn.Conv2d(32, 1, 3, padding=1),
                                                 nn.Sigmoid()
                                                 )
                self.class_conv3_bin = nn.Sequential(nn.Conv2d(filters[0], 32, 3, 1, 1),  # 64
                                                 conv_act,
                                                 nn.Conv2d(32, 32, 3, padding=1),
                                                 conv_act,
                                                 self.dropblock,
                                                 nn.Conv2d(32, 1, 3, padding=1),
                                                 nn.Sigmoid()
                                                 )


            else:
                self.classifier1 = nn.Sequential(nn.Conv2d(filters[2], 32, 3, 1, 1),  # 256
                                                 conv_act,
                                                 nn.Conv2d(32, 32, 3, padding=1),
                                                 conv_act,
                                                 nn.Conv2d(32, out_c, 3, padding=1))
                self.classifier2 = nn.Sequential(nn.Conv2d(filters[1], 32, 3, 1, 1),  # 128
                                                 conv_act,
                                                 nn.Conv2d(32, 32, 3, padding=1),
                                                 conv_act,
                                                 nn.Conv2d(32, out_c, 3, padding=1))
                self.classifier3 = nn.Sequential(nn.Conv2d(filters[0], 32, 3, 1, 1),  # 64
                                                 conv_act,
                                                 nn.Conv2d(32, 32, 3, padding=1),
                                                 conv_act,
                                                 nn.Conv2d(32, out_c, 3, padding=1))

                self.classifier1_bin = nn.Sequential(nn.Conv2d(filters[2], 32, 3, 1, 1),  # 256
                                                 conv_act,
                                                 nn.Conv2d(32, 32, 3, padding=1),
                                                 conv_act,
                                                 nn.Conv2d(32, 1, 3, padding=1),
                                                 nn.Sigmoid())
                self.classifier2_bin = nn.Sequential(nn.Conv2d(filters[1], 32, 3, 1, 1),  # 128
                                                 conv_act,
                                                 nn.Conv2d(32, 32, 3, padding=1),
                                                 conv_act,
                                                 nn.Conv2d(32, 1, 3, padding=1),
                                                 nn.Sigmoid()
                                                     )
                self.classifier3_bin = nn.Sequential(nn.Conv2d(filters[0], 32, 3, 1, 1),  # 64
                                                 conv_act,
                                                 nn.Conv2d(32, 32, 3, padding=1),
                                                 conv_act,
                                                 nn.Conv2d(32, 1, 3, padding=1),
                                                 nn.Sigmoid())
        else:
            logger.info("using no deep supervsion...")



   def forward(self, x1, x2):
        if self.use_drop:
            self.dropblock.step()
        feat1_0,feat1_1,feat1_2,feat1_3,feat1_4=self.feat_Extactor(x1)
        feat2_0, feat2_1, feat2_2, feat2_3, feat2_4=self.feat_Extactor(x2)


        if self.diff_mode=='diff':
            feat12_0 = torch.abs(feat1_0 - feat2_0)  # [4,64,256,256]
            feat12_1 = torch.abs(feat1_1 - feat2_1)  # [4,64,128,128]
            feat12_2 = torch.abs(feat1_2 - feat2_2)  # [4,128,64,64]
            feat12_3 = torch.abs(feat1_3 - feat2_3)  # [4,256,32,32]
            feat12_4 = torch.abs(feat1_4 - feat2_4)  # [4,512,16,16]
        else:
            feat12_0=self.conv_cat01(torch.cat([feat1_0,feat2_0],dim=1))
            feat12_1 = self.conv_cat01(torch.cat([feat1_1, feat2_1], dim=1))
            feat12_2 = self.conv_cat2(torch.cat([feat1_2, feat2_2], dim=1))
            feat12_3 = self.conv_cat3(torch.cat([feat1_3, feat2_3], dim=1))
            feat12_4 = self.conv_cat4(torch.cat([feat1_4, feat2_4], dim=1))

        d4_1 = self.decoder4(feat12_4, feat12_3, feat1_3)  # #[4,256,16,16] 16
        d3_1 = self.decoder3(d4_1, feat12_2, feat1_2)  # [4,128,32,32] 32
        d2_1 = self.decoder2(d3_1, feat12_1, feat1_1)  # [4,64,64,64]64
        d1_1 = self.decoder1(d2_1, feat12_0, feat1_0)  # 128

        d4_2 = self.decoder4(feat12_4, feat12_3, feat2_3)  # 16
        d3_2 = self.decoder3(d4_2, feat12_2, feat2_2)  # 32
        d2_2 = self.decoder2(d3_2, feat12_1, feat2_1)  # 64
        d1_2 = self.decoder1(d2_2, feat12_0, feat2_0)  # 128

        d4_12 = self.decoder4_diff(feat12_4, feat12_3)  # 16
        d3_12 = self.decoder3_diff(d4_12, feat12_2)  # 32
        d2_12 = self.decoder2_diff(d3_12, feat12_1)  # 64
        d1_12 = self.decoder1_diff(d2_12, feat12_0)  # 128



        pred1 = self.class_conv(d1_1)
        pred2 = self.class_conv(d1_2)
        pred12=self.class_conv_bin(d1_12)
        if self.training_mode:
            if self.use_DS:

                pred1_0, pred2_0,pred12_0 = self.class_conv1(d4_1), self.class_conv1(d4_2),self.class_conv1_bin(d4_12)
                pred1_1, pred2_1,pred12_1 = self.class_conv2(d3_1), self.class_conv2(d3_2),self.class_conv2_bin(d3_12)
                pred1_2, pred2_2,pred12_2 = self.class_conv3(d2_1), self.class_conv3(d2_2),self.class_conv3_bin(d2_12)

                return (pred1_0, pred2_0, pred12_0), (pred1_1, pred2_1,pred12_1), (pred1_2, pred2_2, pred12_2), (pred1, pred2, pred12)

        return  pred12,pred1,pred2


class EDCls_UNet2_MC6Bin(nn.Module):

   def __init__(self, in_c=3, use_att=False,diff_mode='diff', training_mode=True,att_mode='BAM',act_mode='relu',nf=64,out_c=7,net_feat=None,use_DS=False,use_CatOut=False,cat_mode='cat_feat',use_se=False,drop_rate=0.1,se_block='BAM'):
        super(EDCls_UNet2_MC6Bin, self).__init__()
        self.feat_Extactor=net_feat
        filters=[64,128,256,512]
        if act_mode=='relu':
            conv_act=nn.ReLU(inplace=True)
        else:
            conv_act=nn.LeakyReLU(0.2, inplace=True)

        self.use_att = use_att
        self.use_CatOut=use_CatOut
        self.cat_mode = cat_mode
        self.training_mode=training_mode


        if use_att:#=====can be removed

            deconv_att=True
            logger.info("use att for decoder...")
        else:
            deconv_att=False
            logger.info("no att for decoder...")

        self.use_drop = False
        if drop_rate>0:
            self.use_drop=True
            message="using dropblock with rate {:.3f} ".format(drop_rate)
            logger.info(message)

            self.dropblock = LinearScheduler(
                DropBlock2D(drop_prob=drop_rate, block_size=7),
                start_value=0,
                stop_value=drop_rate,
                nr_steps=10000
            )

        #==========================================================
        logger.info("use new residual for decoder...")
        # self.decoder4 = unetUp3(filters[2] * 2, filters[2], use_att=deconv_att, act_mode=act_mode,
        #                                 use_se=use_se)
        # self.decoder3 = unetUp3(filters[1] * 2, filters[1], use_att=deconv_att, act_mode=act_mode,
        #                                 use_se=use_se)
        # self.decoder2 = unetUp3(filters[0] * 2, filters[0], use_att=deconv_att, act_mode=act_mode,
        #                                 use_se=use_se)
        # self.decoder1 = unetUp3(filters[0], filters[0], use_att=deconv_att, act_mode=act_mode,
                                        #use_se=use_se)
        self.decoder4= unetUp2(filters[2] * 2, filters[2], use_att=deconv_att, act_mode=act_mode)
        self.decoder3= unetUp2(filters[1] * 2, filters[1], use_att=deconv_att, act_mode=act_mode)
        self.decoder2= unetUp2(filters[0] * 2, filters[0], use_att=deconv_att, act_mode=act_mode)
        self.decoder1= unetUp2(filters[0], filters[0], use_att=deconv_att, act_mode=act_mode)

        self.decoder4_diff = unetUp2(filters[2] * 2, filters[2], use_att=deconv_att, act_mode=act_mode)
        self.decoder3_diff = unetUp2(filters[1] * 2, filters[1], use_att=deconv_att, act_mode=act_mode)
        self.decoder2_diff = unetUp2(filters[0] * 2, filters[0], use_att=deconv_att, act_mode=act_mode)
        self.decoder1_diff = unetUp2(filters[0], filters[0], use_att=deconv_att, act_mode=act_mode)

        self.class_conv = nn.Sequential(nn.ConvTranspose2d(64, 32, 4, 2, 1),  # size*2
                                        conv_act,
                                        nn.Conv2d(32, 32, 3, padding=1),
                                        conv_act,
                                        self.dropblock,
                                        nn.Conv2d(32, out_c, 3, padding=1)
                                        )

        self.class_conv_bin = nn.Sequential(nn.ConvTranspose2d(64, 32, 4, 2, 1),  # size*2
                                        conv_act,
                                        nn.Conv2d(32, 32, 3, padding=1),
                                        conv_act,
                                        self.dropblock,
                                        nn.Conv2d(32, 1, 3, padding=1),
                                        nn.Sigmoid()
                                        )

        self.diff_mode=diff_mode
        logger.info("diff mode is {}".format(diff_mode))
        if diff_mode=='cat':
            self.conv_cat01=nn.Conv2d(filters[0]*2,filters[0],1)
            self.conv_cat2=nn.Conv2d(filters[1]*2,filters[1],1)
            self.conv_cat3 = nn.Conv2d(filters[2] * 2, filters[2], 1)
            self.conv_cat4 = nn.Conv2d(filters[3] * 2, filters[3], 1)
        self.use_DS=use_DS
        if self.use_DS:
            logger.info("using deep supervsion...")
            if self.use_drop:
                self.class_conv1 = nn.Sequential(
                                                 nn.Conv2d(filters[2], 32, 3, 1, 1),  # 256
                                                 conv_act,
                                                 nn.Conv2d(32, 32, 3, padding=1),
                                                 conv_act,
                                                 self.dropblock,
                                                 nn.Conv2d(32, out_c, 3, padding=1)
                                                 )
                self.class_conv2 = nn.Sequential(nn.Conv2d(filters[1], 32, 3, 1, 1),  # 128
                                                 conv_act,
                                                 nn.Conv2d(32, 32, 3, padding=1),
                                                 conv_act,
                                                 self.dropblock,
                                                 nn.Conv2d(32, out_c, 3, padding=1)
                                                 )
                self.class_conv3 = nn.Sequential(nn.Conv2d(filters[0], 32, 3, 1, 1),  # 64
                                                 conv_act,
                                                 nn.Conv2d(32, 32, 3, padding=1),
                                                 conv_act,
                                                 self.dropblock,
                                                 nn.Conv2d(32, out_c, 3, padding=1)
                                                 )

                self.class_conv1_bin = nn.Sequential(
                    nn.Conv2d(filters[2], 32, 3, 1, 1),  # 256
                    conv_act,
                    nn.Conv2d(32, 32, 3, padding=1),
                    conv_act,
                    self.dropblock,
                    nn.Conv2d(32, 1, 3, padding=1),
                    nn.Sigmoid()
                )
                self.class_conv2_bin = nn.Sequential(nn.Conv2d(filters[1], 32, 3, 1, 1),  # 128
                                                 conv_act,
                                                 nn.Conv2d(32, 32, 3, padding=1),
                                                 conv_act,
                                                 self.dropblock,
                                                 nn.Conv2d(32, 1, 3, padding=1),
                                                 nn.Sigmoid()
                                                 )
                self.class_conv3_bin = nn.Sequential(nn.Conv2d(filters[0], 32, 3, 1, 1),  # 64
                                                 conv_act,
                                                 nn.Conv2d(32, 32, 3, padding=1),
                                                 conv_act,
                                                 self.dropblock,
                                                 nn.Conv2d(32, 1, 3, padding=1),
                                                 nn.Sigmoid()
                                                 )


            else:
                self.classifier1 = nn.Sequential(nn.Conv2d(filters[2], 32, 3, 1, 1),  # 256
                                                 conv_act,
                                                 nn.Conv2d(32, 32, 3, padding=1),
                                                 conv_act,
                                                 nn.Conv2d(32, out_c, 3, padding=1))
                self.classifier2 = nn.Sequential(nn.Conv2d(filters[1], 32, 3, 1, 1),  # 128
                                                 conv_act,
                                                 nn.Conv2d(32, 32, 3, padding=1),
                                                 conv_act,
                                                 nn.Conv2d(32, out_c, 3, padding=1))
                self.classifier3 = nn.Sequential(nn.Conv2d(filters[0], 32, 3, 1, 1),  # 64
                                                 conv_act,
                                                 nn.Conv2d(32, 32, 3, padding=1),
                                                 conv_act,
                                                 nn.Conv2d(32, out_c, 3, padding=1))

                self.classifier1_bin = nn.Sequential(nn.Conv2d(filters[2], 32, 3, 1, 1),  # 256
                                                 conv_act,
                                                 nn.Conv2d(32, 32, 3, padding=1),
                                                 conv_act,
                                                 nn.Conv2d(32, 1, 3, padding=1),
                                                 nn.Sigmoid())
                self.classifier2_bin = nn.Sequential(nn.Conv2d(filters[1], 32, 3, 1, 1),  # 128
                                                 conv_act,
                                                 nn.Conv2d(32, 32, 3, padding=1),
                                                 conv_act,
                                                 nn.Conv2d(32, 1, 3, padding=1),
                                                 nn.Sigmoid()
                                                     )
                self.classifier3_bin = nn.Sequential(nn.Conv2d(filters[0], 32, 3, 1, 1),  # 64
                                                 conv_act,
                                                 nn.Conv2d(32, 32, 3, padding=1),
                                                 conv_act,
                                                 nn.Conv2d(32, 1, 3, padding=1),
                                                 nn.Sigmoid())
        else:
            logger.info("using no deep supervsion...")



   def forward(self, x1, x2):
        if self.use_drop:
            self.dropblock.step()
        feat1_0,feat1_1,feat1_2,feat1_3,feat1_4=self.feat_Extactor(x1)
        feat2_0, feat2_1, feat2_2, feat2_3, feat2_4=self.feat_Extactor(x2)


        if self.diff_mode=='diff':
            feat12_0 = torch.abs(feat1_0 - feat2_0)  # [4,64,256,256]
            feat12_1 = torch.abs(feat1_1 - feat2_1)  # [4,64,128,128]
            feat12_2 = torch.abs(feat1_2 - feat2_2)  # [4,128,64,64]
            feat12_3 = torch.abs(feat1_3 - feat2_3)  # [4,256,32,32]
            feat12_4 = torch.abs(feat1_4 - feat2_4)  # [4,512,16,16]
        else:
            feat12_0=self.conv_cat01(torch.cat([feat1_0,feat2_0],dim=1))
            feat12_1 = self.conv_cat01(torch.cat([feat1_1, feat2_1], dim=1))
            feat12_2 = self.conv_cat2(torch.cat([feat1_2, feat2_2], dim=1))
            feat12_3 = self.conv_cat3(torch.cat([feat1_3, feat2_3], dim=1))
            feat12_4 = self.conv_cat4(torch.cat([feat1_4, feat2_4], dim=1))

        d4_1 = self.decoder4(feat1_4,feat1_3)  # #[4,256,16,16] 16
        d3_1 = self.decoder3(d4_1, feat1_2)  # [4,128,32,32] 32
        d2_1 = self.decoder2(d3_1, feat1_1)  # [4,64,64,64]64
        d1_1 = self.decoder1(d2_1, feat1_0)  # 128

        d4_2 = self.decoder4(feat2_4, feat2_3)  # 16
        d3_2 = self.decoder3(d4_2,  feat2_2)  # 32
        d2_2 = self.decoder2(d3_2,feat2_1)  # 64
        d1_2 = self.decoder1(d2_2, feat2_0)  # 128

        d4_12 = self.decoder4_diff(feat12_4, feat12_3)  # 16
        d3_12 = self.decoder3_diff(d4_12, feat12_2)  # 32
        d2_12 = self.decoder2_diff(d3_12, feat12_1)  # 64
        d1_12 = self.decoder1_diff(d2_12, feat12_0)  # 128

        pred1 = self.class_conv(d1_1)
        pred2 = self.class_conv(d1_2)
        pred12=self.class_conv_bin(d1_12)
        if self.training_mode:
            if self.use_DS:

                pred1_0, pred2_0,pred12_0 = self.class_conv1(d4_1), self.class_conv1(d4_2),self.class_conv1_bin(d4_12)
                pred1_1, pred2_1,pred12_1 = self.class_conv2(d3_1), self.class_conv2(d3_2),self.class_conv2_bin(d3_12)
                pred1_2, pred2_2,pred12_2 = self.class_conv3(d2_1), self.class_conv3(d2_2),self.class_conv3_bin(d2_12)

                return (pred1_0, pred2_0, pred12_0), (pred1_1, pred2_1,pred12_1), (pred1_2, pred2_2, pred12_2), (pred1, pred2, pred12)

        return  pred1,pred2, pred12


class SCDNet(nn.Module):
    def __init__(self,use_DS=True):
        super(SCDNet, self).__init__()
        self.use_DS = use_DS
        self.encoder =unet_2D_Encoder(use_se=True)
        #self.SCD=EDCls_UNet2_MC7Bin(use_DS=use_DS,use_att=True,net_feat=self.encoder)
        #self.SCD=EDCls_UNet2_MC6Bin(use_DS=use_DS,use_att=True,net_feat=self.encoder)
        self.SCD=EDCls_UNet2_New2(use_DS=use_DS,use_att=True,net_feat=self.encoder)
    def forward(self, input):
        self.SCD.dropblock.step()
        dim = len(input.size())
        if (dim != 4):
            input = torch.unsqueeze(input, dim=0)
        (x1,x2)=torch.chunk(input,2,dim=1)
        if self.use_DS:
            pred1,pred2,pred3,out1,out2 = self.SCD(x1,x2)
            return pred1,pred2,pred3,out1,out2
        else:
            pred1,pred2 = self.SCD(x1,x2)
            return pred1,pred2

from thop import profile
net=SCDNet()

x2=torch.randn((1,3,512,512))
x1=torch.randn((1,3,512,512))

#获取浮点数总数
# stat(net, input_size=(6,512,512))
print("----------------------------------------------------")
print("----------------------------------------------------")
# # summary(net,input_size=(6,512,512),device="cpu")
# # print("----------------------------------------------------")
# # print("----------------------------------------------------")
flops, params = profile(net,(torch.cat([x1,x2],dim=1)))
print("Total parameters: {:.2f}Mb".format(params / 1e6))
print("Total flops: {:.2f}Gbps".format(flops / 1e9))