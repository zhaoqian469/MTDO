
from modules.base_networks import *
from modules.HAT_arc import *
import torch
import torch.nn.functional as F


# Adjacent Time Difference Module #
class ATDM(nn.Module):
    def __init__(self, nframes, apha=0.5, belta=0.5, nres_b=1):
        super(ATDM, self).__init__()

        self.nframes = nframes
        self.apha = apha
        self.belta = belta
        base_filter = 128  # bf
        self.feat0 = ConvBlock(3, base_filter, 3, 1, 1, activation='prelu', norm=None)  # h*w*3-->h*w*base_filter
        self.feat_diff = ConvBlock(3, 64, 3, 1, 1, activation='prelu', norm=None)  # h*w*3 --> h*w*64
        self.conv1 = ConvBlock((self.nframes-1)*64, base_filter, 3, 1, 1, activation='prelu', norm=None)

        # Res-Block2,h*w*bf-->h*w*64
        modules_body1 = [
            ResnetBlock(base_filter, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu', norm=None) \
            for _ in range(nres_b)]
        modules_body1.append(ConvBlock(base_filter, 64, 3, 1, 1, activation='prelu', norm=None))
        self.res_feat1 = nn.Sequential(*modules_body1)

        # Res-Block1,h*w*bf-->H*W*64
        modules_body2 = [
            ResnetBlock(base_filter, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu', norm=None) \
            for _ in range(nres_b)]
        modules_body2.append(ConvBlock(base_filter, 64, 3, 1, 1, activation='prelu', norm=None))
        self.res_feat2 = nn.Sequential(*modules_body2)

        self.avg_diff = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, lr, neigbor):
        lr_id = self.nframes // 2
        neigbor.insert(lr_id, lr)
        frame_list = neigbor
        rgb_diff = []

        # Adjacent Time Difference Map
        for i in range(self.nframes-1):
            rgb_diff.append(frame_list[i] - frame_list[i+1])
        rgb_diff = torch.stack(rgb_diff, dim=1)
        B, N, C, H, W = rgb_diff.size()  # [1, nframe-1, 3, 160, 160]
        lr_f0 = self.feat0(lr)  # h*w*3 --> h*w*128
        diff_f = self.feat_diff(rgb_diff.contiguous().view(-1, C, H, W))  # reshape, (4,4,3,64,64)-->(16,3,64,64)-->(16,64,64,64)
        down_diff_f = self.avg_diff(diff_f).contiguous().view(B, N, -1, H//2, W//2)  # downsampling,(16,64,64,64)-->(16,64,32,32)-->[4,4,64,32,32]

        stack_diff = []
        for j in range(N):
            stack_diff.append(down_diff_f[:, j, :, :, :])
        stack_diff = torch.cat(stack_diff, dim=1)
        stack_diff = self.conv1(stack_diff)

        #ResBlock2
        up_diff1 = self.res_feat1(stack_diff)
        #upsample
        up_diff1 = F.interpolate(up_diff1, scale_factor=2, mode='bilinear', align_corners=True)
        up_diff2 = F.interpolate(stack_diff, scale_factor=2, mode='bilinear', align_corners=True)
        compen_lr = self.apha * lr_f0 + self.belta * up_diff2
        #ResBlock1
        compen_lr = self.res_feat2(compen_lr)

        compen_lr = self.apha * compen_lr + self.belta * up_diff1
        return compen_lr

# Skipped Temporal Difference Module #
class STDM(nn.Module):
    def __init__(self, nframes, apha=0.5, belta=0.5, nres_b=1):
        super(STDM, self).__init__()

        self.nframes = nframes
        self.apha = apha
        self.belta = belta
        base_filter = 128  # bf
        self.feat0 = ConvBlock(3, base_filter, 3, 1, 1, activation='prelu', norm=None)  # h*w*3-->h*w*base_filter
        self.feat_diff = ConvBlock(3, 64, 3, 1, 1, activation='prelu', norm=None)  # h*w*3 --> h*w*64
        self.conv1_2 = ConvBlock((self.nframes - 3) * 64, base_filter, 3, 1, 1, activation='prelu', norm=None)

        # Res-Block2,h*w*bf-->h*w*64
        modules_body1 = [
            ResnetBlock(base_filter, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu', norm=None) \
            for _ in range(nres_b)]
        modules_body1.append(ConvBlock(base_filter, 64, 3, 1, 1, activation='prelu', norm=None))
        self.res_feat1 = nn.Sequential(*modules_body1)

        # Res-Block1,h*w*bf-->H*W*64
        modules_body2 = [
            ResnetBlock(base_filter, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu', norm=None) \
            for _ in range(nres_b)]
        modules_body2.append(ConvBlock(base_filter, 64, 3, 1, 1, activation='prelu', norm=None))
        self.res_feat2 = nn.Sequential(*modules_body2)
        self.avg_diff = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, lr, neigbor):
        frame_list = neigbor
        rgb_diff2 = []

        # Skipped Temporal Difference Map #
        for m in range(0, self.nframes - 1, 2):
            rgb_diff2.append(frame_list[m] - frame_list[m + 2])
        rgb_diff2 = torch.stack(rgb_diff2, dim=1)  #(4,3,64,64)-->(4,2,3,64,64)
        B, N, C, H, W = rgb_diff2.size()  # [1, nframe-1, 3, 160, 160]

        lr_f0 = self.feat0(lr)  # h*w*3 --> h*w*128
        diff_f2 = self.feat_diff(rgb_diff2.contiguous().view(-1, C, H,
                                                           W))  # reshape, (4,2,3,64,64)-->(8,3,64,64)-->(8,64,64,64)
        down_diff_f2 = self.avg_diff(diff_f2).contiguous().view(B, N, -1, H // 2,
                                                              W // 2)  # downsampling,(8,64,64,64)-->(8,64,32,32)-->(4,2,64,32,32)

        stack_diff2 = []
        for n in range(N):
            stack_diff2.append(down_diff_f2[:, n, :, :, :])
        stack_diff2 = torch.cat(stack_diff2, dim=1)  # (4,64,32,32)-->(4,128,32,32)
        stack_diff2 = self.conv1_2(stack_diff2)   #(4,128,32,32)-->(4,128,32,32)

        # ResBlock2
        up_diff1_add = self.res_feat1(stack_diff2)
        # upsample
        up_diff1_add = F.interpolate(up_diff1_add, scale_factor=2, mode='bilinear', align_corners=True)
        up_diff2_add = F.interpolate(stack_diff2, scale_factor=2, mode='bilinear', align_corners=True)

        compen_lr2 = self.apha * lr_f0 + self.belta * up_diff2_add

        # ResBlock1
        compen_lr2 = self.res_feat2(compen_lr2)

        compen_lr2 = self.apha * compen_lr2 + self.belta * up_diff1_add

        return compen_lr2

# Cross Temporal Difference Module
class CTDM(nn.Module):
    def __init__(self, nframes):
        super(CTDM, self).__init__()

        self.nframes = nframes
        base_filter = 64

        self.compress_3 = ConvBlock(self.nframes*64, base_filter, 3, 1, 1, activation='prelu', norm=None)  # h*w*3-->h*w*256
        self.conv1 = ConvBlock(base_filter, base_filter, 3, 1, 1, activation='prelu', norm=None)

    def forward(self, frame_fea_list):
        frame_list_reverse = frame_fea_list
        frame_list_reverse.reverse()  # [[B,64,h,w], ..., ]   reverse

        # forward
        forward_fea3 = self.conv1(self.compress_3(torch.cat(frame_fea_list, 1)))
        # backward
        backward_fea3 = self.conv1(self.compress_3(torch.cat(frame_list_reverse, 1)))
        # res
        forward_diff_fea3 = forward_fea3 - backward_fea3
        backward_diff_fea3 = backward_fea3 - forward_fea3

        return forward_diff_fea3, backward_diff_fea3

# Temporal Differences-Guided Dynamic Routing Optimization Module #
class T_DROM(nn.Module):
    def __init__(self, nframes,apha = 0.5, belta = 0.5):
        super(T_DROM, self).__init__()

        self.nframes = nframes
        self.apha = apha
        self.belta = belta
        base_filter = 128
        base_filter2 = 64

        self.feat0 = ConvBlock(3, base_filter, 3, 1, 1, activation='prelu', norm=None)  # h*w*3-->h*w*base_filter
        self.conv2 = ConvBlock(192, 192, 3, 1, 1, activation='prelu', norm=None)
        self.conv3 = ConvBlock(192, 192, 3, 1, 1, activation='prelu', norm=None)
        self.conv4 = ConvBlock(192, 64, 3, 1, 1, activation='prelu', norm=None)
        self.conv5 = ConvBlock(base_filter2, base_filter2, 3, 1, 1, activation='prelu', norm=None)
        self.conv6 = ConvBlock(base_filter2, base_filter2, 3, 1, 1, activation='prelu', norm=None)
        self.conv7 = ConvBlock(base_filter2, self.nframes*64, 3, 1, 1, activation='prelu', norm=None)
        self.sigmoid = nn.Sigmoid()
        self.avg_diff = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, lr ,A_compen_x, S_compen_x, C_compen_x_f, C_compen_x_b, frame_fea_list):
        lr_f0 = self.feat0(lr)
        stack_A_compen_x = torch.cat([A_compen_x, lr_f0], dim=1)
        stack_S_compen_x = torch.cat([S_compen_x, lr_f0], dim=1)

        # Route1
        id_f1 = stack_A_compen_x
        enhance_f1 = self.conv4(id_f1)
        f1 = enhance_f1

        #Route2
        # MFEU(Multi-Scale Feature Enhancement Unit)
        id_f3 = stack_S_compen_x
        pool_f3 = self.conv3(self.avg_diff(stack_S_compen_x))
        up_f3 = F.interpolate(pool_f3, scale_factor=2, mode='bilinear', align_corners=True)
        enhance_f3 = self.conv2(stack_S_compen_x)
        f3 = self.conv4(id_f3 + enhance_f3 + up_f3)

        # Rout1+2
        F_Rout1_2 = self.apha * f1 + self.belta * f3

        #Rout3
        frame_fea = torch.cat(frame_fea_list, 1)
        #MFEU(Multi-Scale Feature Enhancement Unit)
        id_f3 = C_compen_x_f
        id_b3 = C_compen_x_b
        pool_f3 = self.conv6(self.avg_diff(C_compen_x_f))
        up_f3 = F.interpolate(pool_f3, scale_factor=2, mode='bilinear', align_corners=True)
        pool_b3 = self.conv6(self.avg_diff(C_compen_x_b))
        up_b3 = F.interpolate(pool_b3, scale_factor=2, mode='bilinear', align_corners=True)
        enhance_f3 = self.conv5(C_compen_x_f)
        enhance_b3 = self.conv5(C_compen_x_b)
        f3 = self.sigmoid(self.conv7(id_f3 + enhance_f3 + up_f3))
        b3 = self.sigmoid(self.conv7(id_b3 + enhance_b3 + up_b3))

        att3 = f3 + b3

        F_Rout3 = att3 * frame_fea + frame_fea

        return F_Rout1_2, F_Rout3

class TSA_Fusion(nn.Module):
    ''' Temporal Spatial Attention fusion module
    Temporal: correlation;
    Spatial: 3 pyramid levels.
    '''

    def __init__(self, nf=64, nframes=5, center=2):
        super(TSA_Fusion, self).__init__()
        self.center = center
        # temporal attention (before fusion conv)
        self.tAtt_1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.tAtt_2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        # fusion conv: using 1x1 to save parameters and computation
        self.fea_fusion = nn.Conv2d(nframes * nf, nf, 1, 1, bias=True)

        # spatial attention (after fusion conv)
        self.sAtt_1 = nn.Conv2d(nframes * nf, nf, 1, 1, bias=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.avgpool = nn.AvgPool2d(3, stride=2, padding=1)
        self.sAtt_2 = nn.Conv2d(nf * 2, nf, 1, 1, bias=True)
        self.sAtt_3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.sAtt_4 = nn.Conv2d(nf, nf, 1, 1, bias=True)
        self.sAtt_5 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.sAtt_L1 = nn.Conv2d(nf, nf, 1, 1, bias=True)
        self.sAtt_L2 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)
        self.sAtt_L3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.sAtt_add_1 = nn.Conv2d(nf, nf, 1, 1, bias=True)
        self.sAtt_add_2 = nn.Conv2d(nf, nf, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, aligned_fea):
        B, N, C, H, W = aligned_fea.size()  # N video frames
        #### temporal attention
        emb_ref = self.tAtt_2(aligned_fea[:, self.center, :, :, :].clone())
        emb = self.tAtt_1(aligned_fea.contiguous().view(-1, C, H, W)).contiguous().view(B, N, -1, H, W)

        cor_l = []
        for i in range(N):
            emb_nbr = emb[:, i, :, :, :]
            cor_tmp = torch.sum(emb_nbr * emb_ref, 1).unsqueeze(1)
            cor_l.append(cor_tmp)
        cor_prob = torch.sigmoid(torch.cat(cor_l, dim=1))  # B, N, H, W
        cor_prob = cor_prob.unsqueeze(2).repeat(1, 1, C, 1, 1).contiguous().view(B, -1, H, W)
        aligned_fea = aligned_fea.contiguous().view(B, -1, H, W) * cor_prob

        #### fusion
        fea = self.lrelu(self.fea_fusion(aligned_fea))

        #### spatial attention
        att = self.lrelu(self.sAtt_1(aligned_fea))
        att_max = self.maxpool(att)
        att_avg = self.avgpool(att)
        att = self.lrelu(self.sAtt_2(torch.cat([att_max, att_avg], dim=1)))
        # pyramid levels
        att_L = self.lrelu(self.sAtt_L1(att))
        att_max = self.maxpool(att_L)
        att_avg = self.avgpool(att_L)
        att_L = self.lrelu(self.sAtt_L2(torch.cat([att_max, att_avg], dim=1)))
        att_L = self.lrelu(self.sAtt_L3(att_L))
        att_L = F.interpolate(att_L, scale_factor=2, mode='bilinear', align_corners=False)

        att = self.lrelu(self.sAtt_3(att))
        att = att + att_L
        att = self.lrelu(self.sAtt_4(att))
        att = F.interpolate(att, scale_factor=2, mode='bilinear', align_corners=False)
        att = self.sAtt_5(att)
        att_add = self.sAtt_add_2(self.lrelu(self.sAtt_add_1(att)))
        att = torch.sigmoid(att)

        fea = fea * att * 2 + att_add
        return fea

class myNet(nn.Module):
    def __init__(self, nframes):
        super(myNet, self).__init__()
        self.nframes = nframes
        self.lr_idx = self.nframes // 2
        self.apha = 0.5
        self.belta = 0.5

        self.fea0 = ConvBlock(3, 64, 3, 1, 1, activation='prelu', norm=None)
        self.fea_all = ConvBlock(3, 64, 3, 1, 1, activation='prelu', norm=None)
        feature_extraction = [
            ResnetBlock(64, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu', norm=None) \
            for _ in range(5)]
        self.res_feat_ext = nn.Sequential(*feature_extraction)

        self.A_tdm = ATDM(nframes=self.nframes, apha=self.apha, belta=self.belta)
        self.S_tdm = STDM(nframes=self.nframes, apha=self.apha, belta=self.belta)
        self.C_tdm = CTDM(nframes=self.nframes)
        self.T_drom =T_DROM(nframes=self.nframes, apha=self.apha, belta=self.belta)
        self.fus = nn.Conv2d(64*self.nframes, 64, 3, 1, 1)
        self.msd = MSD()
        self.TSA_Fusion = TSA_Fusion(64, nframes=self.nframes, center=self.lr_idx)
        self.hat = HAT()

        # ResBlock2
        modules_body2 = [
            ResnetBlock(64, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu', norm=None) \
            for _ in range(5)]
        self.res_feat2 = nn.Sequential(*modules_body2)

        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv2d') != -1:
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif classname.find('ConvTranspose2d') != -1:
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()


    def forward(self, x, neigbors):
        B, C, H, W = x.size()
        fea_x = self.fea0(x)

        A_compen_x = self.A_tdm(x, neigbors)
        S_compen_x = self.S_tdm(x, neigbors)

        frame_all = neigbors
        feat_all = torch.stack(frame_all, dim=1)  # (4,3,64,64)-->(4,5,3,64,64)
        feat_all = self.fea_all(feat_all.contiguous().view(-1, C, H, W))  # (20,64,64,64)
        feat_all = self.res_feat_ext(feat_all)
        feat_all = feat_all.contiguous().view(B, self.nframes, -1, H, W)  # [4, 5, 64, 64, 64]

        # MSD
        aligned_fea = []
        ref_fea = feat_all[:, self.lr_idx, :, :, :]
        for i in range(self.nframes):
            neigbor_fea = feat_all[:, i, :, :, :]
            aligned_fea.append(self.msd(neigbor_fea, ref_fea))

        frame_fea_list = []  # [b 64 ps ps] * n
        for i in range(self.nframes):
            frame_fea_list.append(aligned_fea[i])

        C_compen_x_f, C_compen_x_b = self.C_tdm(frame_fea_list)
        F_Rout1_2, F_Rout3 = self.T_drom(x,A_compen_x, S_compen_x, C_compen_x_f, C_compen_x_b,frame_fea_list)
        F_Rout3 = self.fus(F_Rout3)  # [b 64 ps ps]

        aligned_fea_all = torch.stack(aligned_fea, dim=1)  # (4,5,64,64,64)
        fea = self.TSA_Fusion(aligned_fea_all)  # [b 64 ps ps]

        # Motion Alignment and Correction Unit(MACU) #
        res = fea - F_Rout3
        res = self.res_feat2(res)
        fea = fea + res

        fea = fea + F_Rout1_2 + fea_x

        final = self.hat(fea)

        return final