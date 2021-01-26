import torch
import torch.nn as nn
import torch.nn.functional as F


def L2(f_):
    return (((f_ ** 2).sum(dim=1)) ** 0.5).reshape(f_.shape[0], 1, f_.shape[2], f_.shape[3]) + 1e-8


def similarity(feat):
    feat = feat.float()
    tmp = L2(feat).detach()
    feat = feat / tmp
    feat = feat.reshape(feat.shape[0], feat.shape[1], -1)
    return torch.einsum('icm,icn->imn', [feat, feat])


def sim_dis_compute(f_S, f_T):
    sim_err = ((similarity(f_T) - similarity(f_S)) ** 2) / ((f_T.shape[-1] * f_T.shape[-2]) ** 2) / f_T.shape[0]
    sim_dis = sim_err.sum()
    return sim_dis


class CriterionL2(nn.Module):
    def __init__(self):
        super(CriterionL2, self).__init__()
        self.mseloss = nn.MSELoss()

    def forward(self, preds_S, preds_T):
        preds_T[0].detach()
        N, C, W, H = preds_S.shape
        loss = self.mseloss(preds_S, preds_T)
        return loss


class CriterionJSDPixelWise(nn.Module):
    def __init__(self):
        super(CriterionJSDPixelWise, self).__init__()
        self.KLDivLoss = nn.KLDivLoss(reduction='batchmean')

    def forward(self, preds_S, preds_T):
        preds_T.detach()
        N, C, W, H = preds_S.shape
        loss_pixel = 0
        for i in range(N):
            softmax_pred_T = F.softmax(preds_T[i].unsqueeze(0).permute(0, 2, 3, 1).contiguous().view(-1, C), dim=1)
            softmax_pred_S = F.softmax(preds_S[i].unsqueeze(0).permute(0, 2, 3, 1).contiguous().view(-1, C), dim=1)
            log_mean_output = ((softmax_pred_T + softmax_pred_S) / 2).log()
            loss = (self.KLDivLoss(log_mean_output, softmax_pred_T) + self.KLDivLoss(log_mean_output,
                                                                                     softmax_pred_S)) / 2
            loss_pixel = loss_pixel + loss
        return loss_pixel / N


class CriterionHLDPixelWise(nn.Module):
    def __init__(self):
        super(CriterionHLDPixelWise, self).__init__()

    def forward(self, preds_S, preds_T):
        preds_T.detach()
        N, C, W, H = preds_S.shape
        for i in range(N):
            softmax_pred_T = F.softmax(preds_T[i].unsqueeze(0).permute(0, 2, 3, 1).contiguous().view(-1, C), dim=1)
            softmax_pred_S = F.softmax(preds_S[i].unsqueeze(0).permute(0, 2, 3, 1).contiguous().view(-1, C), dim=1)
            loss = torch.sqrt(1 - torch.sum(torch.sqrt(softmax_pred_S * softmax_pred_T)))
            loss_pixel = loss_pixel + loss
        return loss_pixel


class CriterionWassPixelWise(nn.Module):
    def __init__(self):
        super(CriterionWassPixelWise, self).__init__()

    def forward(self, preds_S, preds_T):
        preds_T.detach
        N, C, W, H = preds_S.shape

        softmax_pred_T = F.softmax(preds_T.permute(0, 2, 3, 1).contiguous().view(-1, C), dim=1)
        softmax_pred_S = F.softmax(preds_S.permute(0, 2, 3, 1).contiguous().view(-1, C), dim=1)
        meanT = - torch.mean(softmax_pred_T, dim=0)
        meanS = torch.mean(softmax_pred_S, dim=0)
        loss = torch.sum(meanT + meanS)
        return loss


class CriterionCEPixelWise(nn.Module):
    def __init__(self):
        super(CriterionCEPixelWise, self).__init__()

    def forward(self, preds_S, preds_T):
        preds_T[0].detach()
        assert preds_S[0].shape == preds_T[0].shape, 'the output dim of teacher and student differ'
        N, C, W, H = preds_S.shape
        loss_pixel = 0
        for i in range(N):
            softmax_pred_T = F.softmax(preds_T[i].unsqueeze(0).permute(0, 2, 3, 1).contiguous().view(-1, C), dim=1)
            logsoftmax = nn.LogSoftmax(dim=1)
            loss = (torch.sum(
                - softmax_pred_T * logsoftmax(
                    preds_S[i].unsqueeze(0).permute(0, 2, 3, 1).contiguous().view(-1, C)))) / W / H
            loss_pixel = loss_pixel + loss
        return loss_pixel / N


class CriterionKLPixelWise(nn.Module):
    def __init__(self):
        super(CriterionKLPixelWise, self).__init__()
        self.klloss = nn.KLDivLoss(reduction='batchmean')
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, preds_S, preds_T):
        preds_T[0].detach()
        assert preds_S[0].shape == preds_T[0].shape, 'the output dim of teacher and student differ'
        N, C, W, H = preds_S.shape
        loss_pixel = 0
        for i in range(N):
            softmax_pred_T = F.softmax(preds_T[i].unsqueeze(0).permute(0, 2, 3, 1).contiguous().view(-1, C), dim=1)
            softmax_pred_S = self.logsoftmax(preds_S[i].unsqueeze(0).permute(0, 2, 3, 1).contiguous().view(-1, C))
            loss = self.klloss(softmax_pred_S, softmax_pred_T)
            loss_pixel = loss_pixel + loss
        return loss_pixel / N


class CriterionPairWise(nn.Module):
    def __init__(self, scale=0.5):
        '''inter pair-wise loss from inter feature maps'''
        super(CriterionPairWise, self).__init__()
        self.criterion = sim_dis_compute
        self.scale = scale

    def forward(self, preds_S, preds_T):
        feat_S = preds_S
        feat_T = preds_T
        feat_T.detach()

        total_w, total_h = feat_T.shape[2], feat_T.shape[3]
        patch_w, patch_h = int(total_w * self.scale), int(total_h * self.scale)
        maxpool = nn.MaxPool2d(kernel_size=(patch_w, patch_h), stride=(patch_w, patch_h), padding=0,
                               ceil_mode=True)  # change
        loss = self.criterion(maxpool(feat_S), maxpool(feat_T))
        return loss


class CriterionFeatureCorrelation(nn.Module):
    def __init__(self, poolsize):
        super(CriterionFeatureCorrelation, self).__init__()
        self.poolsize = poolsize
        self.criterion = nn.L1Loss()
        self.total_l1 = 0

    def forward(self, preds_S, preds_T):
        featS_ = F.adaptive_avg_pool2d(preds_S, (self.poolsize, self.poolsize))
        featT_ = F.adaptive_avg_pool2d(preds_T, (self.poolsize, self.poolsize))
        featS_re = featS_.view(featS_.shape[0], featS_.shape[1], -1)
        featS_swap = featS_re.permute(0, 2, 1)
        featT_re = featT_.view(featT_.shape[0], featT_.shape[1], -1)
        featT_swap = featT_re.permute(0, 2, 1)
        for i in range(preds_S.shape[0]):
            S_re = featS_re[i]
            S_swap = featS_swap[i]
            T_re = featT_re[i]
            T_swap = featT_swap[i]
            T_crr = torch.mm(T_swap, T_re)
            S_crr = torch.mm(S_swap, S_re)
            l1 = self.criterion(S_crr, T_crr)
            l1 = l1 / (int(self.poolsize) * int(self.poolsize))
            self.total_l1 = self.total_l1 + l1
        self.total_l1 = self.total_l1 / i
        return self.total_l1
