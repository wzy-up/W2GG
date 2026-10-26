from cmath import tau
from random import sample
import torch
import torch.nn as nn 
from torch.functional import F
from w2gg.data.datasets.utils import box_iou


class BceLoss(object):
    def __init__(self, min_iou, max_iou, mask2d):
        self.min_iou, self.max_iou = min_iou, max_iou
        self.mask2d = mask2d
        self.bceloss = torch.nn.BCELoss(reduction='none')
        self.hinge_loss = False

    def linear_scale(self, iou):
        return (iou - self.min_iou) / (self.max_iou - self.min_iou)

    def __call__(self, scores2d, ious2d, epoch):
        iou1d = ious2d.masked_select(self.mask2d)
        scores1d = scores2d.masked_select(self.mask2d)
        loss = 0
        iou1d = self.linear_scale(iou1d).clamp(0, 1)
        loss += self.bceloss(scores1d, iou1d).mean()
        return loss


def build_bce_loss(cfg, mask2d):
    min_iou = cfg.MODEL.W2GG.LOSS.MIN_IOU 
    max_iou = cfg.MODEL.W2GG.LOSS.MAX_IOU
    return BceLoss(min_iou, max_iou, mask2d)

class BceLossWithWeight(object):
    def __init__(self, min_iou, max_iou, mask2d):
        self.min_iou, self.max_iou = min_iou, max_iou
        self.mask2d = mask2d
        self.bceloss = torch.nn.BCELoss(reduction='none')

    def linear_scale(self, iou):
        return (iou - self.min_iou) / (self.max_iou - self.min_iou)

    def __call__(self, scores2d, ious2d, weights, epoch):
        # [num_sent_sum, num_sparse]
        iou1d = ious2d.masked_select(self.mask2d)
        scores1d = scores2d.masked_select(self.mask2d)
        loss = 0
        iou1d = self.linear_scale(iou1d).clamp(0, 1)
        loss = self.bceloss(scores1d, iou1d)
        # weights: [num_sent_sum]
        loss = (weights.unsqueeze(dim=1) * loss).mean()
      
        return loss

def build_bce_with_weight_loss(cfg, mask2d):
    min_iou = cfg.MODEL.W2GG.LOSS.MIN_IOU 
    max_iou = cfg.MODEL.W2GG.LOSS.MAX_IOU
    return BceLossWithWeight(min_iou, max_iou, mask2d)

class SmoothCrossEntropy(nn.Module):
    def __init__(self):
        super(SmoothCrossEntropy, self).__init__()
        self.log_softmax = nn.LogSoftmax(dim=1)
    
    def forward(self, score, target):
        # scores: [N, C]
        # target: [N, C]
        loss = self.log_softmax(score)
        loss = (target * loss)
        return -loss 
    

class GroupContrastiveLoss(object):
    def __init__(self, cfg, mask2d):
        self.cfg = cfg 
        self.mask2d = mask2d 
        self.tau = 0.1
        self.topk = cfg.SOLVER.TOPK 
        self.warmup = -1 # -1 denotes no warmup 
      
    def __call__(self, feat2ds, sent_feats, gmasks2d, weight2d, epoch=0):
        """
        Args:
            feat2ds (tensor): [B, C, T, T]
            sent_feats (list): list[item], each item(tensor): [num_sent, C]
            gmasks2d (list): list[item], item(tensor, bool): [num_sent, T, T] denotes which region contains glance frame 
            weight2d (list): list[item], item(tensor): [num_sent, T, T], gaussian weights for each clip
        """
        # prepare feats 
        B, C, T, _ = feat2ds.size()
        # [B, C, num_sparse]
        feat1ds = feat2ds.masked_select(self.mask2d).reshape(B, C, -1)#有效候选moment（稀疏采样后的）
        feat1ds_norm = F.normalize(feat1ds, dim=1)
        num_sparse = feat1ds.size(2)#本批次有效moment数量
        # [num_sent_sum, C]
        sent_feats_all = torch.cat(sent_feats, dim=0)
        sent_feats_all_norm = F.normalize(sent_feats_all, dim=1)
        num_sent_sum = sent_feats_all.size(0)#本批次句子数量
        
        num_sent_list = [sent.size(0) for sent in sent_feats]
        gmasks2d_all = torch.cat(gmasks2d, dim=0)
        gmasks1d_all = gmasks2d_all.masked_select(self.mask2d).reshape(-1, num_sparse)
        weight2d_all = torch.cat(weight2d, dim=0)
        # [num_sent_sum, num_sparse]
        weight1d_all = weight2d_all.masked_select(self.mask2d).reshape(-1, num_sparse)
        
        # [C, B*num_sparse]
        feat1ds_norm = feat1ds_norm.permute(1, 0, 2).contiguous().view(C, -1)
        # [num_sent_sum, B*num_spare]
        sim = torch.mm(sent_feats_all_norm, feat1ds_norm)#[num_sent_sum, C]*[C, B*num_sparse] 结果的第一行是指第一个句子和本批次其他所有moment的相似度得分
        sim = sim.view(-1, B, num_sparse)
        sim_split = torch.split(sim, num_sent_list, dim=0)
        gmasks1d_split = torch.split(gmasks1d_all, num_sent_list, dim=0)#记录每个句子的moment是否包含glance
        weight1d_split = torch.split(weight1d_all, num_sent_list, dim=0)
        valid_masks = []
        pos_masks = []
        pos_weights = []
        # process each video 
        for i, (split, gmask2d) in enumerate(zip(sim_split, gmasks1d_split)):
            # split: [num_sent, B, num_sparse]
            valid_mask = torch.ones_like(split)
            pos_mask = torch.zeros_like(split)
            pos_weight = torch.zeros_like(split)
            # [num_sent, num_sparse]
            pos_weight[:, i, :] = weight1d_split[i] 
            
            s = split[:, i, :] # s: [num_sent, num_sparse]
            mask_sim = s * gmask2d # gmask2d: [num_sent, num_sparse]
            
            # select topk position as positive clip according to sim
            mask_weight1d = weight1d_split[i] * gmask2d 
            
            if epoch <= self.warmup:
                # select topk according to gaussian weight 
                values, indices = torch.topk(mask_weight1d, self.topk, dim=1)
            else:
                ## select topk according to masked sim 
                joint_value = mask_sim * mask_weight1d # use both gaussian prior and semantic similarity 
                # joint_value = mask_sim # only use sim 
                values, indices = torch.topk(joint_value, self.topk, dim=1)
                 
            topk_mask = torch.zeros_like(mask_sim)
            topk_mask = torch.scatter(topk_mask, 1, indices, 1)
            pos_mask[:, i, :] = topk_mask 
            
            # rm semi-positive 
            rm_mask = 1 - gmask2d.float()
            rm_mask = torch.scatter(rm_mask, 1, indices, 1) # set topk-position as valid 
            valid_mask[:, i, :] = rm_mask 
            
            valid_masks.append(valid_mask)
            pos_masks.append(pos_mask)
            pos_weights.append(pos_weight)
    
        valid_masks_all = torch.cat(valid_masks, dim=0)
        pos_masks_all = torch.cat(pos_masks, dim=0)
        pos_weight_all = torch.cat(pos_weights, dim=0)
        
        # reshape 
        # [num_sent_sum, B*num_sparse]
        sim = sim.view(num_sent_sum, -1)
        valid_masks_all = valid_masks_all.view(num_sent_sum, -1)
        pos_masks_all = pos_masks_all.view(num_sent_sum, -1)
        pos_weight_all = pos_weight_all.view(num_sent_sum, -1)
        
        
        exp_sim = torch.exp(sim / self.tau)
        # [num_sent_sum, topk]
        pos_sim = exp_sim.masked_select(pos_masks_all.type(torch.bool)).view(num_sent_sum, -1)
        pos_w = pos_weight_all.masked_select(pos_masks_all.type(torch.bool)).view(num_sent_sum, -1)

        exp_sim = exp_sim * valid_masks_all
        sim_sum = torch.sum(exp_sim, dim=1, keepdim=True)
        sim_sum = sim_sum.repeat(1, pos_sim.size(1))

        logits = pos_sim / sim_sum

        # ---- NaN/Inf diagnostics ----
        if not torch.isfinite(exp_sim).all():
            print(f"[W2GG-DIAG][GroupContrastiveLoss] exp_sim non-finite cnt={(~torch.isfinite(exp_sim)).sum().item()}")
        if not torch.isfinite(sim_sum).all() or (sim_sum == 0).any():
            print(f"[W2GG-DIAG][GroupContrastiveLoss] sim_sum invalid non_finite={(~torch.isfinite(sim_sum)).sum().item()} zero_cnt={(sim_sum == 0).sum().item()}")
        if not torch.isfinite(logits).all() or (logits <= 0).any():
            print(f"[W2GG-DIAG][GroupContrastiveLoss] logits invalid non_finite={(~torch.isfinite(logits)).sum().item()} le0_cnt={(logits <= 0).sum().item()}")

        log_term = torch.log(logits)
        if not torch.isfinite(log_term).all():
            print(f"[W2GG-DIAG][GroupContrastiveLoss] log(logits) non-finite cnt={(~torch.isfinite(log_term)).sum().item()}")

        # weight sum
        log_loss = (pos_w * log_term).mean()
        if not torch.isfinite(log_loss):
            print(f"[W2GG-DIAG][GroupContrastiveLoss] log_loss non-finite | pos_w_non_finite={(~torch.isfinite(pos_w)).sum().item()} pos_sim_non_finite={(~torch.isfinite(pos_sim)).sum().item()}")
        return -log_loss
    
    