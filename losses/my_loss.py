import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor
from torch.nn.modules.loss import _Loss
from typing import List, Optional


class WeightedLoss(_Loss):
    """Wrapper class around loss function that applies weighted with fixed factor.
    This class helps to balance multiple losses if they have different scales
    """

    def __init__(self, loss, weight=1.0):
        super().__init__()
        self.loss = loss
        self.weight = weight

    def forward(self, *input):
        return self.loss(*input) * self.weight


def effective_num_weights(pixel_counts: torch.Tensor, beta: float = 0.999):
    """
    pixel_counts: [C] 每类像素数（用全训练集统计一次即可）
    返回归一化权重 [C]，稀有类权重大
    """
    pc = pixel_counts.float().clamp_min(1.0)
    eff_num = (1.0 - beta) / (1.0 - beta ** pc)
    w = eff_num / eff_num.mean()
    return w


class ClassBalancedFocalCELoss(nn.Module):
    """
    多类 Softmax + Focal(γ) + Class-Balanced 权重，有 ignore_index
    """
    def __init__(self, class_weights: torch.Tensor = None, gamma: float = 2.0,
                 ignore_index: int = 255):
        super().__init__()
        self.register_buffer('class_weights', class_weights if class_weights is not None else None)
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, target: torch.Tensor):
        # logits: [B,C,H,W], target: [B,H,W]
        logp = F.log_softmax(logits, dim=1)     # [B,C,H,W]
        p = logp.exp()
        B,C,H,W = logits.shape

        tgt = target.long()
        mask = (tgt != self.ignore_index)
        # 展平到 [N, C] 计算
        logp = logp.permute(0,2,3,1)[mask]     # [M,C]
        p    = p.permute(0,2,3,1)[mask]        # [M,C]
        tgt  = tgt[mask]                        # [M]

        pt   = p[torch.arange(p.shape[0]), tgt]
        loss = -(1 - pt) ** self.gamma * logp[torch.arange(logp.shape[0]), tgt]
        if self.class_weights is not None:
            loss = self.class_weights[tgt] * loss
        return loss.mean()


class FocalTverskyLoss(nn.Module):
    """
    多类 macro 版 Focal-Tversky
    alpha: FP 权重, beta: FN 权重；小目标想要高召回 -> beta > alpha
    gamma: focal 指数（1~2 常用）
    """
    def __init__(self, alpha: float = 0.3, beta: float = 0.7, gamma: float = 1.5,
                 ignore_index: int = 255, eps: float = 1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta  = beta
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.eps = eps

    def forward(self, logits: torch.Tensor, target: torch.Tensor):
        probs = F.softmax(logits, dim=1)       # [B,C,H,W]
        B,C,H,W = probs.shape
        tgt = target.clone()
        mask = (tgt != self.ignore_index)
        tgt[~mask] = 0

        loss = 0.0
        m = mask.unsqueeze(1).float()
        for c in range(C):
            pc = probs[:, c] * m[:,0]          # [B,H,W]
            gc = (tgt == c).float() * m[:,0]
            tp = (pc * gc).sum()
            fp = (pc * (1 - gc)).sum()
            fn = ((1 - pc) * gc).sum()
            ti = (tp + self.eps) / (tp + self.alpha * fp + self.beta * fn + self.eps)
            l  = (1 - ti) ** self.gamma
            loss += l
        return loss / C


class TopKCrossEntropyLoss(nn.Module):
    def __init__(self, k: float = 0.2, ignore_index: int = 255):
        super().__init__()
        self.k = k
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, target: torch.Tensor):
        ce = F.cross_entropy(logits, target, reduction='none', ignore_index=self.ignore_index)  # [B,H,W]
        B = ce.shape[0]
        ce = ce.view(B, -1)
        kk = max(1, int(self.k * ce.shape[1]))
        topk, _ = torch.topk(ce, kk, dim=1)
        return topk.mean()


class SAMAwareLoss(nn.Module):
    """
    主头： CB-Focal CE  +  Focal-Tversky  +  Top-K CE
    可显著提升稀有小目标（human）召回；其余类保持稳定
    """
    def __init__(self, class_pixel_counts: Optional[List[int]] = None,
                 gamma: float = 2.0, alpha: float = 0.3, beta: float = 0.7, ft_gamma: float = 1.5,
                 k_top: float = 0.2, ignore_index: int = 255,
                 w_ce: float = 0.5, w_tversky: float = 0.3, w_topk: float = 0.2):
        super().__init__()
        cw = None
        if class_pixel_counts is not None:
            cw = effective_num_weights(torch.tensor(class_pixel_counts), beta=0.999)
        self.ce  = WeightedLoss(ClassBalancedFocalCELoss(class_weights=cw, gamma=gamma, ignore_index=ignore_index),
                                weight=w_ce)
        self.ftv = WeightedLoss(FocalTverskyLoss(alpha=alpha, beta=beta, gamma=ft_gamma, ignore_index=ignore_index),
                                weight=w_tversky)
        self.topk= WeightedLoss(TopKCrossEntropyLoss(k=k_top, ignore_index=ignore_index),
                                weight=w_topk)

    def forward(self, logits, labels):
        return self.ce(logits, labels) + self.ftv(logits, labels) + self.topk(logits, labels)
