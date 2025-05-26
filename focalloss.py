import torch
import torch.nn as nn
import torch.nn.functional as F
class FocalLoss(nn.Module):
    def __init__(self,
                 alpha=None,
                 gamma: float = 2.0,
                 reduction: str = 'mean',
                 ignore_index: int = -100):
        """
        alpha:         None 或者 shape=[num_classes] 的 Tensor 或 float，
                       二分类时 float 会被转换为 [alpha, 1-alpha] Tensor
        gamma:         聚焦系数 γ，典型值 2.0
        reduction:     'none' | 'mean' | 'sum'
        ignore_index:  忽略标签
        """
        super().__init__()
        # 如果是单一 float（用于二分类），转换为 [α, 1−α] Tensor
        if isinstance(alpha, float):
            self.alpha = torch.tensor([alpha, 1.0 - alpha], dtype=torch.float32)
        else:
            self.alpha = alpha  # None 或 已经是 Tensor
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        """
        inputs:  [N, C] logits 或者 [N] logits（二分类）
        targets: [N] LongTensor
        """
        # 二分类分支
        if inputs.dim() == 1 or inputs.size(1) == 1:
            logits = inputs.view(-1)
            targets = targets.view(-1).float()
            bce_loss = F.binary_cross_entropy_with_logits(
                logits, targets, reduction='none')
            pt = torch.exp(-bce_loss)  # pt = sigmoid(logit) 或 1−pt
            if self.alpha is not None:
                # 对二分类 alpha 张量广播
                alpha_factor = (targets * self.alpha[0] +
                                (1 - targets) * self.alpha[1]).to(bce_loss.device)
                loss = alpha_factor * (1 - pt) ** self.gamma * bce_loss
            else:
                loss = (1 - pt) ** self.gamma * bce_loss

        # 多分类分支
        else:
            logp = F.log_softmax(inputs, dim=1)     # [N, C] :contentReference[oaicite:0]{index=0}
            p = torch.exp(logp)                     # [N, C]
            # 取对应类别的 log p_t 和 p_t
            targets = targets.view(-1, 1)
            logpt = logp.gather(1, targets).view(-1)
            pt = p.gather(1, targets).view(-1)
            # 准备 α_t
            if self.alpha is not None:
                # 确保 α 是 Tensor 并在同 device
                if not isinstance(self.alpha, torch.Tensor):
                    raise ValueError("alpha must be Tensor for multiclass")
                at = self.alpha.to(inputs.device).gather(0, targets.view(-1))
            else:
                at = 1.0
            loss = -at * (1 - pt) ** self.gamma * logpt

        # 忽略特定标签
        if self.ignore_index >= 0:
            valid = targets.view(-1) != self.ignore_index
            loss = loss[valid]

        # Reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss