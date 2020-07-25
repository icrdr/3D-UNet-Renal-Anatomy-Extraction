# %%
import torch
import torch.nn as nn
import torch.nn.functional as F


def logits(input):
    '''
    (N, C, d1, d2, ..., dn)
    '''
    return torch.softmax(input, dim=1) if input.size(1) > 1 else torch.sigmoid(input)


def flatten_and_tranpose_C(input, target):
    '''
        (N, C, d1, d2, ..., dn) => (N*d1*d2*...*dn, C)
    '''
    N = input.size(0)
    C = input.size(1)
    # N,C,d1,d2,...,dn => N,C,d1*d2*...*dn
    input = input.view(N, C, -1)
    # N,C,d1*d2*...*dn => N,d1*d2*...*dn,C
    input = input.transpose(1, 2)
    # N,d1*d2*...*dn,C => N*d1*d2*...*dn,C

    input = input.contiguous().view(-1, C)
    target = F.one_hot(target, num_classes=C)
    target = target.contiguous().view(-1, C)
    return input, target


def dice(input, target, alpha=0.5, beta=0.5, smooth=1e-7):
    '''
    :param input:
        type float
        shape (N, d1, d2,..., dn)
        value 0<=v<=1
    :param target:
        type long
        shape (N, d1, d2,..., dn)
        value 0<=v<=1
    '''
    p = input.contiguous().view(-1)
    g = target.contiguous().view(-1)
    true_pos = (p * g).sum()
    false_neg = ((1-p) * g).sum()
    false_pos = (p * (1-g)).sum()
    return (true_pos + smooth)/(true_pos + alpha*false_neg + beta*false_pos + smooth)


def focal_loss(input, target, gamma=2, weight_c=None, weight_v=None):
    '''
    :param input:
        type float
        shape (N*d1*d2*...*dn, C)
        value 0<=v<=1
    :param target:
        type long
        shape (N*d1*d2*...*dn, C)
        value 0<=v<=1
    '''
    C = input.size(-1)
    mask = target > 0
    mask = mask.any(0).type(torch.float).to('cpu')
    weight_c = torch.ones(C) if weight_c is None else torch.tensor(weight_c)
    weight_v = torch.ones(C) if weight_v is None else torch.tensor(weight_v)

    weight = weight_v * mask * weight_c
    weight = F.normalize(weight_v.type(torch.float), p=1, dim=0)

    if input.size(-1) > 1:
        logpt = F.log_softmax(input, -1)  # only work for float16
        pt = logpt.exp()
    else:
        pt = F.sigmoid(input)
        logpt = torch.log(pt)

    focals = -(1-pt)**gamma * target * logpt
    focals = C*focals.mean(dim=0).to('cpu')
    focals = weight*focals

    return focals.sum()


class Dice(nn.Module):
    '''
    input:
        type float
        shape (N, C, d1, d2, ..., dn)
        value 0<=v<=1
    target:
        type long
        shape (N, d1, d2, ..., dn)
        value 0<=v<=C−1
    '''

    def __init__(self, weight_v=None, alpha=0.5, beta=0.5, smooth=1e-7):
        super(Dice, self).__init__()
        self.weight_v = weight_v
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, input, target):
        C = input.size(1)
        weight_v = torch.ones(C) if self.weight_v is None else torch.tensor(self.weight_v)
        weight = F.normalize(weight_v.type(torch.float), p=1, dim=0)

        input = logits(input)
        input, target = flatten_and_tranpose_C(input, target)

        dices = torch.zeros(C)
        for i in range(C):
            dices[i] = dice(input[:, i],
                            target[:, i],
                            alpha=self.alpha,
                            beta=self.beta,
                            smooth=self.smooth)
        dices = weight*dices
        return dices.sum()


class DiceLoss(nn.Module):
    '''
    input:
        type float
        shape (N, C, d1, d2, ..., dn)
        value 0<=v<=1
    target:
        type long
        shape (N, d1, d2, ..., dn)
        value 0<=v<=C−1
    '''

    def __init__(self, weight_c=None, weight_v=None, alpha=0.5, beta=0.5, smooth=1e-7):
        super(DiceLoss, self).__init__()
        self.weight_c = weight_c
        self.weight_v = weight_v
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, input, target):
        C = input.size(1)
        input = logits(input)
        input, target = flatten_and_tranpose_C(input, target)

        mask = target > 0
        mask = mask.any(0).type(torch.float).to('cpu')

        weight_c = torch.ones(C) if self.weight_c is None else torch.tensor(self.weight_c)
        weight_v = torch.ones(C) if self.weight_v is None else torch.tensor(self.weight_v)

        weight = weight_v*mask*weight_c
        weight = F.normalize(weight_v.type(torch.float), p=1, dim=0)

        dices = torch.zeros(C)
        for i in range(C):
            dices[i] = dice(input[:, i],
                            target[:, i],
                            alpha=self.alpha,
                            beta=self.beta,
                            smooth=self.smooth)
        dices = weight*(1 - dices)

        return dices.sum()


class FocalLoss(nn.Module):
    '''
    :param input:
        type float
        shape (N, C, d1, d2, ..., dn)
        value 0<=v<=1
    :param target:
        type long
        shape (N, d1, d2, ..., dn)
        value 0<=v<=C−1
    '''

    def __init__(self, gamma=2, weight_c=None, weight_v=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight_c = weight_c
        self.weight_v = weight_v

    def forward(self, input, target):
        input, target = flatten_and_tranpose_C(input, target)
        loss = focal_loss(input, target,
                          gamma=self.gamma,
                          weight_c=self.weight_c,
                          weight_v=self.weight_v)
        return loss


class HybirdLoss(nn.Module):
    '''
    input:
        type float
        shape (N, C, d1, d2, ..., dn)
        value 0<=v<=1
    target:
        type long
        shape (N, d1, d2, ..., dn)
        value 0<=v<=C−1
    '''

    def __init__(self, gamma=2, weight_c=None, weight_v=None,
                 alpha=0.5, beta=0.5, smooth=1e-7):
        super(HybirdLoss, self).__init__()
        self.weight_c = weight_c
        self.weight_v = weight_v
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.gamma = gamma

    def forward(self, input, target):
        C = input.size(1)
        input, target = flatten_and_tranpose_C(input, target)

        # logit
        if C > 1:
            logpt = F.log_softmax(input, -1)
            pt = logpt.exp()
        else:
            pt = F.sigmoid(input)
            logpt = torch.log(pt)

        mask = target > 0
        mask = mask.any(0).type(torch.float).to('cpu')

        weight_c = torch.ones(C) if self.weight_c is None else torch.tensor(self.weight_c)
        weight_v = torch.ones(C) if self.weight_v is None else torch.tensor(self.weight_v)

        weight = weight_c*weight_v*mask
        weight = F.normalize(weight_v.type(torch.float), p=1, dim=0)

        # focal loss
        focals = -(1-pt)**self.gamma * target * logpt
        focals = C*focals.mean(dim=0).to('cpu')

        # dice loss
        dices = torch.zeros(C)
        for i in range(C):
            dices[i] = dice(pt[:, i],
                            target[:, i],
                            alpha=self.alpha,
                            beta=self.beta,
                            smooth=self.smooth)

        # combined loss
        loss = weight*(1-dices + focals)
        return loss.sum()


# %%
# p = torch.rand(2, 3, 10, 10, 10)
# g = torch.randint(0, 1, (2, 10, 10, 10))
# l0 = DiceLoss(weight_c=[1, 2, 3], weight_v=[10, 200, 200])
# l1 = FocalLoss(weight_c=[1, 2, 3], weight_v=[10, 200, 200])
# l2 = HybirdLoss(weight_c=[1, 2, 3], weight_v=[10, 200, 200])
# print(l0(p, g))
# print(l1(p, g))
# print(l2(p, g))

# class FocalDiceCoefLoss(nn.Module):
#     '''
#     input:
#         type float
#         shape (N, C, d1, d2, ..., dn)
#         value 0<=v<=1
#     target:
#         type long
#         shape (N, d1, d2, ..., dn)
#         value 0<=v<=C−1
#     '''

#     def __init__(self, alpha=None, f_alpha=None, f_gamma=2,  f_reduction='mean',
#                  d_w=None, d_alpha=0.5, d_smooth=1e-7, d_with_logits=True):
#         super(FocalDiceCoefLoss, self).__init__()
#         self.alpha = alpha
#         self.focal = FocalLoss(gamma=f_gamma, alpha=f_alpha, reduction=f_reduction)
#         self.dice = DiceCoefLoss(w=d_w, alpha=d_alpha,
#                                  smooth=d_smooth, with_logits=d_with_logits)

#     def forward(self, input, target):
#         if self.alpha is None:
#             self.alpha = [0.5, 0.5]

#         return self.alpha[0] * self.focal(input, target) + self.alpha[1] * self.dice(input, target)


# class FocalLoss2(nn.Module):
#     '''
#     :param input:
#         type float
#         shape (N, C, d1, d2, ..., dn)
#         value 0<=v<=1
#     :param target:
#         type long
#         shape (N, d1, d2, ..., dn)
#         value 0<=v<=C−1
#     '''

#     def __init__(self, gamma=2, reduction='mean'):
#         super(FocalLoss2, self).__init__()
#         self.gamma = gamma
#         self.reduction = reduction

#     def forward(self, input, target):
#         C = input.size(1)
#         if C > 1:
#             # pytorch build-in CE is with logits by default,
#             # so we don't need to worry about.
#             logpt = -F.cross_entropy(input, target, reduction='none')
#         else:
#             # squeeze C to fit pytorch build-in BCE
#             input = torch.squeeze(input, 1)
#             target = target.type_as(input)
#             logpt = -F.binary_cross_entropy_with_logits(input, target, reduction='none')

#         pt = logpt.exp()
#         loss = -1 * (1-pt)**self.gamma * logpt

#         if self.reduction == 'mean':
#             return loss.mean()
#         elif self.reduction == 'sum':
#             return loss.sum()
#         else:
#             return loss


# %%
