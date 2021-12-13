import torch
import torch.nn.functional as F


def get_colors():
    return {
        1: [152, 223, 138],
        2: [174, 199, 232],
        3: [255, 127, 14],
        4: [91, 163, 138],
        5: [255, 187, 120],
        6: [188, 189, 34],
        7: [140, 86, 75],
        8: [255, 152, 150],
        9: [214, 39, 40],
        10: [197, 176, 213],
        11: [196, 156, 148],
        12: [23, 190, 207],
        13: [112, 128, 144]
    }


def nanmean(v, *args, inplace=False, **kwargs):
    if not inplace:
        v = v.clone()
    is_nan = torch.isnan(v)
    v[is_nan] = 0
    return v.sum(*args, **kwargs) / (~is_nan).float().sum(*args, **kwargs)


def cal_loss(pred, gold, smoothing=True):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.2
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        loss = -(one_hot * log_prb).sum(dim=1).mean()
    else:
        loss = F.cross_entropy(pred, gold, reduction='mean')

    return loss


def calculate_sem_IoU(pred_all, seg):
    pred = pred_all.max(dim=-1)[1]
    IoUs = torch.tensor([])
    I_all = torch.zeros(2)
    U_all = torch.zeros(2)
    for sem_idx in range(seg.shape[0]):
        for sem in range(2):
            I = torch.sum(torch.logical_and(
                pred[sem_idx] == sem, seg[sem_idx] == sem))
            U = torch.sum(torch.logical_or(
                pred[sem_idx] == sem, seg[sem_idx] == sem))
            I_all[sem] += I
            U_all[sem] += U

        IoUs = torch.cat((IoUs, (I_all / U_all)))
    torch.where(pred == sem, 1, 0)
    return IoUs.mean()
