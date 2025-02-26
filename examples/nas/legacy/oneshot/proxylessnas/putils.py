import torch
import nni.retiarii.nn.pytorch as nn
### Changjae Lee @ 2022-09-22 
#from sklearn.metrics import confusion_matrix 
# https://torchmetrics.readthedocs.io/en/stable/classification/fbeta_score.html 
from torchmetrics import FBetaScore 

def get_parameters(model, keys=None, mode='include'):
    if keys is None:
        for name, param in model.named_parameters():
            yield param
    elif mode == 'include':
        for name, param in model.named_parameters():
            flag = False
            for key in keys:
                if key in name:
                    flag = True
                    break
            if flag:
                yield param
    elif mode == 'exclude':
        for name, param in model.named_parameters():
            flag = True
            for key in keys:
                if key in name:
                    flag = False
                    break
            if flag:
                yield param
    else:
        raise ValueError('do not support: %s' % mode)


def get_same_padding(kernel_size):
    if isinstance(kernel_size, tuple):
        assert len(kernel_size) == 2, 'invalid kernel size: %s' % kernel_size
        p1 = get_same_padding(kernel_size[0])
        p2 = get_same_padding(kernel_size[1])
        return p1, p2
    assert isinstance(kernel_size, int), 'kernel size should be either `int` or `tuple`'
    assert kernel_size % 2 > 0, 'kernel size should be odd number'
    return kernel_size // 2


def build_activation(act_func, inplace=True):
    if act_func == 'relu':
        return nn.ReLU(inplace=inplace)
    elif act_func == 'relu6':
        return nn.ReLU6(inplace=inplace)
    elif act_func == 'tanh':
        return nn.Tanh()
    elif act_func == 'sigmoid':
        return nn.Sigmoid()
    elif act_func is None:
        return None
    else:
        raise ValueError('do not support: %s' % act_func)


def make_divisible(v, divisor, min_val=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_val is None:
        min_val = divisor
    new_v = max(min_val, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

### Changjae Lee @ 2022-09-19 
# def accuracy(output, target, topk=(1,)):
#     """ Computes the precision@k for the specified values of k """
#     maxk = max(topk)
#     batch_size = target.size(0)

#     _, pred = output.topk(maxk, 1, True, True)
#     pred = pred.t()
#     # one-hot case
#     if target.ndimension() > 1:
#         target = target.max(1)[1]

#     correct = pred.eq(target.view(1, -1).expand_as(pred))

#     res = dict()
#     for k in topk:
#         correct_k = correct[:k].reshape(-1).float().sum(0)
#         res["acc{}".format(k)] = correct_k.mul_(1.0 / batch_size).item()
#     return res

### Changjae Lee @ 2022-09-19 
def bin_accuracy(output, target): 
    batch_size = target.size(0)
    correct = 0 
    
    # output (batch_size, 2) 
    # labels (batch_size, ) 
    
    _, pred = output.topk(1, 1, True, True)
    #pred = pred.t()
    #correct = pred.eq(target.view(1, -1).expand_as(pred)) 
    correct += pred.eq(target.view_as(pred)).sum() 
    correct = correct.float().mul_(1.0 / batch_size).item()

    res = dict() 
    res["acc{}".format(1)] = correct 

    return res  

### Changjae Lee @ 2022-09-22 
def bin_f_beta(output, target, device): 
    _, pred = output.topk(1, 1, True, True) 
    f_beta = FBetaScore(num_classes=2, beta=2).to(device) 
    F_beta_score = f_beta(pred.view(-1), target.view(-1))
    # C = confusion_matrix(target.view(-1), pred.view(-1)) 
    
    # precision = C[1][1] / (C[1][1] + C[0][1]) 
    # sensitivity = C[1][1] / (C[1][1] + C[1][0]) 
    
    #F_beta_score = (1+2**2) * (precision * sensitivity) / ((2**2)*precision + sensitivity)

    res = dict() 
    # https://stackoverflow.com/questions/49768306/pytorch-tensor-to-numpy-array 
    res["acc{}".format(1)] = F_beta_score.detach().cpu().numpy()

    return res  

class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing=0.1, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        num_classes = pred.size(self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (num_classes - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))
