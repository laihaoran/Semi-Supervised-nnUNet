from torch import nn, Tensor
import torch.nn.functional as F
import torch
from typing import Callable, Optional
from nnformer.utilities.nd_softmax import softmax_helper

class RobustCrossEntropyLoss(nn.CrossEntropyLoss):
    """
    this is just a compatibility layer because my target tensor is float and has an extra dimension
    """
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        if len(target.shape) == len(input.shape):
            assert target.shape[1] == 1
            target = target[:, 0]
        return super().forward(input, target.long())


def label_process(net_output, gt):
    """
    net_output must be (b, c, x, y(, z)))
    gt must be a label map (shape (b, 1, x, y(, z)) OR shape (b, x, y(, z))) or one hot encoding (b, c, x, y(, z))
    if mask is provided it must have shape (b, 1, x, y(, z)))
    :param net_output:
    :param gt:
    :param axes: can be (, ) = no summation
    :param mask: mask must be 1 for valid pixels and 0 for invalid pixels
    :param square: if True then fp, tp and fn will be squared before summation
    :return:
    """
    # if axes is None:
    #     axes = tuple(range(2, len(net_output.size())))

    shp_x = net_output.shape
    shp_y = gt.shape

    with torch.no_grad():
        if len(shp_x) != len(shp_y):
            gt = gt.view((shp_y[0], 1, *shp_y[1:]))

        if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
            # if this is the case then gt is probably already a one hot encoding
            y_onehot = gt
        else:
            gt = gt.long()
            y_onehot = torch.zeros(shp_x)
            if net_output.device.type == "cuda":
                y_onehot = y_onehot.cuda(net_output.device.index)
            y_onehot.scatter_(1, gt, 1)

    return net_output, y_onehot


class Talyer2CrossEntropyLoss(nn.Module):
    """
      this is just a compatibility layer because my target tensor is float and has an extra dimension
    """
    __constants__ = ['ignore_index', 'reduction']
    ignore_index: int

    def __init__(self,weight: Optional[Tensor] = None, size_average=None, ignore_index: int = -100,
                 reduce=None, reduction: str = 'mean', apply_nonlin=softmax_helper):
        super(Talyer2CrossEntropyLoss, self).__init__()
        self.MAE = nn.L1Loss()
        self.MSE = nn.MSELoss()
        self.apply_nonlin = apply_nonlin

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        if self.apply_nonlin is not None:
            input = self.apply_nonlin(input)
        input, target = label_process(input, target)
        TCEloss = 0.5 * self.MAE(input, target) + 0.5 * self.MSE(input, target)
        return TCEloss


# x = torch.randn((2, 2, 48, 192, 192)).cuda()
# y = torch.randint(0, 2, (2, 1, 48, 192, 192)).cuda()
#
# c = Talyer2CrossEntropyLoss()
# loss = c(x, y)



