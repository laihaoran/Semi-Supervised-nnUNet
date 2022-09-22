from torch import nn, Tensor
import torch.nn.functional as F
import torch
from nnformer.utilities.nd_softmax import softmax_helper
from nnformer.training.loss_functions.crossentropy import Talyer2CrossEntropyLoss

def reshape_tensor_to_2D(x):
    """
    reshape input variables of shape [B, C, D, H, W] to [voxel_n, C]
    """
    tensor_dim = len(x.size())
    num_class = list(x.size())[1]
    if (tensor_dim == 5):
        x_perm = x.permute(0, 2, 3, 4, 1)
    elif (tensor_dim == 4):
        x_perm = x.permute(0, 2, 3, 1)
    else:
        raise ValueError("{0:}D tensor not supported".format(tensor_dim))

    y = torch.reshape(x_perm, (-1, num_class))
    return y

def reshape_tensor_to_3D(x):
    """
    reshape input variables of shape [B, C, D, H, W] to [voxel_n, C]
    """
    tensor_dim = len(x.size())
    num_class = list(x.size())[1]
    batch_size = list(x.size())[0]
    if (tensor_dim == 5):
        x_perm = x.permute(0, 2, 3, 4, 1)
    elif (tensor_dim == 4):
        x_perm = x.permute(0, 2, 3, 1)
    else:
        raise ValueError("{0:}D tensor not supported".format(tensor_dim))

    y = torch.reshape(x_perm, (batch_size, -1, num_class))
    return y

def label_process(net_output, gt, axes=None, mask=None, square=False):
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
    if axes is None:
        axes = tuple(range(2, len(net_output.size())))

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


class NoiseRobustDiceLoss(nn.Module):
    """
    Noise-robust Dice loss according to the following paper.
        G. Wang et al. A Noise-Robust Framework for Automatic Segmentation of COVID-19
        Pneumonia Lesions From CT Images, IEEE TMI, 2020.
    """
    def __init__(self,apply_nonlin=None, batch_dice=False, do_bg=True, smooth=1.):
        super(NoiseRobustDiceLoss, self).__init__()
        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.gamma = 1.5
        # self.enable_pix_weight = params['NoiseRobustDiceLoss_Enable_Pixel_Weight'.lower()]
        # self.enable_cls_weight = params['NoiseRobustDiceLoss_Enable_Class_Weight'.lower()]
        # self.gamma = params['NoiseRobustDiceLoss_gamma'.lower()]

    def forward(self, x, y, loss_mask=None):
        shp_x = x.shape

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        # predict = loss_input_dict['prediction']
        # soft_y  = loss_input_dict['ground_truth']
        # pix_w   = loss_input_dict['pixel_weight']
        # cls_w   = loss_input_dict['class_weight']
        # softmax = loss_input_dict['softmax']

        predict, soft_y = label_process(x, y, axes, loss_mask, False)
        if not self.do_bg:
            predict = predict[:, 1:]
            soft_y = soft_y[:, 1:]
        if self.batch_dice:
            predict = reshape_tensor_to_2D(predict)
            soft_y  = reshape_tensor_to_2D(soft_y)

            numerator = torch.abs(predict - soft_y)
            numerator = torch.pow(numerator, self.gamma)
            denominator = predict + soft_y
            # if(self.enable_pix_weight):
            #     if(pix_w is None):
            #         raise ValueError("Pixel weight is enabled but not defined")
            #     pix_w = reshape_tensor_to_2D(pix_w)
            #     numerator = numerator * pix_w
            #     denominator = denominator * pix_w
            numer_sum = torch.sum(numerator,  dim = 0)
            denom_sum = torch.sum(denominator,  dim = 0)
            loss_vector = numer_sum / (denom_sum + 1e-5)
            # if(self.enable_cls_weight):
            #     if(cls_w is None):
            #         raise ValueError("Class weight is enabled but not defined")
            #     weighted_dice = loss_vector * cls_w
            #     loss =  weighted_dice.sum() / cls_w.sum()
            # else:
            loss = torch.mean(loss_vector)
        else:
            loss = 0
            predict = reshape_tensor_to_3D(predict)
            soft_y = reshape_tensor_to_3D(soft_y)
            for i in range(predict.size(0)):
                numerator = torch.abs(predict[i] - soft_y[i])
                numerator = torch.pow(numerator, self.gamma)
                denominator = predict[i] + soft_y[i]
                numer_sum = torch.sum(numerator, dim=0)
                denom_sum = torch.sum(denominator, dim=0)
                loss_vector = numer_sum / (denom_sum + 1e-5)
                loss += torch.mean(loss_vector)
            loss /= predict.size(0)
        return loss


class Robust_DC_and_CE_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, aggregate="sum", square_dice=False, weight_ce=1, weight_dice=1,
                 log_dice=False, ignore_label=None):
        """
        CAREFUL. Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super(Robust_DC_and_CE_loss, self).__init__()
        if ignore_label is not None:
            assert not square_dice, 'not implemented'
            ce_kwargs['reduction'] = 'none'
        self.log_dice = log_dice
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.aggregate = aggregate
        self.ce = Talyer2CrossEntropyLoss(**ce_kwargs)

        self.ignore_label = ignore_label

        self.dc = NoiseRobustDiceLoss(apply_nonlin=softmax_helper, **soft_dice_kwargs)


    def forward(self, net_output, target):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'not implemented for one hot encoding'
            mask = target != self.ignore_label
            target[~mask] = 0
            mask = mask.float()
        else:
            mask = None

        dc_loss = self.dc(net_output, target, loss_mask=mask) if self.weight_dice != 0 else 0
        if self.log_dice:
            dc_loss = -torch.log(-dc_loss)

        ce_loss = self.ce(net_output, target[:, 0].long()) if self.weight_ce != 0 else 0
        if self.ignore_label is not None:
            ce_loss *= mask[:, 0]
            ce_loss = ce_loss.sum() / mask.sum()

        if self.aggregate == "sum":
            result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        else:
            raise NotImplementedError("nah son") # reserved for other stuff (later)
        return result