import random
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable

import os
import sys
import math
import numpy as np

class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                padding=0, bias=False) or None

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    def __init__(self, depth=34, num_classes=10, widen_factor=10, dropRate=0.0):
        super(WideResNet, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    # def forward(self, x, layer=None, ablated_units=None, pred_cls=None):
    def forward(self, x, layer=None, binary_mask=None, pred_cls=None):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        if layer is not None:
            if 'block2' in layer:
                if pred_cls is None and binary_mask is not None:
                    out = out * binary_mask
                elif pred_cls is not None and binary_mask is not None:
                    out = out * (pred_cls @ binary_mask)
                else:
                    raise NotImplementedError

        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)

        if layer is not None:
            if 'block3' in layer:
                if pred_cls is None and binary_mask is not None:
                    out = out * binary_mask
                elif pred_cls is not None and binary_mask is not None:
                    out = out * (pred_cls @ binary_mask)
                else:
                    raise NotImplementedError

        out = self.fc(out)
        
        if layer is not None:
            if 'block4' in layer:
                out = out * pred_cls

        return out

def WideResNet34_10(num_classes=10):
    return WideResNet(depth=34, num_classes=num_classes, widen_factor=10, dropRate=0.0)

def WideResNet28_10(num_classes=10):
    return WideResNet(depth=28, num_classes=num_classes, widen_factor=10, dropRate=0.0)

class InterpMaskedWideResNet(WideResNet):
    def __init__(self, layer_name, checkpoint_name, mask_which, important_dim, depth=34, widen_factor=10, num_classes=10, dropout_rate=0.0, rs=False, rs_sigma=8/255, rs_nsmooth=1, batch_size=100):
        super(InterpMaskedWideResNet, self).__init__(depth, num_classes, widen_factor, dropout_rate)
        self.layer_name = layer_name
        self.checkpoint_name = checkpoint_name
        self.mask_which = mask_which
        self.important_dim = important_dim
        self.num_classes = num_classes
        self.layer2dim = {'block2': 320, 'block3': 640}
        self.layer_dim = self.layer2dim[layer_name]
        self.rs = rs
        if self.rs:
            self.rs_sigma = rs_sigma
            self.rs_nsmooth = rs_nsmooth
            print('using RS with', self.rs_nsmooth, 'samples and sigma', self.rs_sigma, 'instead of vanilla forward pass')
        else:
            print('using vanilla forward pass before masking')

        if self.mask_which != 'none' and self.mask_which != 'twoforwardpasses':
            self.neuron_importance, self.binary_masks = self.load_neuron_importance()
        else:
            self.neuron_importance = None

    def load_neuron_importance(self):
        if self.mask_which == 'loir':
            root_name = 'saved_loir_rankings/'
        elif self.mask_which == 'cdir':
            root_name = 'saved_cdir_rankings/'
        elif self.mask_which == 'random':
            root_name = None
        else:
            raise NotImplementedError('check argument mask_which')

        model_save_name = os.path.splitext(os.path.basename(self.checkpoint_name))[0]

        if root_name is not None:
            folder_name = root_name + model_save_name + '/' + self.layer_name

            ablated_acc = torch.zeros((self.layer_dim, self.num_classes)).cuda()

            # loading the neuron importance for every class
            for k in range(self.layer_dim):
                ablated_acc[k] = torch.Tensor(np.load(folder_name + '/unit' + str(k) + '.npy'))
        else:
            # if root_name is None, then make sure it's not cdir or loir (just sanity checking)
            assert self.mask_which != 'cdir' and self.mask_which != 'loir'

        neuron_class_importance = torch.ones((self.num_classes, self.layer_dim - self.important_dim)) * -1

        for curr_cls in range(neuron_class_importance.shape[0]):
            if self.mask_which == 'random':
                neuron_class_importance[curr_cls] = torch.Tensor(random.sample(range(self.layer_dim), self.important_dim)).cuda()
            elif self.mask_which == 'cdir' or self.mask_which == 'loir':
                # HIGHER SIMILARITY/CHANGE IS HIGHER IMPORTANCE --> hence we sort in descending order
                # for logit comparison (difference of logits b/f and a/f masking), higher difference means higher importance
                neuron_class_importance[curr_cls] = ablated_acc[:, curr_cls].sort(0, descending=True)[1][-(self.layer_dim - self.important_dim):]

        binary_masks = torch.ones((self.num_classes, self.layer_dim)).cuda()
        for curr_cls in range(self.num_classes):
            curr_unitslist = [curr_unit.item() for curr_unit in neuron_class_importance[curr_cls]]
            binary_masks[curr_cls, curr_unitslist] = 0

        print('Neuron importance loaded with method {} from folder {}'.format(self.mask_which, folder_name))
        return neuron_class_importance, binary_masks


    def forward(self, x, layer=None, ablated_units=None, pred_cls=None, tau=1.0):
        if self.mask_which == 'twoforwardpasses':
            # no masking, just to see the effect of gradient masking in two forward passes
            pred_cls = super(InterpMaskedWideResNet, self).forward(x)
            return super(InterpMaskedWideResNet, self).forward(x)
        elif self.mask_which == 'none':
            if self.rs:
                sigma = self.rs_sigma
                n_smooth = self.rs_nsmooth
                tmp = torch.zeros((x.shape[0] * n_smooth, *x.shape[1:]))
                x_tmp = x.repeat((1, n_smooth, 1, 1)).view(tmp.shape).to(x.device)
                
                noise = torch.randn_like(x_tmp) * sigma
                noisy_x = x_tmp + noise
                noisy_preds = super(InterpMaskedWideResNet, self).forward(noisy_x)

                pred_cls = torch.ones((x.shape[0], self.num_classes), device=x.device) * -1
                # get smoothed prediction for each point
                for m in range(x.shape[0]):
                    pred_cls[m] = torch.mean(noisy_preds[(m * n_smooth):((m + 1) * n_smooth)], dim=0)
                return pred_cls
            else:
                return super(InterpMaskedWideResNet, self).forward(x)
        else:
            # vanilla forward pass
            if self.rs:
                # sigma = (8 / 255) * 2
                # n_smooth = 64
                sigma = self.rs_sigma
                n_smooth = self.rs_nsmooth
                tmp = torch.zeros((x.shape[0] * n_smooth, *x.shape[1:]))
                x_tmp = x.repeat((1, n_smooth, 1, 1)).view(tmp.shape).cuda()
                
                noise = torch.randn_like(x_tmp) * sigma
                noisy_x = x_tmp + noise
                noisy_preds = super(InterpMaskedWideResNet, self).forward(noisy_x)

                pred_cls = torch.ones((x.shape[0], self.num_classes), device=x.device) * -1
                # get smoothed prediction for each point
                for m in range(x.shape[0]):
                    pred_cls[m] = torch.mean(noisy_preds[(m * n_smooth):((m + 1) * n_smooth)], dim=0)
            else:
                pred_cls = super(InterpMaskedWideResNet, self).forward(x)

            pred_cls = F.softmax(pred_cls * 100, dim=1)

            # interp.-guided masking for 2nd forward pass
            return super(InterpMaskedWideResNet, self).forward(x, self.layer_name, pred_cls=pred_cls, binary_mask=self.binary_masks)
