# import apex.amp as amp
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import clip
import math
import os
from tqdm import tqdm
import sys


def get_clip_text_features(model, text, batch_size=1000):
    """
    gets text features without saving, useful with dynamic concept sets
    """
    text_features = []
    with torch.no_grad():
        for i in range(math.ceil(len(text)/batch_size)):
            text_features.append(model.encode_text(text[batch_size*i:batch_size*(i+1)]))
    text_features = torch.cat(text_features, dim=0)
    return text_features

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            # t.mul_(s).add_(m)
            t = torch.mul(t, s) + m
            # The normalize code -> t.sub_(m).div_(s)
        return tensor

class CustomNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            # t.sub_(m).div_(s)
            t = t - m
            t = torch.div(t, s)
        return tensor


def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)

def get_stats(arch='parn18'):
    if arch == 'parn18':
        cifar10_mean = (0.4914, 0.4822, 0.4465)
        cifar10_std = (0.2471, 0.2435, 0.2616)
    elif arch == 'rn18' or arch == 'wrn34_10':
        cifar10_mean = (0.0, 0.0, 0.0)
        cifar10_std = (1.0, 1.0, 1.0)

    mu = torch.tensor(cifar10_mean).view(3,1,1).cuda()
    std = torch.tensor(cifar10_std).view(3,1,1).cuda()

    upper_limit = ((1 - mu)/ std)
    lower_limit = ((0 - mu)/ std)
    return cifar10_mean, cifar10_std, mu, std, upper_limit, lower_limit


def get_loaders(dataset, dir_, batch_size, arch, mode='Crop288', num_workers=0):
    if dataset == 'cifar10':
        if arch == 'rn18_val' or arch == 'wrn34_10':
            test_transform = transforms.Compose([
                transforms.ToTensor(),
            ])
        else:
            raise NotImplementedError(f'{dataset}, {arch}')
    elif dataset == 'cifar100':
        if arch == 'parn18_oaat' or arch == 'wrn34_10_las':
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
            test_transform = transforms.Compose([
                transforms.ToTensor(),
            ])
        else:
            raise NotImplementedError(f'{dataset}, {arch}')
    elif dataset == 'imagenet':
        target_mean = [0.485, 0.456, 0.406]
        target_std = [0.229, 0.224, 0.225]
        if mode == 'Crop288':
            test_transform = transforms.Compose([transforms.CenterCrop(288),
                                                 transforms.ToTensor(), 
                                                 transforms.Normalize(
                                                            mean=target_mean, 
                                                            std=target_std)
                                                ])
        elif mode == 'Res256Crop224':
            test_transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
                               transforms.ToTensor(), transforms.Normalize(mean=target_mean, std=target_std)])
        else:
            raise NotImplementedError(f'{dataset}, {arch}, {mode}')

    pin_mem = False
    if dataset == 'cifar10':
        train_eval_dataset = datasets.CIFAR10(
            dir_, train=True, transform=test_transform, download=True)
        test_dataset = datasets.CIFAR10(
            dir_, train=False, transform=test_transform, download=True)
    elif dataset == 'cifar100':
        train_eval_dataset = datasets.CIFAR100(
            dir_, train=True, transform=test_transform, download=True)
        test_dataset = datasets.CIFAR100(
            dir_, train=False, transform=test_transform, download=True)
    elif dataset == 'imagenet':
        train_eval_dataset = datasets.ImageFolder('/akshayvol/datasets/imagenet_trainsubset/train', test_transform)
        test_dataset = datasets.ImageFolder('/akshayvol/datasets/imagenet/val', test_transform)

    train_eval_loader = torch.utils.data.DataLoader(
        dataset=train_eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=pin_mem,
        num_workers=num_workers,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=pin_mem,
        num_workers=num_workers,
    )
    return train_eval_loader, test_loader



def evaluate_standard(test_loader, model, layer_name='layer4', ablated_units_per_class=None, use_pred_cls=True):
    test_loss = 0
    test_acc = 0
    n = 0
    model.eval()
    with torch.no_grad():
        for i, (X, y) in enumerate(test_loader):
            X, y = X.cuda(), y.cuda()
            output = model(X)
            # for the default case where predicted class is used to find the ablated units from the given set of ablated units per class
            if ablated_units_per_class is not None and use_pred_cls:
                output = model(X, layer_name, ablated_units_per_class, output.max(1)[1])
            # for finding important neurons when ablated unit is specified manually (which cannot be done via predicted class)
            elif ablated_units_per_class is not None and not use_pred_cls:
                output = model(X, layer_name, ablated_units_per_class)
            loss = F.cross_entropy(output, y)
            test_loss += loss.item() * y.size(0)
            test_acc += (output.max(1)[1] == y).sum().item()
            n += y.size(0)
    return test_loss/n, test_acc/n

def evaluate_standard_classwise(test_loader, model, layer_name='layer4', num_classes=10, ablated_units=None):
    # test_acc = torch.Tensor([0] * num_classes).cuda()
    # n = torch.Tensor([0] * num_classes).cuda()
    true_counts = torch.zeros(num_classes, dtype=torch.int64).cuda()
    pred_counts = torch.zeros(num_classes, dtype=torch.int64).cuda()
    correct_counts = torch.zeros(num_classes, dtype=torch.int64).cuda()
    total_count = 0
    model.eval()
    with torch.no_grad():
        for i, (X, y) in enumerate(test_loader):
            total_count += X.shape[0]
            X, y = X.cuda(), y.cuda()
            with autocast():
                if ablated_units is None:
                    output = model(X)
                else:
                    output = model(X, layer_name, ablated_units)
                loss = F.cross_entropy(output, y)
            # test_loss += loss.item() * y.size(0)
            # test_acc += (output.max(1)[1] == y).sum().item()
            preds = output.max(1)[1]
            correct = (preds == y).long()
            # test_acc += y.bincount(correct, minlength=num_classes)
            # # n += y.size(0)
            # n += y.bincount(minlength=num_classes)
            true_counts.add_(y.bincount(minlength=num_classes))
            pred_counts.add_(preds.bincount(minlength=num_classes))
            correct_counts.add_(y.bincount(correct, minlength=num_classes).long())

    true_neg_counts = (
            (total_count - true_counts) - (pred_counts - correct_counts))
    precision = (correct_counts.float() / pred_counts.float()).cpu()
    recall = (correct_counts.float() / true_counts.float()).cpu()
    accuracy = (correct_counts + true_neg_counts).float().cpu() / total_count
    true_neg_rate = (true_neg_counts.float() /
            (total_count - true_counts).float()).cpu()
    balanced_accuracy = (recall + true_neg_rate) / 2
    return precision, recall, accuracy, balanced_accuracy
    # return (test_acc / n).cpu().numpy()

# does not depend on layer, saving the output logits
def save_predicted_logits(test_loader, model, ckpt_name, num_classes=10):
    output_logits = torch.zeros((test_loader.dataset.__len__(), num_classes))
    print('saving output probabilities with shape:', output_logits.shape)
    batch_size = test_loader.batch_size
    model.eval()
    with torch.no_grad():
        for i, (X, y) in enumerate(tqdm(test_loader)):
            X, y = X.cuda(), y.cuda()
            with autocast():
                output = model(X)
            output = output.detach().cpu()
            output_logits[i * batch_size: (i + 1) * batch_size] = output
    os.makedirs('saved_predicted_logits', exist_ok=True)
    torch.save(output_logits, f'saved_predicted_logits/{ckpt_name[:-4]}_predlogits.pth')


def compare_predicted_probs(test_loader, model, ckpt_name, layer_name='layer4', ablated_units=None, num_classes=10, use_probs=False, gt_only=False):
    orig_output_logits = torch.load(f'saved_predicted_logits/{ckpt_name[:-4]}_predlogits.pth')
    difference_accumulator = torch.zeros((num_classes))
    batch_size = test_loader.batch_size
    assert ablated_units is not None
    model.eval()
    with torch.no_grad():
        for i, (X, y) in enumerate(test_loader):
            X, y = X.cuda(), y.cuda()
            with autocast():
                output = model(X, layer_name, ablated_units)
            output = output.detach().cpu()
            curr_orig_output = orig_output_logits[i * batch_size: (i + 1) * batch_size]
            difference = torch.abs(output - curr_orig_output)
            if gt_only:
                difference = difference * F.one_hot(y, num_classes).detach().cpu()
            # summing along batch dimension
            difference_accumulator += torch.sum(difference, dim=0)
    
    # normalizing the sum to get the average difference
    difference_accumulator /= test_loader.dataset.__len__()
    if gt_only:
        # if gt only, then only 5k images per class (divide by 5k effectively instead of 50k)
        difference_accumulator *= num_classes
    difference_accumulator = difference_accumulator.detach().cpu().numpy()
    return difference_accumulator
