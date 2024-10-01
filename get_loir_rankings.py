import argparse
import copy
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from multiprocessing import Pool

from models import resnet_val, wideresnet_trades
from models.resnet50 import ResNet50
from utils import (clamp, get_loaders, evaluate_standard, evaluate_standard_classwise, compare_predicted_probs, save_predicted_logits)

import random
from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--data-dir', default='./data', type=str)
    parser.add_argument('--load-model', type=str, help='filename of checkpoint')
    parser.add_argument('--dataset', type=str, choices=['cifar10', 'cifar100', 'imagenet'], default='cifar10')
    parser.add_argument('--arch', type=str, default='rn18_val', choices=['rn18_val', 'wrn34_10', 'rn50'])
    parser.add_argument('--preprocessing', type=str, default='Crop288', choices=['Crop288', 'Res256Crop224', 'Crop288-autoaug'], help='preprocessing type, only for ImageNet')
    parser.add_argument('--num-parallel-threads', default=8, type=int, help='Number of parallel threads, \
                                    use 8 for 24GB GPU and 4 for 12GB GPU, this significantly affects runtime')
    parser.add_argument('--layer-name', default='layer4', help='Name of layer whose output is ablated')
    parser.add_argument('--out-dir', default='saved_loir_rankings', type=str, help='Output directory')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument('--start-dim', default=0, type=int, help='index of neuron to start from, only for ImageNet')
    parser.add_argument('--end-dim', default=2048, type=int, help='index of neuron to end at, only for ImageNet')
    return parser.parse_args()


def get_acc_ablated(args_):
    model_test, train_eval_loader, model_name, out_dir, layer_name, unit, num_classes, layer_dim = args_
    model_save_name = os.path.splitext(os.path.basename(model_name))[0]
    binary_mask = torch.ones((layer_dim)).cuda()
    binary_mask[unit] = 0
    difference_accum = compare_predicted_probs(train_eval_loader, model_test, model_name, layer_name=layer_name, ablated_units=binary_mask, num_classes=num_classes, gt_only=True)
    print('unit', unit, '| average logits change', difference_accum)
    np.save(os.path.join(out_dir, model_save_name, layer_name, 'unit' + str(unit) + '.npy'), difference_accum)


def main():
    args = get_args()

    model_save_name = os.path.splitext(os.path.basename(args.load_model))[0]

    os.makedirs(os.path.join(args.out_dir, model_save_name, args.layer_name), exist_ok=True)

    if args.dataset != 'imagenet':
        list_of_files = os.listdir(os.path.join(args.out_dir, model_save_name, args.layer_name))    
        if len(list_of_files) == 0:
            start_dim = 0
        else:
            # [4:] to remove "unit" from the filename
            # extension is .npy so using [:-4] for removing the extension
            list_of_ids = [filename[4:] for filename in list_of_files]
            list_of_ids = [int(filename[:-4]) for filename in list_of_ids]
            start_dim = max(list_of_ids) - 1
            assert(start_dim >= 0)
    else:
        start_dim = args.start_dim
        end_dim = args.end_dim

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    print('finding important units for', args.load_model)

    if args.arch == 'rn18_val':
        layer2dim = {
            'layer1': 64,
            'layer2': 128,
            'layer3': 256,
            'layer4': 512
        }
    elif args.arch == 'rn50':
        layer2dim = {
            'layer3': 1024,
            'layer4': 2048
        }
    elif args.arch == 'wrn34_10':
        layer2dim = {'block2': 320, 'block3': 640}
    layer_dim = layer2dim[args.layer_name]

    # train_eval_loader gets the training set but with the test transforms 
    # (i.e. plain training data without the training augs.)
    if args.num_parallel_threads > 1:
        train_eval_loader, test_loader = get_loaders(args.dataset, args.data_dir, args.batch_size, args.arch, args.preprocessing)
    else:
        train_eval_loader, test_loader = get_loaders(args.dataset, args.data_dir, args.batch_size, args.arch, args.preprocessing, num_workers=8)
    print(args.dataset, 'dataset loaded')

    num_classes = {
        'cifar10': 10,
        'cifar100': 100,
        'imagenet': 1000
    }

    if args.arch == 'rn18_val':
        model_test = resnet_val.ResNet18().cuda()
    elif args.arch == 'wrn34_10':
        model_test = wideresnet_trades.WideResNet34_10().cuda()
    elif args.arch == 'rn50':
        model_test = ResNet50().cuda()

    model_test.load_state_dict(torch.load('checkpoints/'+args.load_model))
    model_test.float()
    model_test.eval()
    print('model loaded')

    # takes too long for imagenet (but can remove the condition if needed)
    if args.dataset != 'imagenet':
        test_loss, test_acc = evaluate_standard(test_loader, model_test)
        print('Test accuracy (for checking correctness of model and data loading):', test_acc)

    # skip saving if already saved previously
    if not os.path.exists(f'saved_predicted_logits/{model_save_name}_predlogits.pth'):
        print('saving predicted logits now')
        save_predicted_logits(train_eval_loader, model_test, args.load_model, num_classes[args.dataset])
    else:
        print(f'loading orig. pred. logits from saved_predicted_logits/{model_save_name}_predlogits.pth')

    if args.num_parallel_threads > 1:
        for k in tqdm(range(start_dim // args.num_parallel_threads, layer_dim // args.num_parallel_threads)):
            start_time = time.time()
            p = Pool(args.num_parallel_threads)
            unitlist = list()
            for idx in range(args.num_parallel_threads):
                unitlist.append((model_test, train_eval_loader, args.load_model, args.out_dir, args.layer_name, args.num_parallel_threads * k + idx, num_classes[args.dataset], layer_dim))

            p.map(get_acc_ablated, unitlist)
            print('Time taken : ', time.time() - start_time)
    else:
        ### multi-threading had lot of problems when run with imagenet (memory usage by each thread was too much even for 2 threads on our machine)
        ### so we just run it in a single thread but it can be parallelized manually by setting different smaller intervals of start_dim and end_dim
        assert args.dataset == 'imagenet'
        print(f'starting from {start_dim}, ending at {end_dim}')
        for k in tqdm(range(start_dim, end_dim)):
            start_time = time.time()
            print('Running for neuron', k)
            get_acc_ablated((model_test, train_eval_loader, args.load_model, args.out_dir, args.layer_name, k, num_classes[args.dataset], layer_dim))
            print('Time taken : ', time.time() - start_time)

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    main()
