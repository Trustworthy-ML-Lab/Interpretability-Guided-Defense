import argparse

import os
import numpy as np
import random
import math
import torch
from torchvision import transforms

from robustbench import benchmark
from models.resnet_val import InterpMaskedResNet
from models.resnet50 import InterpMaskedResNet50
from models.wideresnet_trades import InterpMaskedWideResNet

from robustbench.model_zoo.architectures.utils_architectures import normalize_model
from autoattack import AutoAttack
from robustbench.model_zoo.enums import BenchmarkDataset, ThreatModel
from robustbench.utils import clean_accuracy, load_model, parse_args, update_json
from robustbench.data import get_preprocessing, load_clean_dataset


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', default='rn18', choices=['rn18', 'rn50', 'wrn34_10'], help='architecture of model')
    parser.add_argument('--threat-model', default="Linf", choices=['Linf', 'L2', 'corruptions'])
    parser.add_argument('--epsilon', default=8, type=int)
    parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'cifar100', 'imagenet'])
    parser.add_argument('--load-model', type=str, help='filename of checkpoint')
    parser.add_argument('--data-dir', type=str, default='./data', help='path to dataset')
    parser.add_argument('--mask-which', default='none', type=str, choices=['none', 'cdir', 'loir'], help='which neurons to mask?')
    parser.add_argument('--rs', action='store_true', default=False, help='give this argument when rs is enabled in vanilla forw. pass')
    parser.add_argument('--rs-sigma', default=4, type=int, help='sigma for smoothing noise, will be divided by 255 (e.g. give 8, it will be used as 8/255)')
    parser.add_argument('--rs-nsmooth', default=1, type=int, help='number of samples for smoothing')
    parser.add_argument('--layer-name', default='layer4', choices=['layer1', 'layer2', 'layer3', 'layer4', 'layer5', 'block2', 'block3', 'layer_2', 'layer_3', 'stages[0]', 'stages[1]', 'stages[2]', 'stages[3]', 'layer[0]', 'layer[1]', 'layer[2]'], help='Name of layer whose output is ablated')
    parser.add_argument('--important-dim', default=20, type=int, help='Number of important neurons to be retained in forward pass')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument('--batch-size', default=1000, type=int, help='Batch size for evaluation')
    parser.add_argument('--preprocessing', default='Crop288', choices=['Crop288', 'Res256Crop224', 'BicubicRes256Crop224'], help='Preprocessing method, only for ImageNet')
    parser.add_argument('--n-ex', default=10000, type=int, help='number of examples to evaluate on')
    parser.add_argument('--version', type=str, choices=['standard', 'rand'], default='standard', help='which version of AA to use')
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    dataset = args.dataset  # one of {"cifar10", "cifar100", "imagenet"}
    rs_sigma_int = args.rs_sigma
    if args.rs and args.rs_sigma > 0.1:
        args.rs_sigma /= 255

    num_classes = {
        'cifar10': 10,
        'cifar100': 100,
        'imagenet': 1000
    }

    if args.mask_which == 'none':
        print(f'Loading base model {args.load_model}')
    else:
        print(f'Loading IG-def {args.mask_which} model of {args.load_model}, retaining only {args.important_dim} neurons in {args.layer_name}')

    if args.arch == 'rn18':
        model = InterpMaskedResNet(args.layer_name, args.load_model, args.mask_which, args.important_dim, rs=args.rs, rs_sigma=args.rs_sigma, rs_nsmooth=args.rs_nsmooth)
    elif args.arch == 'rn50':
        model = InterpMaskedResNet50(args.layer_name, args.load_model, args.mask_which, args.important_dim, rs=args.rs, rs_sigma=args.rs_sigma, rs_nsmooth=args.rs_nsmooth)
    elif args.arch == 'wrn34_10':
        model = InterpMaskedWideResNet(args.layer_name, args.load_model, args.mask_which, args.important_dim, rs=args.rs, rs_sigma=args.rs_sigma, rs_nsmooth=args.rs_nsmooth)

    model_save_name = os.path.splitext(os.path.basename(args.load_model))[0]
    os.makedirs(f'saved_adversarial_images/{model_save_name}', exist_ok=True)

    model_name = model_save_name + '_' + args.layer_name + '_' + args.mask_which + str(args.important_dim)
    if args.rs:
        model_name += f'_rs_n{args.rs_nsmooth}_s{rs_sigma_int}'
    device = torch.device("cuda:0")

    model.load_state_dict(torch.load('checkpoints/'+args.load_model), strict=False)
    model.eval()

    if args.dataset == 'cifar10' or args.dataset == 'cifar100':
        n_examples = args.n_ex
        transform = transforms.Compose([transforms.ToTensor()])
    elif args.dataset == 'imagenet':
        target_mean = [0.485, 0.456, 0.406]
        target_std = [0.229, 0.224, 0.225]
        n_examples = 5000
        args.epsilon = 4
        if args.arch == 'rn50':
            if args.preprocessing == 'Crop288':
                transform = transforms.Compose([transforms.CenterCrop(288),
                               transforms.ToTensor(), 
                            ])
            elif args.preprocessing == 'Res256Crop224':
                transform = transforms.Compose([
                                transforms.Resize(256), 
                                transforms.CenterCrop(224),
                                transforms.ToTensor(), 
                            ])
            elif args.preprocessing == 'BicubicRes256Crop224':
                transform = transforms.Compose([
                                transforms.Resize(256, interpolation=transforms.InterpolationMode("bicubic")),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(), 
                                # transforms.Normalize(mean=target_mean, std=target_std)
                            ])
            else:
                raise NotImplementedError("check preprocessing type")
        # following the RobustBench standardization since they don't want the normalization to interfere with attacks
        # so normalization is applied in model forward pass instead of directly on data
        # but for some reason, this is only for ImageNet, using data-normalization on CIFAR10 works fine
        model = normalize_model(model, target_mean, target_std).eval()

    model = model.to(device)
    print('using', n_examples, 'for evaluation')
    dataset_ = BenchmarkDataset(dataset)
    threat_model_ = ThreatModel(args.threat_model)

    prepr = get_preprocessing(dataset_, threat_model_, model_name,
                                    preprocessing=transform)

    clean_x_test, clean_y_test = load_clean_dataset(dataset_, n_examples,
                                                        args.data_dir, prepr)

    accuracy = clean_accuracy(model,
                                clean_x_test,
                                clean_y_test,
                                batch_size=args.batch_size,
                                device=device)
    print(f'Model: {model_name}, Clean accuracy: {accuracy:.2%}')

    print('AutoAttack on', args.dataset, 'with epsilon', args.epsilon, '/ 255')
    adversary = AutoAttack(model,
                            norm=threat_model_.value,
                            eps=args.epsilon/255,
                            version=args.version,
                            device=device,
                            )
    x_adv = adversary.run_standard_evaluation(clean_x_test,
                                                clean_y_test,
                                                bs=args.batch_size,
                                                state_path=None)
    temp_name = 'aa'
    if args.version == 'rand':
        temp_name = 'aa_rand'
    if args.mask_which == 'none':
        if args.rs:
            torch.save(x_adv, f'saved_adversarial_images/{model_save_name}/{temp_name}_{n_examples}_eps{args.epsilon}by255_rs_ns{args.rs_nsmooth}_sg{rs_sigma_int}by255_basemodel.pth')
        else:
            torch.save(x_adv, f'saved_adversarial_images/{model_save_name}/{temp_name}_{n_examples}_eps{args.epsilon}by255_basemodel.pth')
    else:
        if args.rs:
            torch.save(x_adv, f'saved_adversarial_images/{model_save_name}/{temp_name}_{n_examples}_eps{args.epsilon}by255_rs_ns{args.rs_nsmooth}_sg{rs_sigma_int}by255_{args.mask_which}{args.important_dim}.pth')
        else:
            torch.save(x_adv, f'saved_adversarial_images/{model_save_name}/{temp_name}_{n_examples}_eps{args.epsilon}by255_{args.mask_which}{args.important_dim}.pth')
    adv_accuracy = clean_accuracy(model,
                                x_adv,
                                clean_y_test,
                                batch_size=args.batch_size,
                                device=device)

    print(f'AutoAttack robust accuracy: {adv_accuracy:.2%}')
