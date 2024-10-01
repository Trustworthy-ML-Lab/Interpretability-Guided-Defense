import os
import argparse
import torch
import pandas as pd
import numpy as np
from torchvision import transforms
from autoaugment import CIFAR10Policy

import matplotlib.pyplot as plt
import seaborn as sns

import utils
import similarity


parser = argparse.ArgumentParser(description='CLIP-Dissect')

parser.add_argument("--clip-model", type=str, default="ViT-B/16", 
                    choices=['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14'],
                   help="Which CLIP-model to use")
parser.add_argument("--target-model", type=str, default="resnet50", 
                   help=""""Which model to dissect, supported options are pretrained imagenet models from
                        torchvision and resnet18_places""")
parser.add_argument("--target-layers", type=str, default=["conv1", "layer1", "layer2", "layer3", "layer4"],
                    help="Which layer activations to look at. Following the naming scheme of the PyTorch module used", nargs='+')
parser.add_argument("--d-probe", type=str, default="cifar10_train", 
                    choices = ["cifar10_train", "cifar100_train", "imagenet_trainsubset"])
parser.add_argument("--load-model", type=str, help='filename of weights inside ./checkpoints folder')
parser.add_argument("--concept-set", type=str, default="data/20k.txt", help="Path to txt file containing concept set")
parser.add_argument("--preprocessing", type=str, default="Crop288", choices=["Crop288", "Res256Crop224", "BicubicRes256Crop224"], help="type of preprocessing, only for ImageNet")
parser.add_argument("--autoaug", default=False, action='store_true', help='whether to use autoaugment on the data')
parser.add_argument("--batch-size", type=int, default=200, help="Batch size when running CLIP/target model")
parser.add_argument("--device", type=str, default="cuda", help="whether to use GPU/which gpu")
parser.add_argument("--activation-dir", type=str, default="saved_activations", help="where to save activations")
parser.add_argument("--result-dir", type=str, default="results", help="where to save results")
parser.add_argument("--pool-mode", type=str, default="avg", help="Aggregation function for channels, max or avg")
parser.add_argument("--similarity-fn", type=str, default="soft_wpmi", choices=["soft_wpmi", "wpmi", "rank_reorder"])
parser.add_argument("--visualize", default=False, action='store_true', help='whether to save histogram visualizations of neuron dissection')


parser.parse_args()

if __name__ == '__main__':
    args = parser.parse_args()

    model_save_name = os.path.splitext(os.path.basename(args.load_model))[0]
    args.result_dir = os.path.join(args.result_dir, model_save_name)
    os.makedirs(args.result_dir, exist_ok=True)

    if args.d_probe == 'cifar10_train':
        cifar10_mean = (0.4914, 0.4822, 0.4465)
        cifar10_std = (0.2471, 0.2435, 0.2616)
        num_classes = 10

        if args.target_model == 'resnet18_val':
            target_preprocess = transforms.Compose([transforms.ToTensor()])

    elif args.d_probe == 'imagenet_trainsubset':
        print('using preprocessing on ImageNet :', args.preprocessing, 'along with normalization')
        target_preprocess = utils.get_resnet_imagenet_preprocess(args.preprocessing)
        num_classes = 1000
    elif args.d_probe == 'cifar100_train':
        num_classes = 100
        if args.target_model == 'preactresnet18_oaat' or args.target_model == 'wideresnet34_10':
            target_preprocess = transforms.Compose([transforms.ToTensor()])
        else:
            raise NotImplementedError("CIFAR100 only for PreActResNet18 from OAAT as of now")

    similarity_fn = eval("similarity.{}".format(args.similarity_fn))

    with open(args.concept_set, 'r') as f: 
        words = (f.read()).split('\n')
    if '' in words:
        words.remove('')
    print('Concept set:', words)

    utils.save_activations(clip_name = args.clip_model, target_name = args.target_model, load_model = args.load_model,
                           target_layers = args.target_layers, d_probe = args.d_probe, 
                           concept_set = args.concept_set, words=words, batch_size = args.batch_size, 
                           device = args.device, pool_mode=args.pool_mode, 
                           target_preprocess = target_preprocess,
                           save_dir = args.activation_dir)
    
    outputs = {"layer":[], "unit":[], "description":[], "similarity":[]}
    
    for target_layer in args.target_layers:
        save_names = utils.get_save_names(clip_name = args.clip_model, target_name = args.target_model,
                                  target_layer = target_layer, d_probe = args.d_probe,
                                  concept_set = args.concept_set, pool_mode = args.pool_mode,
                                  save_dir = args.activation_dir)
        target_save_name, clip_save_name, text_save_name = save_names

        similarities, target_feats = utils.get_similarity_from_activations(
            target_save_name, clip_save_name, text_save_name, similarity_fn, args.device, 
            dataset_name=args.d_probe, num_classes=num_classes
        )
        vals, ids = torch.max(similarities, dim=1)

        save_folder_root = '../saved_cdir_rankings'

        os.makedirs(os.path.join(save_folder_root, model_save_name, target_layer), exist_ok=True)

        for k in range(vals.shape[0]):
            np.save(os.path.join(save_folder_root, model_save_name, target_layer, 'unit' + str(k) + '.npy'), similarities[k].cpu().numpy())

        print('saved in folder ', os.path.join(save_folder_root, model_save_name, target_layer))
        
        del similarities
        torch.cuda.empty_cache()
        
        descriptions = [words[int(idx)] for idx in ids]

        if args.visualize:
            w_, c_ = np.unique(descriptions, return_counts=True)
            c_sorted = [x for _, x in sorted(zip(w_, c_))]
            w_sorted = sorted(w_)
            plt.barh(w_sorted, c_sorted)

            plt.xticks(rotation=45)
            plt.tight_layout()
            if args.keep_empty_word:
                plt.savefig(os.path.join(args.result_dir, target_layer+'_empty.png'))
            elif args.autoaug:
                plt.savefig(os.path.join(args.result_dir, target_layer+'_autoaug.png'))
            else:
                plt.savefig(os.path.join(args.result_dir, target_layer+'.png'))
            plt.clf()
