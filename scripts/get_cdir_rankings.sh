
## need to use clip-dissect directory (we keep it separate to keep our high-level code repo simple)
cd clip-dissect/

python3 -u get_cdir_rankings.py --target-model resnet18_val --d-probe cifar10_train --concept-set data/cifar10_classes.txt \
--load-model Addepalli2022Efficient_RN18.pt

# python3 -u get_cdir_rankings.py --target-model wideresnet34_10 --d-probe cifar10_train --concept-set data/cifar10_classes.txt \
# --load-model TRADES-AWP_cifar10_linf_wrn34-10.pt

# python3 -u get_cdir_rankings.py --target-model resnet50 --d-probe imagenet_trainsubset --concept-set data/in1k_classes.txt \
# --load-model imagenet_model_weights_4px.pth.tar

cd ../
