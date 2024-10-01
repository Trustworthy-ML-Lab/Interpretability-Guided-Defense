

## evaluating base model
# python3 -u eval.py --arch rn18 --load-model Addepalli2022Efficient_RN18.pt --mask-which none

## evaluating CD-IR defended model
python3 -u eval.py --dataset cifar10 --arch rn18 --load-model Addepalli2022Efficient_RN18.pt \
--mask-which cdir --layer-name layer4 --important-dim 50 --rs --rs-sigma 4 --rs-nsmooth 1

## evaluating LO-IR defended model
# python3 -u eval.py --dataset cifar10 --arch rn18 --load-model Addepalli2022Efficient_RN18.pt \
# --mask-which loir --layer-name layer4 --important-dim 50 --rs --rs-sigma 4 --rs-nsmooth 1

