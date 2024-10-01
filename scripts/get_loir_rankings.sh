
python3 -u get_loir_rankings.py --layer-name layer4 --arch rn18_val --load-model Addepalli2022Efficient_RN18.pt \
--out-dir saved_loir_rankings --batch-size 1000

# python3 -u get_loir_rankings.py --layer-name block3 --arch wrn34_10 --load-model TRADES-AWP_cifar10_linf_wrn34-10.pt \
# --out-dir saved_loir_rankings --batch-size 500

#### this may take a long time (since rn50 layer4 has 2048 neurons and ImageNet has a larger input resolution), so you may want to split this into 
#### multiple runs on different GPUs using `--start-dim 0 --end-dim 100`, `--start-dim 100 --end-dim 200`, and so on
# python3 -u get_loir_rankings.py --dataset imagenet --layer-name layer4 --arch rn50 --load-model imagenet_model_weights_4px.pth.tar \
# --out-dir saved_loir_rankings --batch-size 100
