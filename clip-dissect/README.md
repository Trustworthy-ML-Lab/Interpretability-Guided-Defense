## CLIP-Dissect

Modified from [CLIP-Dissect](https://github.com/Trustworthy-ML-Lab/sandbox-clip-dissect), thanks to [Tuomas](https://github.com/tuomaso) and [Divyansh](https://github.com/somil55) for their awesome work.

### For ImageNet-train subset
1. Download the [ImageNet 2012 training subset](https://image-net.org/challenges/LSVRC/2012/2012-downloads.php) (Warning: it is ~137 GB in size). The login is freely accessible via an academic email ID.
2. Preprocess the zip file into `ImageFolder` format (each class having its own folder of images) by following [these instructions](https://github.com/soumith/imagenet-multiGPU.torch#data-processing).
3. Create a balanced 10% subset of all 1000 classes using [this repo](https://github.com/BenediktAlkin/ImageNetSubsetGenerator). Specifically, 

```
python3 main_subset.py --in1k_path <ImageNet1K_path> --out_path <out_path> --version  in1k_10percent_seed0 --mode imagefolder
```

This creates a new folder at `<out_path>` with the subset (~14 GB in size) which can be loaded with `torchvision.datasets.ImageFolder` (same as for full training set). In future, will modify this to get a text file of filenames instead to avoid using extra storage space.

### CLIP-Dissect Importance Ranking
The following bash file runs importance ranking for the specified pretrained checkpoint
```
bash describe_neurons.sh
```
