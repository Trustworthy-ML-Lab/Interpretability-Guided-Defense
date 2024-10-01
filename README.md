# Interpretability-Guided-Defense

* This is the official repository for the ECCV 2024 paper: "Interpretability-Guided Test-Time Adversarial Defense"
  * We propose the first neuron-interpretability-guided test-time defense (**IG-Defense**) utilizing neuron importance ranking to improve adversarial robustness. IG-Defense is training-free, efficient, and effective.
  * We uncover novel insights into improving adversarial robustness by analyzing adversarial attacks through the lens of neuron-level interpretability.
  * Our proposed **IG-Defense** consistently improves the robustness on standard CIFAR10, CIFAR100, and ImageNet-1k benchmarks.
  * We also demonstrate improved robustness upto 3.4\%, 3.8\%, and 1.5\% against a wide range of white-box, black-box, and adaptive attacks respectively with the lowest inference time (4x faster) among existing test-time defenses.
* We illustrate the overview of our IG-Defense below. For more information about IG-Defense, please check out our [project page](https://lilywenglab.github.io/Interpretability-Guided-Defense/).

<p align="center">
<img src="https://github.com/user-attachments/assets/5cd73bf7-c8c7-4707-8828-a6be5ad21c64" width="900">
</p>

## Requirements

Python 3.6 (not very strict though, anything 3.6+ should work out)

Install the required packages:
```
pip install -r requirements.txt
pip install git+https://github.com/RobustBench/robustbench.git
```

### Pretrained weights

* Download [DAJAT ResNet18 CIFAR10 pretrained weights](https://drive.google.com/uc?id=1m5vhdzIUUKhDbsZdOG9z76Eyp6f4xe_f), [TRADES-AWP WideResNet-34-10 CIFAR10 pretrained weights](https://drive.google.com/uc?id=1hlVTLZkveYGWpE9-46Wp5NVZt1slz-1T), and [FAT ResNet50 ImageNet-1k pretrained weights](https://drive.google.com/uc?id=1UrNEtLWs-fjlM2GPb1JpBGtpffDuHH_4) to the `checkpoints/` directory.
* Other pretrained weights can be used from the [RobustBench model zoo](https://github.com/RobustBench/robustbench/tree/master/robustbench/model_zoo). The corresponding model code needs to be added to the `models/` directory (and modified similar to given example models).

## Neuron Importance Ranking Methods

* The scripts below can be used to obtain the CLIP-Dissect (CD-IR) and Leave-one-Out (LO-IR) neuron importance rankings.
* By default, they are for DAJAT ResNet18 pretrained weights, but commented out examples are given for TRADES-AWP and FAT ResNet50 (ImageNet) models.
    * For ImageNet, modify L130,131 of `utils.py` and L19 of `clip-dissect/utils.py` with the path to ImageNet dataset. We did not use the entire ImageNet training set since it takes too long, we created a random 10% train-subset using [this code repo](https://github.com/BenediktAlkin/ImageNetSubsetGenerator).

    ```
    bash scripts/get_cdir_rankings.sh
    bash scripts/get_loir_rankings.sh
    ```

## Analysis Experiment

* The analysis experiment (Fig. 2 in the paper) uses the LO-IR neuron importance rankings, so please run it first using `bash scripts/get_loir_rankings.sh`.
* After this, we can run the analysis experiment (by default for [DAJAT](https://arxiv.org/abs/2210.15318) ResNet18 pretrained model):

    ```
    bash scripts/analysis.sh
    ```

## AutoAttack Evaluation

* Standard AutoAttack evaluation can be run for the base model, CD-IR defended model and LO-IR defended model (by default for pretrained DAJAT RN18 CIFAR10 model) using

    ```
    bash scripts/eval.sh
    ```

* Adaptive attack evaluation will be released soon.

## Sources
* CLIP-Dissect: https://github.com/Trustworthy-ML-Lab/CLIP-dissect
* RobustBench: https://robustbench.github.io/

## Cite this work
A. Kulkarni and T.-W. Weng, Interpretability-Guided Test-Time Adversarial Defense, ECCV 2024.

```
@inproceedings{kulkarni2024igdefense,
    title={Interpretability-Guided Test-Time Adversarial Defense},
    author={Kulkarni, Akshay and Weng, Tsui-Wei},
    booktitle={European Conference on Computer Vision},
    year={2024}
}
```
