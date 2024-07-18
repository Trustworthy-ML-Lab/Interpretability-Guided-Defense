# Interpretability-Guided-Defense

* This is the official repository for the ECCV 2024 paper: "Interpretability-Guided Test-Time Adversarial Defense"
  * We propose the first neuron-interpretability-guided test-time defense (**IG-Defense**) utilizing neuron importance ranking to improve adversarial robustness. IG-Defense is training-free, efficient, and effective.
  * We uncover novel insights into improving adversarial robustness by analyzing adversarial attacks through the lens of neuron-level interpretability.
  * Our proposed **IG-Defense** consistently improves the robustness on standard CIFAR10, CIFAR100, and ImageNet-1k benchmarks.
  * We also demonstrate improved robustness upto 3.4\%, 3.8\%, and 1.5\% against a wide range of white-box, black-box, and adaptive attacks respectively with the lowest inference time (4x faster) among existing test-time defenses.
* We illustrate the overview of our IG-Defense below. For more information about IG-Defense, please check out our [project page](https://lilywenglab.github.io/Interpretability-Guided-Defense/).

<p align="center">
<img src="https://github.com/user-attachments/assets/77826fb2-72e6-4b97-9fc8-f7d14a596792" width="800">
</p>

## Sources:
* CLIP-Dissect: https://github.com/Trustworthy-ML-Lab/CLIP-dissect
* RobustBench: https://robustbench.github.io/
* Adaptive Attack Evaluations: https://github.com/fra31/evaluating-adaptive-test-time-defenses 

## Cite this work
A. Kulkarni and T.-W. Weng, "Interpretability-Guided Test-Time Adversarial Defense", ECCV 2024.

```
  @inproceedings{kulkarni2024interpretabilityguided,
      title={Interpretability-Guided Test-Time Adversarial Defense},
      author={Akshay Kulkarni, and Tsui-Wei Weng},
      booktitle={European Conference on Computer Vision},
      year={2024}
  }
```
