# Fully unsupervised dynamic MRI reconstruction via diffeo-temporal equivariance

[Paper](https://arxiv.org/abs/2410.08646) | [Repo](https://github.com/Andrewwango/ddei) | [Website](https://andrewwango.github.io/ddei)

#### Citation

> A. Wang and M. Davies, [_"Fully unsupervised dynamic MRI reconstruction via diffeo-temporal equivariance"_](https://arxiv.org/abs/2410.08646), Oct. 2024, arXiv

## Abstract

Reconstructing dynamic MRI image sequences from undersampled accelerated measurements is crucial for faster and higher spatiotemporal resolution real-time imaging of cardiac motion, free breathing motion and many other applications. Classical paradigms, such as gated cine MRI, assume periodicity, disallowing imaging of true motion. Supervised deep learning methods are fundamentally flawed as, in dynamic imaging, ground truth fully-sampled videos are impossible to truly obtain. We propose an unsupervised framework to learn to reconstruct dynamic MRI sequences from undersampled measurements alone by leveraging natural geometric spatiotemporal equivariances of MRI. **D**ynamic **D**iffeomorphic **E**quivariant **I**maging (**DDEI**) significantly outperforms state-of-the-art unsupervised methods such as SSDU on highly accelerated dynamic cardiac imaging. Our method is agnostic to the underlying neural network architecture and can be used to adapt the latest models and post-processing approaches.

## Get started

- **Overview, pretrained demo**: see [website](https://andrewwango.github.io/ddei)
- **Data**: download the dataset from the [2023 CMRxRecon challenge website](https://www.synapse.org/Synapse:syn51471091/wiki/622170) and move the fully sampled data from `FullSample` into our folder structure in `data/`
- **Training**: to get started with training see our [training script](train.py).

## Full video results

(Fig. 3 video demo from paper)

![](img/results_fig_1.gif)