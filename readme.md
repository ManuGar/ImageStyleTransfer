# How to use Image Style Transfer

To use this API, the user only has to fix 3 parameters: the algorithm that wants to use, the name of the dataset and output path.
The dataset have to be inside a folder called datasets and divided onto trainA and trainB folders. TrainA folder contains the datset of images that the user want to changes style. TrainB contains the images of the style the user wants to obtain.


There are a list of algorithms available to transfer image style divided into two different ways:
- Style transfer based  on CNNs:
    - nst (Neural Style Transfer): Included in this project
    - strotss (STROTSS): https://github.com/nkolkin13/STROTSS
    - dia (Deep Image Analogy): https://github.com/harveyslash/Deep-Image-Analogy-PyTorch

- style transfer based  on GANs:
    - upit (CycleGAN): https://github.com/tmabraham/upit.git
    - dualGAN (DualGAN):  https://github.com/duxingren14/DualGAN
    - forkGAN (ForkGAN): https://github.com/zhengziqiang/ForkGAN
    - ganilla (GANILLA): https://github.com/giddyyupp/ganilla
    - CUT: https://github.com/taesungp/contrastive-unpaired-translation
    - fastCUT (FastCUT): https://github.com/taesungp/contrastive-unpaired-translation
