# RTL

This repo is the official implementation of [Test-Time Linear Out-of-Distribution Detection](https://openaccess.thecvf.com/content/CVPR2024/papers/Fan_Test-Time_Linear_Out-of-Distribution_Detection_CVPR_2024_paper.pdf).

# Environment
```
conda create -n RTL python=3.10
conda activate RTL
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install scikit-learn pandas scikit-image
```

# Datasets and checkpoints
For CIFAR, we use [Textures](https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz), [Places365](https://data.csail.mit.edu/places/places365/test_256.tar), [LSUN-C](https://drive.google.com/file/d/1JsPu88ThWMOxWaYYN9j6IFypbJ2oO990/view?usp=sharing), [LSUN-R](https://drive.google.com/file/d/1dF96t4QzwDHDBk9fD3DWZobw7mv-dE-q/view?usp=sharing), [iSUN](https://drive.google.com/file/d/1jCFWHZy8iWGB0E1KqdnbAy0XdcQ22vy3/view?usp=sharing) and [SVHN](http://ufldl.stanford.edu/housenumbers/test_32x32.mat) as our OOD datasets. Please organize the datasets as follows:
```Bash
├── dtd
│   └── images
├── iSUN
│   └── iSUN_patches
├── LSUN
│   └── test
├── LSUN_resize
│   └── LSUN_resize
├── Places365
│   └── test_256
└── SVHN
    └── test_32x32.mat
```

For ImageNet, we follow [gradnorm_ood](https://github.com/deeplearning-wisc/gradnorm_ood) and use their dataset splits [iNaturalist](http://pages.cs.wisc.edu/~huangrui/imagenet_ood_dataset/iNaturalist.tar.gz), [SUN](http://pages.cs.wisc.edu/~huangrui/imagenet_ood_dataset/SUN.tar.gz), [Places](http://pages.cs.wisc.edu/~huangrui/imagenet_ood_dataset/Places.tar.gz) and [Textures](https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz). For fair comparision, we use their dataset splits of iNaturalist, Places and SUN and their [checkpoint](http://pages.cs.wisc.edu/~huangrui/finetuned_model/BiT-S-R101x1-flat-finetune.pth.tar). Please organize the datasets as follows:
```Bash
├── dtd
│   ├── images
├── iNaturalist
│   └── images
├── Places
│   └── images
└──  SUN
    └── images
```

# Evaluation
For CIFAR
```Bash
cd CIFAR/CIFAR/
python RTL.py/RTL_plus.py --method_name cifar10_wrn_pretrained/cifar100_wrn_pretrained --score MSP/energy/xent --exp_num 0 --alpha 1e-5 --T 1 --num_to_avg 10
python RTL.py/RTL_plus.py --method_name cifar10_wrn_pretrained/cifar100_wrn_pretrained --score Odin --exp_num 0 --alpha 1e-5 --T 1 --noise  0.0024 --num_to_avg 10
```
For ImageNet
```Bash
cd ImageNet
python feature_extraction.py --in_datadir link_to_imagenet1k_val --out_datadir link_to_ood_datasets --model BiT-S-R101x1 --model_path checkpoints/BiT-S-R101x1-flat-finetune.pth.tar --batch 32
python RTL.py/RTL_plus.py --score MSP/energy/ODIN/xent --alpha 1e-7 --reduce_method pca --reduce_dim 32
```

# Citation
If you find our paper useful for your research and applications, please cite using this BibTeX:
```bibtex
@InProceedings{Fan_2024_CVPR,
    author    = {Fan, Ke and Liu, Tong and Qiu, Xingyu and Wang, Yikai and Huai, Lian and Shangguan, Zeyu and Gou, Shuang and Liu, Fengjian and Fu, Yuqian and Fu, Yanwei and Jiang, Xingqun},
    title     = {Test-Time Linear Out-of-Distribution Detection},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {23752-23761}
}
```