# BoB-Classification

This repository is the official implementation of <strong>Image Classification</strong> task in the [*Battle of the Backbones: A Large-Scale Comparison of Pretrained Models across Computer Vision Tasks*](https://github.com/hsouri/Battle-of-the-Backbones).

## Dependencies

Version Control of Python libraries in environment.yml file. To create a virtual environment:
```bash
conda env create -f environment.yml
```
## Running instructions
Following instructions introduce how to evaluate pretrained backbones with different classficaition tasks. We provide various datasets and pre-trained models. 

### Finetuning (convnext_xl pretrained with vicregl)
```
python3 -m torch.distributed.launch --nproc_per_node=8 train.py /path/to/ImageNet/ILSVRC2012 --dataset ImageNet --config ./model_configs/convnext_xl_vicreg_ft.yaml --lr 1e-3 --experiment convnext_xl_vicreg_ft_lr1e-3
```

### 1\% and 10\% of ImageNet training (stable diffusion pretrained model)
```
python3 -m torch.distributed.launch --nproc_per_node=8 train_semi.py /path/to/ImageNet/ILSVRC2012 --dataset semiimagenet --n_shot 10 --config ./model_configs/vit_small_patch16_224_dino_semi.yaml -b 16 --experiment semi_sd_adamw_lr2.5e-4_wd5e-2_ld65_shot10_da --weight-decay 0.05 --lr 2.5e-4 --drop-path 0.1 --epochs 60 --model stable_diffusion_v1 --layer-decay 0.65 --aa rand-m9-mstd0.5-inc1 --accum_iter 4
```

### Linear Probing (CLIP trained resnet50)
```
python3 -m torch.distributed.launch --nproc_per_node=4 linear_probe.py --data_path /fs/cml-datasets/ImageNet/ILSVRC2012/ --batch_size 512 --epochs 90 --blr 0.1 --weight_decay 0.0 --dist_eval --model resnet50_clip
```
This code is based on the implementations of [**MAE**](https://github.com/facebookresearch/mae), [**TIMM**](https://github.com/huggingface/pytorch-image-models)
