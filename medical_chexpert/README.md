## Training instructions for chexpert
This code is based on the implementations of [**Medical MAE**](https://github.com/lambert-x/medical_mae), 
```
python -m torch.distributed.launch --nproc_per_node=4 --use_env main_finetune_chestxray.py \
                                   --batch_size 32 --epochs 100 --lr 1e-3 --layer_decay 0.75 \
                                   --weight_decay 1e-3 --model resnet50_clip --warmup_epochs 10 \
                                   --drop_path 0.2 --mixup 0. --cutmix 0. --reprob 0. --vit_dropout_rate 0 \
                                   --num_workers 4 --nb_classes 5 --eval_interval 2 --min_lr 1e-5 --build_timm_transform --aa 'rand-m6-mstd0.5-inc1' \
                                   --dataset chexpert --data_path /path/to/CheXpert-v1.0-small/ \
                                   --train_list /path/to/Chexpert/CheXpert-v1.0-small/train.csv \
                                   --val_list /path/to/Chexpert/CheXpert-v1.0-small/valid.csv \
                                   --test_list /path/to/CheXpert-v1.0-small/valid.csv --accum_iter 8 --pretrained --input_size 256 \
                                   --n_shot 0.1 --output_dir output_dir
```
