## 1. Finetune the VQ-VAE
```bash
export CUDA_VISIBLE_DEVICES=0
nohup python scripts/train_vqvae.py \
    --embedding_dim 256 \
    --n_codes 2048 \
    --n_hiddens 240 \
    --n_res_layers 4 \
    --downsample 4 4 4 \
    --gpus 1 \
    --sync_batchnorm \
    --gradient_clip_val 1 \
    --batch_size 16 \
    --num_workers 8 \
    --learning_rate 1e-4 \
    --max_epochs 100 \
    --data_path data/deposition_data_video_split \
    --resolution 128 \
    --sequence_length 16 \
    --max_epochs 100  > $(date +%m%d)"vqvae".log 2>&1 &
```

## 2. Finetune the VideoGPT
记得把第一帧设置为条件帧:
```bash
--n_cond_frames 1 \ 
```
其中`    --class_cond ` 没必要使用。

```bash
export CUDA_VISIBLE_DEVICES=1
nohup python scripts/train_videogpt.py \
    --hidden_dim 576 \
    --n_cond_frames 1 \
    --heads 4 \
    --layers 8 \
    --dropout 0.2 \
    --attn_type full \
    --attn_dropout 0.3 \
    --gpus 1 \
    --gradient_clip_val 1 \
    --batch_size 8 \
    --num_workers 2 \
    --amp_level O1 \
    --precision 16 \
    --resolution 128 \
    --sequence_length 16 \
    --max_epochs 100 \
    --max_steps 1000000 \
    --data_path data/deposition_data_video_split \
    --vqvae "checkpoints/vqvae/vqvae_final.ckpt" \
    --learning_rate 1e-4 > $(date +%m%d)"videogpt_finetune".log 2>&1 &
```

## 3.infer for the test data
```bash
export CUDAVISIBLE_DEVICES=0 && nohup python scripts/sample_videogpt.py --ckpt 'checkpoints/videogpt/videogpt_epoch=99.ckpt' --image > $(date +%m%d)"infer".log 2>&1 &
```