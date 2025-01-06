## 1. Finetune the VQ-VAE
```bash
CUDA_VISIBLE_DEVICES=3
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
    --max_epochs 100  > $(date +%m%d).log 2>&1 &
```

## 2. Finetune the VideoGPT
```bash
python scripts/train_videogpt.py \
    --vqvae "lightning_logs/version_0/checkpoints/model.ckpt" \
    --n_cond_frames 0 \
    --class_cond \
    --hidden_dim 576 \
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
    --data_path data/dummy_data \
    --resolution 128 \
    --sequence_length 16 \
    --max_epochs 1 \
    --max_steps 10 \
    --learning_rate 1e-4
```