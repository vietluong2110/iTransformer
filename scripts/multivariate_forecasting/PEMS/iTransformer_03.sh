export CUDA_VISIBLE_DEVICES=1

model_name=iTransformer

for fix_seed in 2022 2023 2024 2025 2026; do
python -u run.py \
  --fix_seed $fix_seed \
  --is_training 1 \
  --root_path ./dataset/PEMS/ \
  --data_path PEMS03.npz \
  --model_id PEMS03_96_12 \
  --model $model_name \
  --data PEMS \
  --features M \
  --seq_len 96 \
  --pred_len 12 \
  --e_layers 4 \
  --enc_in 358 \
  --dec_in 358 \
  --c_out 358 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --learning_rate 0.001 \
  --itr 1 >./logs/PEMS03/'pl12_seed'$fix_seed.log 
done