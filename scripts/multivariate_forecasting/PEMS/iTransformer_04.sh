export CUDA_VISIBLE_DEVICES=2

model_name=iTransformer

for fix_seed in 2022 2023 2024 2025 2026; do

python -u run.py \
  --is_training 1 \
  --fix_seed $fix_seed \
  --root_path ./dataset/PEMS/ \
  --data_path PEMS04.npz \
  --model_id PEMS04_96_12 \
  --model $model_name \
  --data PEMS \
  --features M \
  --seq_len 96 \
  --pred_len 12 \
  --e_layers 4 \
  --enc_in 307 \
  --dec_in 307 \
  --c_out 307 \
  --des 'Exp' \
  --d_model 1024 \
  --d_ff 1024 \
  --learning_rate 0.0005 \
  --itr 1 \
  --use_norm 0 > ./logs/PEMS04/'pl12_seed'$fix_seed.log 
done