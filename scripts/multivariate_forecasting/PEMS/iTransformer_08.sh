export CUDA_VISIBLE_DEVICES=2

model_name=iTransformer

for fix_seed in 2022 2023 2024 2025 2026; do
python -u run.py \
  --is_training 1 \
  --fix_seed $fix_seed \
  --root_path ./dataset/PEMS/ \
  --data_path PEMS08.npz \
  --model_id PEMS08_96_12 \
  --model $model_name \
  --data PEMS \
  --features M \
  --seq_len 96 \
  --pred_len 12 \
  --e_layers 2 \
  --enc_in 170 \
  --dec_in 170 \
  --c_out 170 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --itr 1 \
  --use_norm 1 > ./logs/PEMS08/'pl12_seed'$fix_seed.log 
done