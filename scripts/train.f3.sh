PYTHONPATH=. python3 bin/train.py \
--config configs/config.mbart50nmt.xl.opus.yaml \
--train 'xl-amr/data/AMR/amr_2.0/kd/*.pt' \
--cache \
--max_cached_samples 320000 \
--ROOT ckpt/f3

