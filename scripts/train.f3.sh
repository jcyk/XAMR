PYTHONPATH=. python3 bin/train.py \
--config configs/config.mbart50nmt.xl.opus.yaml \
--train "${MPATH}/europarl/noisy_training/*.pt" \
--cache \
--max_cached_samples 320000 \
--ROOT ${MPATH}/mbart50nmt/f3

