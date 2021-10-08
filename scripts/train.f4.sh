PYTHONPATH=. python3 bin/train.py \
--config configs/config.mbart50nmt.xl.opus.yaml \
--checkpoint ${MPATH}/mbart50nmt/f3/*.pt \
--ROOT ${MPATH}/mbart50nmt/f4

