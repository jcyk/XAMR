set -e

ckpt=/apdcephfs/share_916081/jcykcai/nonono/bart/runs/0/best-smatch_checkpoint_5_0.8441.pt
model=facebook/bart-large

dataset=/apdcephfs/share_916081/jcykcai/nonono/amr_2-four_translations/amrs/dev/dev_en.txt
PYTHONPATH=. python3 bin/predict_amrs.py \
--model ${model} \
--checkpoint ${ckpt}  \
--datasets ${dataset} \
--nproc-per-node 4 \
--gold-path gold.dev_en.txt \
--pred-path seqdistill.dev_en.txt \
--beam-size 5 \
--batch-size 5000 \
--penman-linearization \
--use-pointer-tokens

