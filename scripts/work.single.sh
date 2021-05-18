set -e

checkpoint=$1 #/apdcephfs/share_916081/jcykcai/nonono/mbart50/runs/2/best-smatch_checkpoint_30_0.8210.pt
model=$2 #facebook/mbart-large-50
name=$3 # en


dataset=/apdcephfs/share_916081/jcykcai/nonono/amr_2-four_translations/amrs/test/test

PYTHONPATH=. python3 bin/predict_amrs.py \
        --model ${model} \
        --checkpoint ${checkpoint} \
        --dataset ${dataset}_${name}.txt \
        --nproc-per-node 4 \
        --gold-path tmp-dev-gold.txt \
        --pred-path tmp-dev-pred.txt \
        --beam-size 5 \
        --batch-size 5000 \
        --penman-linearization \
        --use-pointer-tokens

