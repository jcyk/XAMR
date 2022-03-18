set -e

checkpoint=ckpt/best.pt
model=facebook/mbart-large-50-many-to-many-mmt

for dataset in sts; do
    PYTHONPATH=. python3 bin/predict_amrs.py \
        --model ${model} \
        --checkpoint ${checkpoint} \
        --dataset ${dataset}.txt \
        --nproc-per-node 6 \
        --gold-path ${dataset}-test-gold.txt \
        --pred-path ${dataset}-test-pred.txt \
        --beam-size 4 \
        --batch-size 5000 \
        --penman-linearization \
        --use-pointer-tokens
done
