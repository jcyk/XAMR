set -e

checkpoint=ckpt/0/best-smatch_checkpoint_4_0.8408.pt
model=facebook/bart-large


for lang in de es it zh; do
test_data=xl-amr/data/AMR/amr_2.0/kd/${lang}_train.txt

PYTHONPATH=. python3 bin/predict_amrs.py \
        --model ${model} \
        --checkpoint ${checkpoint} \
        --dataset ${test_data} \
        --nproc-per-node 6 \
        --gold-path  ${lang}-dev-gold.txt \
        --pred-path  ${lang}-dev-pred.txt \
        --beam-size 5 \
        --batch-size 5000 \
        --penman-linearization \
        --use-pointer-tokens
done

