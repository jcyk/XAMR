set -e

checkpoint=$1 
model=facebook/mbart-large-50-many-to-many-mmt

for lang in de es it zh en; do
    dataset=xl-amr/data/AMR/amr_2.0/test/test_${lang}.txt
    PYTHONPATH=. python3 bin/predict_amrs.py \
        --model ${model} \
        --checkpoint ${checkpoint} \
        --dataset ${dataset} \
        --nproc-per-node 4 \
        --gold-path ${lang}-test-gold.txt \
        --pred-path ${lang}-test-pred.txt \
        --beam-size 4 \
        --batch-size 5000 \
        --penman-linearization \
        --use-pointer-tokens
done
