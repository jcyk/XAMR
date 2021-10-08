set -e

checkpoint=$1 
model=$2


#LANGUAGES_names=("de" "es" "it" "zh" "en")
LANGUAGES_names=("it" "zh" "en")
dataset=${MPATH}/amr_2-four_translations/data/test/test

for i in $(seq 1 1 ${#LANGUAGES_names[@]}); do
    name=${LANGUAGES_names[$i-1]}
    echo $name
    PYTHONPATH=. python3 bin/predict_amrs.py \
        --model ${model} \
        --checkpoint ${checkpoint} \
        --dataset ${dataset}_${name}.txt \
        --nproc-per-node 3 \
        --gold-path tmp-dev-gold.txt \
        --pred-path tmp-dev-pred.txt \
        --beam-size 5 \
        --batch-size 5000 \
        --penman-linearization \
        --use-pointer-tokens
done
