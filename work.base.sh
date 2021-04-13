set -e

checkpoint=$1 #/apdcephfs/share_916081/jcykcai/nonono/mbart50/runs/2/best-smatch_checkpoint_30_0.8210.pt
model=$2 #facebook/mbart-large-50

LANGUAGES_codes=("de_DE" "es_XX" "it_IT" "zh_CN")
LANGUAGES_names=("de" "es" "it" "zh")
dataset=/apdcephfs/share_916081/jcykcai/nonono/amr_2-four_translations/amrs/test/test
dataset_en=/apdcephfs/share_916081/jcykcai/nonono/abstract_meaning_representation_amr_2.0/data/amrs/split/test/*.txt

PYTHONPATH=. python3 bin/predict_amrs.py \
--model ${model} \
--checkpoint ${checkpoint}  \
--datasets ${dataset_en} \
--nproc-per-node 4 \
--gold-path tmp-dev-gold.txt \
--pred-path tmp-dev-pred.txt \
--beam-size 5 \
--batch-size 5000 \
--penman-linearization \
--use-pointer-tokens

for i in $(seq 1 1 ${#LANGUAGES_codes[@]}); do
    code=${LANGUAGES_codes[$i-1]}
    name=${LANGUAGES_names[$i-1]}
    echo ${code}, ${name}
    PYTHONPATH=. python3 bin/predict_amrs.py \
        --src_lang ${code} \
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
done
