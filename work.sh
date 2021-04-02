set -e

pip3 install -r requirements.txt

#checkpoint=/apdcephfs/share_916081/jcykcai/nonono/bartA/runs/5/best-smatch_checkpoint_17_0.8411.pt

#CUDA_VISIBLE_DEVICES=3,4,5,6 PYTHONPATH=. python3 bin/predict_amrs.py \
#--model facebook/bart-large \
#--checkpoint ${checkpoint}  \
#--datasets /apdcephfs/share_916081/jcykcai/nonono/abstract_meaning_representation_amr_2.0/data/amrs/split/test/*.txt \
#--nproc-per-node 4 \
#--gold-path tmp-dev-gold.txt \
#--pred-path tmp-dev-pred.txt \
#--beam-size 5 \
#--batch-size 500 \
#--penman-linearization \
#--use-pointer-tokens

checkpoint=/apdcephfs/share_916081/jcykcai/nonono/bart/runs/3/best-smatch_checkpoint_1_0.1867.pt

CUDA_VISIBLE_DEVICES=3,4,5,6 PYTHONPATH=. python3 bin/predict_amrs.py \
--model facebook/bart-large \
--checkpoint ${checkpoint}  \
--datasets /apdcephfs/share_916081/jcykcai/nonono/abstract_meaning_representation_amr_2.0/data/amrs/split/test/*.txt \
--nproc-per-node 4 \
--gold-path tmp-dev-gold.txt \
--pred-path tmp-dev-pred.txt \
--beam-size 5 \
--batch-size 500 \
--penman-linearization \
--use-pointer-tokens


exit 0
#en_XX, de_DE, zh_CN, es_XX, it_IT

checkpoint=/apdcephfs/share_916081/jcykcai/nonono/mbart50/runs/2/best-smatch_checkpoint_30_0.8210.pt

dataset=/apdcephfs/share_916081/jcykcai/nonono/amr_2-four_translations/amrs/test

CUDA_VISIBLE_DEVICES=3,4,5,6 PYTHONPATH=. python3 bin/predict_amrs.py \
--src_lang de_DE \
--model facebook/mbart-large-50 \
    --checkpoint ${checkpoint} \
    --dataset ${dataset}_de.txt \
     --nproc-per-node 4 \
     --gold-path tmp-dev-gold.txt \
     --pred-path tmp-dev-pred.txt \
     --beam-size 5 \
     --batch-size 500 \
     --penman-linearization \
     --use-pointer-tokens

CUDA_VISIBLE_DEVICES=3,4,5,6 PYTHONPATH=. python3 bin/predict_amrs.py \
--src_lang zh_CN \
--model facebook/mbart-large-50 \
    --checkpoint ${checkpoint} \
    --dataset ${dataset}_zh.txt \
     --nproc-per-node 4 \
     --gold-path tmp-dev-gold.txt \
     --pred-path tmp-dev-pred.txt \
     --beam-size 5 \
     --batch-size 500 \
     --penman-linearization \
     --use-pointer-tokens

CUDA_VISIBLE_DEVICES=3,4,5,6 PYTHONPATH=. python3 bin/predict_amrs.py \
--src_lang es_XX \
--model facebook/mbart-large-50 \
    --checkpoint ${checkpoint} \
    --dataset ${dataset}_es.txt \
     --nproc-per-node 4 \
     --gold-path tmp-dev-gold.txt \
     --pred-path tmp-dev-pred.txt \
     --beam-size 5 \
     --batch-size 500 \
     --penman-linearization \
     --use-pointer-tokens

CUDA_VISIBLE_DEVICES=3,4,5,6 PYTHONPATH=. python3 bin/predict_amrs.py \
--src_lang it_IT \
--model facebook/mbart-large-50 \
    --checkpoint ${checkpoint} \
    --dataset ${dataset}_it.txt \
     --nproc-per-node 4 \
     --gold-path tmp-dev-gold.txt \
     --pred-path tmp-dev-pred.txt \
     --beam-size 5 \
     --batch-size 500 \
     --penman-linearization \
     --use-pointer-tokens

checkpoint=/apdcephfs/share_916081/jcykcai/nonono/mbart50nmt/runs/2/best-smatch_checkpoint_19_0.8287.pt
CUDA_VISIBLE_DEVICES=3,4,5,6 PYTHONPATH=. python3 bin/predict_amrs.py \
--src_lang de_DE \
--model facebook/mbart-large-50-many-to-many-mmt \
    --checkpoint ${checkpoint} \
    --dataset ${dataset}_de.txt \
     --nproc-per-node 4 \
     --gold-path tmp-dev-gold.txt \
     --pred-path tmp-dev-pred.txt \
     --beam-size 5 \
     --batch-size 500 \
     --penman-linearization \
     --use-pointer-tokens

CUDA_VISIBLE_DEVICES=3,4,5,6 PYTHONPATH=. python3 bin/predict_amrs.py \
--src_lang zh_CN \
--model facebook/mbart-large-50-many-to-many-mmt \
    --checkpoint ${checkpoint} \
    --dataset ${dataset}_zh.txt \
     --nproc-per-node 4 \
     --gold-path tmp-dev-gold.txt \
     --pred-path tmp-dev-pred.txt \
     --beam-size 5 \
     --batch-size 500 \
     --penman-linearization \
     --use-pointer-tokens

CUDA_VISIBLE_DEVICES=3,4,5,6 PYTHONPATH=. python3 bin/predict_amrs.py \
--src_lang es_XX \
--model facebook/mbart-large-50-many-to-many-mmt \
    --checkpoint ${checkpoint} \
    --dataset ${dataset}_es.txt \
     --nproc-per-node 4 \
     --gold-path tmp-dev-gold.txt \
     --pred-path tmp-dev-pred.txt \
     --beam-size 5 \
     --batch-size 500 \
     --penman-linearization \
     --use-pointer-tokens

CUDA_VISIBLE_DEVICES=3,4,5,6 PYTHONPATH=. python3 bin/predict_amrs.py \
--src_lang it_IT \
--model facebook/mbart-large-50-many-to-many-mmt \
    --checkpoint ${checkpoint} \
    --dataset ${dataset}_it.txt \
     --nproc-per-node 4 \
     --gold-path tmp-dev-gold.txt \
     --pred-path tmp-dev-pred.txt \
     --beam-size 5 \
     --batch-size 500 \
     --penman-linearization \
     --use-pointer-tokens
