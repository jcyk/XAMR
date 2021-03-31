pip3 install -r requirements.txt
CUDA_VISIBLE_DEVICES=4,5 python3 bin/predict_amrs.py \
--model facebook/bart-large \
--checkpoint bart/runs/0/best-smatch_checkpoint_16_0.8395.pt  \
--datasets /apdcephfs/share_916081/jcykcai/nonono/abstract_meaning_representation_amr_2.0/data/amrs/split/test/*.txt \
--nproc-per-node 2\
--gold-path tmp-dev-gold.txt \
--pred-path tmp-dev-pred.txt \
--beam-size 5 \
--batch-size 500 \
--penman-linearization \
--use-pointer-tokens