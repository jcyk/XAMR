set -e
export http_proxy="star-proxy.oa.com:3128"
export https_proxy="star-proxy.oa.com:3128"

pip3 install -r requirements.txt

ckpt=/apdcephfs/share_916081/jcykcai/nonono/bart/runs/0/best-smatch_checkpoint_5_0.8441.pt
model=facebook/bart-large

PYTHONPATH=. python3 bin/predict_amrs_from_plaintext.py \
--model ${model} \
--checkpoint ${ckpt}  \
--nproc-per-node 8 \
--input-path /apdcephfs/share_916081/jcykcai/nonono/europarl/europarl.long.en.txt \
--output-path /apdcephfs/share_916081/jcykcai/nonono/europarl/europarl.long.en.amr \
--beam-size 5 \
--batch-size 3000 \
--penman-linearization \
--use-pointer-tokens

