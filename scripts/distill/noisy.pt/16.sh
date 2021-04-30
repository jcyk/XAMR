export http_proxy="star-proxy.oa.com:3128"
export https_proxy="star-proxy.oa.com:3128"

pip3 install -r requirements.txt
PYTHONPATH=. python3 bin/train.py \
--config configs/config.mbart50nmt.xl.opus.yaml \
--train "/apdcephfs/share_916081/jcykcai/nonono/europarl/noisy_training/*.pt" \
--cache \
--max_cached_samples 160000 \
--ROOT /apdcephfs/share_916081/jcykcai/nonono/mbart50nmt
