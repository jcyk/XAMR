export http_proxy="star-proxy.oa.com:3128"
export https_proxy="star-proxy.oa.com:3128"

pip3 install -r requirements.txt
PYTHONPATH=. python3 bin/train.py \
--config configs/config.mbart50nmt.xl.opus.yaml \
--train "/apdcephfs/share_916081/jcykcai/nonono/europarl/training/*.pt" \
--cache \
--max_cached_samples 320000 \
--noise $1 \
--ROOT /apdcephfs/share_916081/jcykcai/nonono/mbart50nmt/noise/pt


#PYTHONPATH=. python3 bin/train.py \
#--config configs/config.mbart50.xl.opus.yaml \
#--checkpoint /apdcephfs/share_916081/jcykcai/nonono/mbart50/p1f1f2/pt/0/*.pt \
#--ROOT /apdcephfs/share_916081/jcykcai/nonono/mbart50/p1f1f2/ft
