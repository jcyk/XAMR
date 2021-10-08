export http_proxy="star-proxy.oa.com:3128"
export https_proxy="star-proxy.oa.com:3128"

pip3 install -r requirements.txt
PYTHONPATH=. python3 bin/train.py \
--config configs/config.mbart50.xl.opus.yaml \
--checkpoint /apdcephfs/share_916081/jcykcai/nonono/mbart50nmt/noise/pt/$1/*.pt \
--ROOT /apdcephfs/share_916081/jcykcai/nonono/mbart50nmt/noise/ft
