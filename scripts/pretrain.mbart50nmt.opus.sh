export http_proxy="star-proxy.oa.com:3128"
export https_proxy="star-proxy.oa.com:3128"

pip3 install -r requirements.txt
PYTHONPATH=. python3 bin/train.py \
--config configs/config.mbart50nmt.xl.opus.yaml \
--ROOT ../tmp
#--train /apdcephfs/share_916081/jcykcai/nonono/cache/train.opus.mbart50nmt.pt \
#--cache \
#--ROOT /apdcephfs/share_916081/jcykcai/nonono/mbart50nmt
