export http_proxy="star-proxy.oa.com:3128"
export https_proxy="star-proxy.oa.com:3128"

pip3 install -r requirements.txt
PYTHONPATH=. python3 bin/train.py \
--config configs/debug.yaml \
--ROOT ../tmp
