export http_proxy="star-proxy.oa.com:3128"
export https_proxy="star-proxy.oa.com:3128"

pip3 install -r requirements.txt
PYTHONPATH=. python3 bin/train.py \
--config configs/config.mbart50nmt.xl.opus.yaml \
--train "/apdcephfs/share_916081/jcykcai/nonono/amr_2-four_translations/amrs/dev/opus/*.txt" \
--es \
--es_stop_after 2000 \
--ROOT ../tmp
