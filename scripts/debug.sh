export http_proxy="star-proxy.oa.com:3128"
export https_proxy="star-proxy.oa.com:3128"

pip3 install -r requirements.txt


PYTHONPATH=. python3 bin/train.py \
    --config configs/config.mbart50nmt.xl.opus.yaml \
    --train /apdcephfs/share_916081/jcykcai/nonono/europarl/noisy_training/train_en.txt \
    --ROOT ../tmp 
