export http_proxy="star-proxy.oa.com:3128"
export https_proxy="star-proxy.oa.com:3128"

pip3 install -r requirements.txt
PYTHONPATH=. python3 bin/train.py \
--config configs/config.mbart50nmt.xl.opus.yaml \
--teacher_checkpoint /apdcephfs/share_916081/jcykcai/nonono/mbart50nmt/runs/0/best-smatch_checkpoint_4_0.8323.pt \
--kd \
--kd_alpha 0.5 \
--kd_temperature 1 \
--kd_start_after 0 \
--ROOT /apdcephfs/share_916081/jcykcai/nonono/mbart50nmt
