export http_proxy="star-proxy.oa.com:3128"
export https_proxy="star-proxy.oa.com:3128"

pip3 install -r requirements.txt
PYTHONPATH=. python3 bin/train.py \
--config configs/config.mbart50nmt.xl.opus.yaml \
--teacher_checkpoint /apdcephfs/share_916081/jcykcai/nonono/old/mbart50nmt/mt/exclude_zh/ft/1/best-smatch_checkpoint_1_0.7735.pt \
--train "/apdcephfs/share_916081/jcykcai/nonono/europarl/noisy_training/*.txt" \
--kd \
--kd_alpha 0.5 \
--kd_temperature 1 \
--kd_start_after 0 \
--ROOT /apdcephfs/share_916081/jcykcai/nonono/mbart50nmt/selfwordseq/pt

PYTHONPATH=. python3 bin/train.py \
--config configs/config.mbart50nmt.xl.opus.yaml \
--checkpoint /apdcephfs/share_916081/jcykcai/nonono/mbart50nmt/selfwordseq/pt/1/*.pt \
--ROOT /apdcephfs/share_916081/jcykcai/nonono/mbart50nmt/selfwordseq/ft
