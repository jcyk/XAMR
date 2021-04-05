set -e

pip3 install -r requirements.txt

CUDA_VISIBLE_DEVICES=3,4,5,6 sh work.zeroshot.sh /apdcephfs/share_916081/jcykcai/nonono/bart/runs/0/best-smatch_checkpoint_5_0.8441.pt facebook/bart-large

#sh work.base.sh /apdcephfs/share_916081/jcykcai/nonono/mbart50/runs/0/best-smatch_checkpoint_7_0.8271.pt facebook/mbart-large-50 

#sh work.base.sh /apdcephfs/share_916081/jcykcai/nonono/mbart50nmt/runs/0/best-smatch_checkpoint_4_0.8323.pt facebook/mbart-large-50-many-to-many-mmt
