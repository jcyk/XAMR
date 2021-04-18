set -e

pip3 install -r requirements.txt

ckpt=/apdcephfs/share_916081/jcykcai/nonono/mbart50nmt/1/best-smatch_checkpoint_1_0.7617.pt
sh scripts/work.base.sh $ckpt facebook/mbart-large-50-many-to-many-mmt

ckpt=/apdcephfs/share_916081/jcykcai/nonono/mbart50/1/best-smatch_checkpoint_2_0.7516.pt
sh scripts/work.base.sh $ckpt facebook/mbart-large-50

exit 0

ckpt=/apdcephfs/share_916081/jcykcai/nonono/mbart50nmt/runs/0/best-smatch_checkpoint_4_0.8323.pt
sh scripts/work.mt.sh $ckpt facebook/mbart-large-50-many-to-many-mmt

ckpt=/apdcephfs/share_916081/jcykcai/nonono/mbart50/runs/0/best-smatch_checkpoint_7_0.8271.pt
sh scripts/work.mt.sh $ckpt facebook/mbart-large-50

ckpt=/apdcephfs/share_916081/jcykcai/nonono/bart/runs/0/best-smatch_checkpoint_5_0.8441.pt
sh scripts/work.mt.sh $ckpt facebook/bart-large

exit 0

ckpt=/apdcephfs/share_916081/jcykcai/nonono/mbart50/runs/1/best-smatch_checkpoint_1_0.7230.pt
sh work.base.sh $ckpt facebook/mbart-large-50

ckpt=/apdcephfs/share_916081/jcykcai/nonono/mbart50/runs/2/best-smatch_checkpoint_1_0.7010.pt
sh work.base.sh $ckpt facebook/mbart-large-50

ckpt=/apdcephfs/share_916081/jcykcai/nonono/mbart50nmt/runs/1/best-smatch_checkpoint_1_0.7386.pt
sh work.base.sh $ckpt facebook/mbart-large-50-many-to-many-mmt

ckpt=/apdcephfs/share_916081/jcykcai/nonono/mbart50nmt/runs/2/best-smatch_checkpoint_1_0.7145.pt
sh work.base.sh $ckpt facebook/mbart-large-50-many-to-many-mmt


#CUDA_VISIBLE_DEVICES=3,4,5,6 sh work.base.sh /apdcephfs/share_916081/jcykcai/nonono/bart/runs/0/best-smatch_checkpoint_5_0.8441.pt facebook/bart-large

#sh work.base.sh /apdcephfs/share_916081/jcykcai/nonono/mbart50/runs/0/best-smatch_checkpoint_7_0.8271.pt facebook/mbart-large-50 

#sh work.base.sh /apdcephfs/share_916081/jcykcai/nonono/mbart50nmt/runs/0/best-smatch_checkpoint_4_0.8323.pt facebook/mbart-large-50-many-to-many-mmt
