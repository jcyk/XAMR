set -e

pip3 install -r requirements.txt


ckpt=/apdcephfs/share_916081/jcykcai/nonono/mbart50nmt/para/include_zh/ft/7/*.pt
sh scripts/work.base.sh $ckpt facebook/mbart-large-50-many-to-many-mmt

exit 0
ckpt=/apdcephfs/share_916081/jcykcai/nonono/mbart50nmt/runs/opus/1/*.pt
sh scripts/work.base.sh $ckpt facebook/mbart-large-50-many-to-many-mmt


exit 0


ckpt=/apdcephfs/share_916081/jcykcai/nonono/mbart50nmt/self/include_zh/ft/0/*.pt
sh scripts/work.base.sh $ckpt facebook/mbart-large-50-many-to-many-mmt

ckpt=/apdcephfs/share_916081/jcykcai/nonono/mbart50nmt/para/include_zh/pt/1/*.pt
sh scripts/work.base.sh $ckpt facebook/mbart-large-50-many-to-many-mmt

ckpt=/apdcephfs/share_916081/jcykcai/nonono/mbart50nmt/para/include_zh/pt/5/*.pt
sh scripts/work.base.sh $ckpt facebook/mbart-large-50-many-to-many-mmt

ckpt=/apdcephfs/share_916081/jcykcai/nonono/mbart50nmt/para/include_zh/pt/4/*.pt
sh scripts/work.base.sh $ckpt facebook/mbart-large-50-many-to-many-mmt

ckpt=/apdcephfs/share_916081/jcykcai/nonono/mbart50nmt/para/include_zh/pt/6/*.pt
sh scripts/work.base.sh $ckpt facebook/mbart-large-50-many-to-many-mmt



exit 0

ckpt=/apdcephfs/share_916081/jcykcai/nonono/mbart50nmt/runs/opus/best-smatch_checkpoint_1_0.7412.pt
sh scripts/work.base.sh $ckpt facebook/mbart-large-50-many-to-many-mmt

ckpt=/apdcephfs/share_916081/jcykcai/nonono/mbart50/runs/opus/best-smatch_checkpoint_1_0.7294.pt
sh scripts/work.base.sh $ckpt facebook/mbart-large-50
exit 0

ckpt=/apdcephfs/share_916081/jcykcai/nonono/mbart50nmt/runs/en_only/best-smatch_checkpoint_4_0.8323.pt
sh scripts/work.mt.sh $ckpt facebook/mbart-large-50-many-to-many-mmt

ckpt=/apdcephfs/share_916081/jcykcai/nonono/mbart50/runs/en_only/best-smatch_checkpoint_7_0.8271.pt
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


#sh work.base.sh /apdcephfs/share_916081/jcykcai/nonono/bart/runs/0/best-smatch_checkpoint_5_0.8441.pt facebook/bart-large

#sh work.base.sh /apdcephfs/share_916081/jcykcai/nonono/mbart50/runs/0/best-smatch_checkpoint_7_0.8271.pt facebook/mbart-large-50 

#sh work.base.sh /apdcephfs/share_916081/jcykcai/nonono/mbart50nmt/runs/0/best-smatch_checkpoint_4_0.8323.pt facebook/mbart-large-50-many-to-many-mmt
