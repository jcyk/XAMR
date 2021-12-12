
for lang in en zh es it de; do
PYTHONPATH=. python3 bin/train.py \
--config configs/config.mbart50nmt.xl.opus.yaml \
--train xl-amr/data/AMR/amr_2.0/kd/kd_${lang}.txt \
--nproc_per_node 1 \
--make_cache xl-amr/data/AMR/amr_2.0/kd/kd_${lang}.pt
done

