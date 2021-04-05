set -e
#en_XX, de_DE, zh_CN, es_XX, it_IT

lang_code=de_DE
blang=DE
slang=de
prefix=/apdcephfs/share_916081/jcykcai/nonono/amr_2-four_translations
out_dir=/apdcephfs/share_916081/jcykcai/nonono/mbart50nmt_translations
domains="bolt consensus dfa proxy xinhua"
for domain in $domains; do
    python3 translate.py \
        --src_lang_code ${lang_code} \
        --tgt_lang_code en_XX \
        --input_path ${prefix}/data/amr-release-2.0-amrs-test-${domain}.sentences.${blang}.txt \
        --output_path ${out_dir}/translation-test-${domain}.${blang}.txt
done
python3 translate.py \
    --src_lang_code en_XX \
    --tgt_lang_code ${lang_code} \
    --input_path ${out_dir}/dev.src.txt \
    --output_path ${out_dir}/dev_${slang}.txt

python3 translate.py \
    --src_lang_code en_XX \
    --tgt_lang_code ${lang_code} \
    --input_path ${out_dir}/train.src.txt \
    --output_path ${out_dir}/train_${slang}.txt
