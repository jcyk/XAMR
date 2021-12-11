#!/usr/bin/env bash

set -e

dir=data/AMR/amr_2.0
translations=data/AMR/amr_2.0/translations
out_path=${dir}/proj

python -u -m xlamr_stog.data.dataset_readers.amr_projection.project_train_dev \
        --amr_path ${dir} --trans_path ${translations} --out_path ${out_path}/ \


