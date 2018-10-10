#!/usr/bin/env bash

dir=$1
if [ "$dir" == "" ]
then
    dir="/mnt/48TB/temp3"
fi

export LD_PRELOAD=/home/gulby/.pyenv/versions/anaconda3-5.2.0/lib/libmkl_def.so:/home/gulby/.pyenv/versions/anaconda3-5.2.0/lib/libmkl_avx.so:/home/gulby/.pyenv/versions/anaconda3-5.2.0/lib/libmkl_core.so:/home/gulby/.pyenv/versions/anaconda3-5.2.0/lib/libmkl_intel_lp64.so:/home/gulby/.pyenv/versions/anaconda3-5.2.0/lib/libmkl_intel_thread.so:/home/gulby/.pyenv/versions/anaconda3-5.2.0/lib/libiomp5.so
python build_index.py --path=$dir/encodings

cat result_study.txt | spm_decode --model=$dir/patent.txt.refine.sep.combine.skts.combine._train.model --input_format=id > result_study_decoded.txt
