#!/usr/bin/env bash

ls -1 /home/gulby/temp/patent.txt.refine.sep.combine.skts.combine.id.0* | \
    parallel -j 10 python encoder.py \
        --input_file=/home/gulby/temp/patent.txt.refine.sep.combine.skts.combine.id.0{#} \
        --output_path=/mnt/48TB/temp3/encodings
