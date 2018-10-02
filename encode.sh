#!/usr/bin/env bash

ls -1 /mnt/48TB/temp3/patent.txt.refine.sep.combine.skts.combine.id.0* | \
    parallel -j 5 python encoder.py \
        --input_file=/mnt/48TB/temp3/patent.txt.refine.sep.combine.skts.combine.id.0{#} \
        --output_path=/mnt/48TB/temp3/encodings
