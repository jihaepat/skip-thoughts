#!/usr/bin/env bash

dir=$1
if [ "$dir" == "" ]
then
    dir="/home/gulby/git/jtelips/temp"
fi
rm $dir/_splited/*
rm $dir/encodings/*

cat $dir/patent.txt.refine.sep.combine.skts.combine.id | python sampler.py > $dir/_splited/patent.txt.refine.sep.combine.skts.combine.id._train
working_dir=$(pwd)
cd $dir/_splited
split -d --lines=10000000 $dir/patent.txt.refine.sep.combine.skts.combine.id patent.txt.refine.sep.combine.skts.combine.id.
cd $working_dir

ls -1 $dir/_splited/* | parallel -j 3 python encoder.py --input_file={} --output_path=$dir/encodings --model=$dir/skip-best
rm $dir/_splited/*
