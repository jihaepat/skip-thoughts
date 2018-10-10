#!/usr/bin/env bash

export LD_PRELOAD=/home/gulby/.pyenv/versions/anaconda3-5.2.0/lib/libmkl_def.so:/home/gulby/.pyenv/versions/anaconda3-5.2.0/lib/libmkl_avx.so:/home/gulby/.pyenv/versions/anaconda3-5.2.0/lib/libmkl_core.so:/home/gulby/.pyenv/versions/anaconda3-5.2.0/lib/libmkl_intel_lp64.so:/home/gulby/.pyenv/versions/anaconda3-5.2.0/lib/libmkl_intel_thread.so:/home/gulby/.pyenv/versions/anaconda3-5.2.0/lib/libiomp5.so
python build_index.py
