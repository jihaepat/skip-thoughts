""""
Here we implement a class for loading data.
"""

import torch
from torch.autograd import Variable
from config import *
import numpy as np
import random
import os
import sys

np.random.seed(0)


class DataLoader:
    def __init__(self, text_file, total_line_count=None):
        try:
            sys.stderr.write('array load begin...\n')
            sentences = np.load('{}.sentences.npy'.format(text_file))
            total_line_count = len(sentences)
            sys.stderr.write('array load end.\n')
        except:
            if not total_line_count:
                sys.stderr.write('counting begin...\n')
                total_line_count = int(os.popen('wc -l {}'.format(text_file)).read().split()[0])
                sys.stderr.write('total_line_count: {}\n'.format(total_line_count))
                sys.stderr.write('counting end.\n')
            sys.stderr.write('array init begin...\n')
            sentences = np.empty((total_line_count, MAXLEN), dtype=np.int16)
            sentences.fill(EOS)
            sys.stderr.write('array init end.\n')

            sys.stderr.write("Loading text file at {}\n".format(text_file))
            with open(text_file, "rt") as f:
                for i, line in enumerate(f):
                    if i >= total_line_count:
                        break
                    splits = line.split()
                    for j, w in enumerate(splits[:MAXLEN-1]):
                        sentences[i][j] = int(w)
                    if i % 100000 == 0:
                        sys.stderr.write('{} sentences loaded.\n'.format(i))

            np.save('{}.sentences.npy'.format(text_file), sentences)

        self.sentences = sentences

    def convert_var_to_sentences(self, indices):
        return ' '.join([str(int(idx)) for idx in indices])

    def fetch_batch(self, batch_size):
        first_index = random.randint(0, len(self.sentences) - batch_size)
        batch = np.empty((batch_size, MAXLEN), dtype=np.int16)

        for i in range(first_index, first_index + batch_size):
            sentence = self.sentences[i]
            batch[i - first_index] = sentence

        batch = Variable(torch.from_numpy(batch.astype(dtype=np.int)))
        if USE_CUDA:
            batch = batch.cuda(CUDA_DEVICE)
        return batch
