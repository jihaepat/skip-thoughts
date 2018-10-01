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
            lengths = np.load('{}.lengths.npy'.format(text_file))
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
            lengths = np.empty(total_line_count, dtype=np.int16)
            lengths.fill(0)
            sys.stderr.write('array init end.\n')

            sys.stderr.write("Loading text file at {}\n".format(text_file))
            with open(text_file, "rt") as f:
                for i, line in enumerate(f):
                    if i >= total_line_count:
                        break
                    splits = line.split()
                    for j, w in enumerate(splits[:MAXLEN-1]):
                        sentences[i][j] = int(w)
                    # +1 을 해서 끝의 EOS 를 포함시킨다.
                    lengths[i] = min(len(splits) + 1, MAXLEN)
                    if i % 100000 == 0:
                        sys.stderr.write('{} sentences loaded.\n'.format(i))

            np.save('{}.sentences.npy'.format(text_file), sentences)
            np.save('{}.lengths.npy'.format(text_file), lengths)

        self.sentences = sentences
        self.lengths = lengths

    def convert_var_to_sentences(self, indices):
        return ' '.join([str(int(idx)) for idx in indices])

    def fetch_batch(self, batch_size):
        first_index = random.randint(0, len(self.sentences) - batch_size)
        batch = []
        lengths = []

        for i in range(first_index, first_index + batch_size):
            sentence = self.sentences[i]
            sentence = Variable(torch.from_numpy(sentence.astype(dtype=np.int)))
            if USE_CUDA:
                sentence = sentence.cuda(CUDA_DEVICE)
            batch.append(sentence)
            lengths.append(self.lengths[i])

        batch = torch.stack(batch)
        lengths = np.array(lengths)

        return batch, lengths
