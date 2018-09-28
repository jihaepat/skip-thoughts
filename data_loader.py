""""
Here we implement a class for loading data.
"""

import torch
from torch.autograd import Variable
from config import *
import numpy as np
import random

np.random.seed(0)


class DataLoader:
    def __init__(self, text_file, total_line_count=None):
        sentences = []
        lengths = []

        print("Loading text file at {}".format(text_file))
        with open(text_file, "rt") as f:
            for line in f:
                indices = np.empty(MAXLEN, dtype=np.int16)
                indices.fill(EOS)
                splits = line.split()
                for i, w in enumerate(splits[:MAXLEN-1]):
                    indices[i] = int(w)
                assert indices[-1] == EOS
                sentences.append(indices)
                # +1 을 해서 끝의 EOS 를 포함시킨다.
                lengths.append(min(len(splits) + 1, MAXLEN))

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
