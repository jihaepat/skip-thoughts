""""
Here we implement a class for loading data.
"""

import torch
from torch.autograd import Variable
from vocab import *
from config import *
import numpy as np
import random

np.random.seed(0)


class DataLoader:
    def __init__(self, text_file=None, sentences=None, word_dict=None):

        if text_file:
            print("Loading text file at {}".format(text_file))
            with open(text_file, "rt") as f:
                sentences = f.readlines()

        assert sentences

        self.sentences = sentences
        print("Making reverse dictionary")

        self.lengths = [len(sent) for sent in self.sentences]

    def convert_sentence_to_indices(self, sentence):
        # np.array
        indices = np.empty(MAXLEN, dtype=np.int)
        indices.fill(EOS)
        for i, w in enumerate(sentence.split()[:MAXLEN-1]):
            indices[i] = int(w)
        # torch.Variable
        indices = Variable(torch.from_numpy(indices))
        if USE_CUDA:
            indices = indices.cuda(CUDA_DEVICE)
        return indices

    def convert_indices_to_sentences(self, indices):
        return ' '.join([str(int(idx)) for idx in indices])

    def fetch_batch(self, batch_size):
        first_index = random.randint(0, len(self.sentences) - batch_size)
        batch = []
        lengths = []

        for i in range(first_index, first_index + batch_size):
            sent = self.sentences[i]
            ind = self.convert_sentence_to_indices(sent)
            batch.append(ind)
            lengths.append(min(len(sent.split()), MAXLEN))

        batch = torch.stack(batch)
        lengths = np.array(lengths)

        return batch, lengths
