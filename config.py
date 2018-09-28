"""
Configuration file.
"""

VOCAB_SIZE = 16000
USE_CUDA = True
DEVICES = [0]
CUDA_DEVICE = DEVICES[0]
VERSION = 1
MAXLEN = 30
THOUGHT_SIZE = 1200
WORD_SIZE = 620
UNK = 0     # to mean unknown token
BOS = 1     # to mean begin of sentence
EOS = 2     # to mean end of sentence
