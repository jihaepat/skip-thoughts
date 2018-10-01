import argparse
import sys
import torch
import numpy as np
import pathlib

from torch.autograd import Variable
from model import UniSkip
from config import *
from datetime import datetime
from cityhash import CityHash128

# 파라메터 세팅
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='./saved_models/skip-best')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--input_file', type=str, default='')
args = parser.parse_args()
model = args.model
batch_size = args.batch_size
input_file = args.input_file
f = None
if input_file:
    f = open(input_file)
else:
    f = sys.stdin

# 모델 초기화
print('model init begin...')
mod = UniSkip()
mod.load_state_dict(torch.load(model, map_location=lambda storage, loc: storage))
if USE_CUDA:
    mod.cuda(CUDA_DEVICE)
print('model init end.')

# core variables
lines = []
encoder = mod.encoder
sentences = np.empty((batch_size, MAXLEN), dtype=np.int)
sentences.fill(EOS)

# time variables
current_time = datetime.utcnow()
new_current_time = datetime.utcnow()
time_elapsed = str(new_current_time - current_time)
current_time = new_current_time


def process_batch():
    global lines, encoder, sentences
    # numpy array 초기화
    for i, line in enumerate(lines):
        splits = line.split()
        for j, w in enumerate(splits[:MAXLEN - 1]):
            sentences[i][j] = int(w)
    # numpy --> torch
    batch = Variable(torch.from_numpy(sentences))
    if USE_CUDA:
        batch = batch.cuda(CUDA_DEVICE)
    encodings, _ = encoder(batch)
    encodings = encodings.view(-1, THOUGHT_SIZE)
    encodings = encodings.data.cpu().numpy()
    # save
    for line, encoding in zip(lines, encodings):
        h = CityHash128(line)
        ht = '000000{}'.format(h)
        path = './encodings/{}/{}'.format(ht[-3:], ht[-6:-3])
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
        assert encoding.shape == (THOUGHT_SIZE,)
        np.save('{}/{}.npy'.format(path, h), encoding)
    # finalize
    sentences.fill(EOS)
    lines = []


print("encoding begin...")

for input_line in f:
    # batch_size 만큼 쌓기
    if len(lines) < batch_size:
        lines.append(input_line)
        continue
    process_batch()
else:
    if lines:
        process_batch()
if f != sys.stdin:
    f.close()

print('encoding end.')
