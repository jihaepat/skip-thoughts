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
parser.add_argument('--batch_size', type=int, default=7000)
parser.add_argument('--input_file', type=str, default='')
parser.add_argument('--output_path', type=str, default='./encodings')
args = parser.parse_args()
model = args.model
batch_size = args.batch_size
input_file = args.input_file
output_path = args.output_path
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
    batch = Variable(torch.from_numpy(sentences))   ;assert batch.shape == (batch_size, MAXLEN)
    if USE_CUDA:
        batch = batch.cuda(CUDA_DEVICE)
    encodings, _ = encoder(batch)                   ;assert encodings.shape == (batch_size, THOUGHT_SIZE)
    norms = encodings.norm(dim=1)                   ;assert norms.shape == (batch_size,)
    norms_for_normalize = norms.clamp(1e-12).expand(THOUGHT_SIZE, batch_size).transpose(0, 1)
    normalized_encodings = encodings / norms_for_normalize      ;assert normalized_encodings.shape == (batch_size, THOUGHT_SIZE)
    normalized_encodings = normalized_encodings.data.cpu().numpy()
    norms = norms.data.cpu().numpy()
    # save
    for line, normalized, norm in zip(lines, normalized_encodings, norms):
        # assert 0.99 <= np.sqrt(np.dot(normalized, normalized)) <= 1.01
        assert normalized.shape == (THOUGHT_SIZE,)
        h = CityHash128(line)
        ht = '000000{}'.format(h)
        path = '{}/{}/{}'.format(output_path, ht[-3:], ht[-6:-3])
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
        np.save('{}/{}_{}.npy'.format(path, h, norm), normalized)
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
