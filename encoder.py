import argparse
import sys
import torch
import numpy as np
import pickle

from torch.autograd import Variable
from model import UniSkip
from config import *
from datetime import datetime
from cityhash import CityHash128

# 파라메터 세팅
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='./saved_models/skip-best')
parser.add_argument('--batch_size', type=int, default=1000)
parser.add_argument('--input_file', type=str, default='./data/test.id')
parser.add_argument('--output_path', type=str, default='./encodings')
parser.add_argument('--batch_count_per_output_file', type=int, default=1000)
args = parser.parse_args()
model = args.model
batch_size = args.batch_size
input_file = args.input_file
output_path = args.output_path
batch_count_per_output_file = args.batch_count_per_output_file
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
results = {}
processed_batch_count = 0
processed_line_count = 0
duplicate_count = 0

# time variables
current_time = datetime.utcnow()
new_current_time = datetime.utcnow()
time_elapsed = str(new_current_time - current_time)
current_time = new_current_time


def process_batch():
    global lines, sentences, results, processed_batch_count, processed_line_count, duplicate_count
    # numpy array 초기화
    for i, line in enumerate(lines):
        splits = line.split()
        for j, w in enumerate(splits[:MAXLEN - 1]):
            sentences[i][j] = int(w)
        processed_line_count += 1
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
        # if h in results:
        #     assert line == results[h]['line']
        #     duplicate_count += 1
        results[h] = {'norm': norm, 'normalized': normalized}
    # finalize
    sentences.fill(EOS)
    lines = []
    processed_batch_count += 1


def dump_output():
    global results
    with open('{}/{}_{}.pkl'.format(output_path, input_file.split('/')[-1], processed_batch_count), 'wb') as f:
        pickle.dump(results, f)
    results = {}
    print(processed_line_count)
    print(duplicate_count)


print("encoding begin...")

for input_line in f:
    lines.append(input_line)
    if len(lines) < batch_size:
        continue
    process_batch()
    if processed_batch_count % batch_count_per_output_file == 0:
        dump_output()
else:
    if lines:
        process_batch()
        dump_output()
if f != sys.stdin:
    f.close()

print('encoding end.')
