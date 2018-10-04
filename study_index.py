import faiss
import numpy as np
import pickle
import timeit

from glob2 import glob


# 준비
files = glob('/mnt/48TB/temp3/encodings/*')
results = []
for file, _ in zip(files, range(2)):
    with open(file, 'rb') as f:
        results.extend(pickle.load(f).values())
d = 1200
nb = len(results)
print(nb)
nq = 100

# xb 초기화 : 전체 data
xb = np.empty((nb, d), dtype=np.float32)
for i in range(nb):
    xb[i] = results[i]['normalized']

# xq 초기화 : query
xq = np.empty((nq, d), dtype=np.float32)
iq = np.random.randint(0, nb, nq)
print(iq)
for j, i in enumerate(iq):
    xq[j] = xb[i]

# index
quantizer = faiss.IndexFlatL2(d)
index = faiss.IndexIVFFlat(quantizer, d, 2000, faiss.METRIC_L2)
assert not index.is_trained
index.train(xb)
assert index.is_trained
index.add(xb)
index.nprobe = 10

# search
K = 10
start = timeit.default_timer()
D, I = index.search(xq, K)
end = timeit.default_timer()
print(I)

# dump info
print()
with open('result_study.txt', 'w') as f:
    for r, d in zip(I, D):
        for k in range(K):
            print('{}: {}'.format(r[k], (1 - d[k]/2)))
            f.write('{}\n'.format(results[r[k]]['line'].strip()))
        f.write('\n')
        print()

# search time : 0.13s, mem 15.1%
print('time: {}'.format(end - start))
