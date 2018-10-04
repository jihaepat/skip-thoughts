import faiss
import numpy as np
import pickle
import timeit
import random
import os

from glob2 import glob
from time import sleep


class VectorIndex(object):
    def __init__(self, dim, dir, train_file_count, index_type):
        self.dim = dim
        self.dir = dir
        self.train_file_count = train_file_count
        self.index_type = index_type
        self.index = None
        self.data = []

    @property
    def index_file(self):
        return os.path.join(self.dir, '{}.index'.format(self.index_type))

    def train_index(self):
        print('train index...')
        files = glob(os.path.join(self.dir, '*'))
        random.shuffle(files)
        train_data = []
        for file in files[:self.train_file_count]:
            with open(file, 'rb') as f:
                train_data.extend(pickle.load(f).values())
        nt = len(train_data)
        xt = np.empty((nt, self.dim), dtype=np.float32)
        for i in range(nt):
            xt[i] = train_data[i]['normalized']
        train_data = None; del train_data
        sleep(1.0)
        print('    train file load complete')

        index = faiss.index_factory(self.dim, self.index_type)
        print('    index created')
        index.train(xt)
        print('    index trained')
        assert index.is_trained
        faiss.write_index(index, '{}.train'.format(self.index_file))
        print('    index saved')
        self.index = index

    def init_index(self):
        print('init index...')
        try:
            with open('{}.data'.format(self.index_file), 'rb') as f:
                self.data = pickle.load(f)
            self.index = faiss.read_index(self.index_file)
        except:
            try:
                self.index = faiss.read_index('{}.train'.format(self.index_file))
                assert self.index.is_trained
                self.add_to_index()
            except:
                self.train_index()
                self.add_to_index()
        print('init index completed')

    def add_to_index(self):
        files = glob(os.path.join(self.dir, '*'))
        data = []
        for j, file in enumerate(files[:2]):
            with open(file, 'rb') as f:
                print('add data to index...: {}'.format(j))
                values = list(pickle.load(f).values())
                data.extend([value['line'] for value in values])
                vectors = np.empty((len(values), self.dim), np.float32)
                for i in range(len(values)):
                    vectors[i] = values[i]['normalized']
                self.index.add(vectors)
            values, vectors = None, None; del values; del vectors
            faiss.write_index(self.index, self.index_file)
            with open('{}.data'.format(self.index_file), 'wb') as f2:
                pickle.dump(data, f2)
        self.data = data

    def sample(self, n=10):
        files = glob(os.path.join(self.dir, '*'))
        random.shuffle(files)
        with open(files[0], 'rb') as f:
            values = list(pickle.load(f).values())
        vectors = np.empty((n, self.dim), dtype=np.float32)
        data = []
        for i, j in enumerate(np.random.randint(low=0, high=len(values), size=n)):
            vectors[i] = values[j]['normalized']
            data.append(values[j]['line'])
        values = None; del values
        return vectors, data

    def search(self, xq, k):
        return self.index.search(xq, k)


if __name__ == '__main__':
    # init
    index = VectorIndex(dim=1200, dir='/mnt/48TB/temp3/encodings', train_file_count=6, index_type='OPQ60,IMI2x14,PQ60')
    index.init_index()
    index.nprobe = 10

    # search
    xq, data = index.sample(n=10)
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
                print('{}: {}'.format(r[k], (1 - d[k] / 2)))
                f.write('{}\n'.format(index.data[r[k]].strip()))
            f.write('\n')
            print()

    # search time : 0.13s, mem 15.1%
    print('time: {}'.format(end - start))
