import faiss
import numpy as np
import pickle
import timeit
import os
import argparse

from glob2 import glob
from time import sleep


class VectorIndex(object):
    def __init__(self, dim, dir, index_type):
        self.dim = dim
        self.dir = dir
        self.index_type = index_type
        self.index = None
        self.data = []

    @property
    def index_file(self):
        return os.path.join(self.dir, '{}.index'.format(self.index_type))

    def train_index(self):
        print('train index...')
        files = glob(os.path.join(self.dir, '*._train*.pkl'))
        train_data = []
        for file in files:
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
            except:
                self.train_index()
            self.add_to_index()
            with open('{}.data'.format(self.index_file), 'rb') as f:
                self.data = pickle.load(f)
        print('init index completed')

    def add_to_index(self):
        files = glob(os.path.join(self.dir, '*.??_*.pkl'))
        # files = glob(os.path.join(self.dir, '*._train*.pkl'))
        for j, file in enumerate(files):
            print('add data to index...: {}'.format(j))
            with open(file, 'rb') as f:
                values = list(pickle.load(f).values())
            data = [value['line'] for value in values]
            vectors = np.empty((len(values), self.dim), np.float32)
            for i in range(len(values)):
                vectors[i] = values[i]['normalized']
            values = None; del values
            sleep(1.0)
            self.index.add(vectors)
            vectors = None; del vectors
            sleep(1.0)
            faiss.write_index(self.index, self.index_file)
            total_data = []
            try:
                with open('{}.data'.format(self.index_file), 'rb') as f2:
                    total_data = pickle.load(f2)
            except:
                pass
            total_data.extend(data)
            with open('{}.data'.format(self.index_file), 'wb') as f3:
                pickle.dump(total_data, f3)
            data, total_data = None, None; del data; del total_data
            sleep(1.0)

    def sample(self, n):
        # files = glob(os.path.join(self.dir, '*.??_*.pkl'))
        files = glob(os.path.join(self.dir, '*._train*.pkl'))
        # import random
        # random.shuffle(files)
        with open(files[0], 'rb') as f:
            values = list(pickle.load(f).values())
        vectors = np.empty((n, self.dim), dtype=np.float32)
        data = []
        # for i, j in enumerate(np.random.randint(low=0, high=len(values), size=n)):
        #     vectors[i] = values[j]['normalized']
        #     data.append(values[j]['line'])
        for i in range(n):
            vectors[i] = values[i]['normalized']
            data.append(values[i]['line'])
        values = None; del values
        return vectors, data

    def search(self, xq, k):
        return self.index.search(xq, k)

    def set_nprobe(self, nprobe):
        self.index.nprobe = nprobe


if __name__ == '__main__':
    # path
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='/home/gulby/git/jtelips/temp')
    args = parser.parse_args()
    path = args.path

    # init
    index = VectorIndex(dim=1200, dir=path, index_type='IMI2x14,PQ100')
    index.init_index()
    index.set_nprobe(1024)

    # search
    xq, data = index.sample(n=100)
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
    print('time: {}'.format(end - start))
