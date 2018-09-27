import argparse
import math
import torch

from data_loader import DataLoader
from model import UniSkip
from config import *
from datetime import datetime

# 파라메터 세팅
parser = argparse.ArgumentParser()
parser.add_argument('--train_data', type=str, default='./data/test.id')
parser.add_argument('--batch_size', type=int, default=200)
parser.add_argument('--init_model', type=str, default='./saved_models/skip-best')
args = parser.parse_args()
train_data = args.train_data
batch_size = args.batch_size
init_model = args.init_model

# sentences 로딩
d = DataLoader(train_data)
sentences_count = len(d.sentences)
print('total {} sentences'.format(sentences_count))

# 모델 초기화
mod = UniSkip()
if init_model:
    try:
        mod.load_state_dict(torch.load(init_model, map_location=lambda storage, loc: storage))
    except:
        print('load init_model failed: {}'.format(init_model))
if USE_CUDA:
    mod.cuda(CUDA_DEVICE)

# 디버깅용 변수 및 함수
loss_trail = []
last_best_loss = None
current_time = datetime.utcnow()


def debug(i, loss, prev, nex, prev_pred, next_pred):
    global loss_trail
    global last_best_loss
    global current_time

    this_loss = loss.data[0]
    loss_trail.append(this_loss)
    loss_trail = loss_trail[-20:]
    new_current_time = datetime.utcnow()
    time_elapsed = str(new_current_time - current_time)
    current_time = new_current_time
    print("Iteration {}: time = {} last_best_loss = {}, this_loss = {}".format(
              i, time_elapsed, last_best_loss, this_loss))

    print("prev = {}\nnext = {}\npred_prev = {}\npred_next = {}".format(
        d.convert_indices_to_sentences(prev),
        d.convert_indices_to_sentences(nex),
        d.convert_indices_to_sentences(prev_pred),
        d.convert_indices_to_sentences(next_pred),
    ))

    try:
        trail_loss = sum(loss_trail)/len(loss_trail)
        if last_best_loss is None or last_best_loss > trail_loss:
            print("Loss improved from {} to {}".format(last_best_loss, trail_loss))

            save_loc = "./saved_models/skip-best".format(lr, VOCAB_SIZE)
            print("saving model at {}".format(save_loc))
            torch.save(mod.state_dict(), save_loc)

            last_best_loss = trail_loss
    except Exception as e:
       print("Couldn't save model because {}".format(e))


# train!!!
lr = 3e-4
optimizer = torch.optim.Adam(params=mod.parameters(), lr=lr)
iter_count_per_1epoch = int(math.ceil(sentences_count/batch_size))
print('iter_count_per_1epoch : {}'.format(iter_count_per_1epoch))

print("Starting training...")

for i in range(0, iter_count_per_1epoch):
    sentences, lengths = d.fetch_batch(batch_size)

    loss, prev, nex, prev_pred, next_pred  = mod(sentences, lengths)

    if i % 10 == 0:
        debug(i, loss, prev, nex, prev_pred, next_pred)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print('End training.')
