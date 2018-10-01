import argparse
import math
import torch
import sys

from data_loader import DataLoader
from model import UniSkip
from config import *
from datetime import datetime

# 파라메터 세팅
parser = argparse.ArgumentParser()
parser.add_argument('--train_data', type=str, default='./data/test.id')
parser.add_argument('--init_model', type=str, default='')
parser.add_argument('--batch_size', type=int, default=400)
parser.add_argument('--total_epoch', type=int, default=10)
parser.add_argument('--save_model', type=int, default=1)
parser.add_argument('--total_line_count', type=int, default=0)
args = parser.parse_args()
train_data = args.train_data
init_model = args.init_model
batch_size = args.batch_size
total_epoch = args.total_epoch
save_model = args.save_model
total_line_count = args.total_line_count

# sentences 로딩
d = DataLoader(train_data, total_line_count)
sentences_count = len(d.sentences)
sys.stderr.write('total {} sentences\n'.format(sentences_count))

# 모델 초기화
sys.stderr.write('model init begin...\n')
mod = UniSkip()
if init_model:
    try:
        mod.load_state_dict(torch.load(init_model, map_location=lambda storage, loc: storage))
    except:
        sys.stderr.write('load init_model failed: {}\n'.format(init_model))
if USE_CUDA:
    mod.cuda(CUDA_DEVICE)
sys.stderr.write('model init end.\n')

# 디버깅용 변수 및 함수
loss_trail = []
last_best_loss = None
current_time = datetime.utcnow()


def debug(epoch, i, loss, prev, nex, prev_pred, next_pred):
    global loss_trail
    global last_best_loss
    global current_time

    this_loss = loss.item()
    loss_trail.append(this_loss)
    loss_trail = loss_trail[-20:]
    new_current_time = datetime.utcnow()
    time_elapsed = str(new_current_time - current_time)
    current_time = new_current_time
    sys.stderr.write("Epoch {} - Iteration {}: time = {} last_best_loss = {}, this_loss = {}\n".format(
              epoch, i, time_elapsed, last_best_loss, this_loss))

    print("{}\n{}\n\n{}\n{}\n".format(
        d.convert_var_to_sentences(prev),
        d.convert_var_to_sentences(prev_pred),
        d.convert_var_to_sentences(nex),
        d.convert_var_to_sentences(next_pred),
    ))

    try:
        trail_loss = sum(loss_trail)/len(loss_trail)
        if last_best_loss is None or last_best_loss > trail_loss:
            sys.stderr.write("Loss improved from {} to {}\n".format(last_best_loss, trail_loss))

            if save_model:
                save_loc = "./saved_models/skip-best".format(lr, VOCAB_SIZE)
                sys.stderr.write("saving model at {}\n".format(save_loc))
                torch.save(mod.state_dict(), save_loc)

            last_best_loss = trail_loss
    except Exception as e:
        sys.stderr.write("Couldn't save model because {}\n".format(e))


# train!!!
lr = 3.16e-4
optimizer = torch.optim.Adam(params=mod.parameters(), lr=lr)
iter_count_per_epoch = int(math.ceil(sentences_count/batch_size))
sys.stderr.write('iter_count_per_epoch : {}\n'.format(iter_count_per_epoch))

sys.stderr.write("training begin...\n")

for epoch in range(0, total_epoch):
    for i in range(0, iter_count_per_epoch):
        sentences, lengths = d.fetch_batch(batch_size)

        loss, prev, nex, prev_pred, next_pred = mod(sentences, lengths)

        if i % 10 == 0:
            debug(epoch, i, loss, prev, nex, prev_pred, next_pred)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # save after every epoch
    if save_model:
        save_loc_epoch = "./saved_models/skip-{}-epoch".format(epoch)
        sys.stderr.write("saving model at {}\n".format(save_loc_epoch))
        torch.save(mod.state_dict(), save_loc_epoch)

sys.stderr.write('training end.\n')
