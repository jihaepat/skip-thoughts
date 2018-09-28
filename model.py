"""
This file implements the Skip-Thought architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from config import *


class Encoder(nn.Module):
    @staticmethod
    def reverse_variable(var):
        idx = [i for i in range(var.size(0) - 1, -1, -1)]
        idx = Variable(torch.LongTensor(idx))

        if USE_CUDA:
            idx = idx.cuda(CUDA_DEVICE)

        inverted_var = var.index_select(0, idx)
        return inverted_var

    def __init__(self):
        super().__init__()
        self.word2embd = nn.Embedding(VOCAB_SIZE, WORD_SIZE)
        self.lstm = nn.LSTM(WORD_SIZE, THOUGHT_SIZE)

    def forward(self, sentences):
        # sentences = (batch_size, maxlen), with padding on the right.

        sentences = sentences.transpose(0, 1)  # (maxlen, batch_size)

        word_embeddings = F.tanh(self.word2embd(sentences))  # (maxlen, batch_size, WORD_SIZE)

        # The following is a hack: We read embeddings in reverse. This is required to move padding to the left.
        # If reversing is not done then the RNN sees a lot a garbage values right before its final state.
        # This reversing also means that the words will be read in reverse. But this is not a big problem since
        # several sequence to sequence models for Machine Translation do similar hacks.
        rev = self.reverse_variable(word_embeddings)

        _, (thoughts, _) = self.lstm(rev)
        thoughts = thoughts[-1]  # (batch, THOUGHT_SIZE)

        return thoughts, word_embeddings


class DuoDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.prev_lstm = nn.LSTM(THOUGHT_SIZE + WORD_SIZE, WORD_SIZE)
        self.next_lstm = nn.LSTM(THOUGHT_SIZE + WORD_SIZE, WORD_SIZE)
        self.worder = nn.Linear(WORD_SIZE, VOCAB_SIZE)

    def forward(self, thoughts, word_embeddings):
        # thoughts = (batch_size, THOUGHT_SIZE)
        # word_embeddings = # (maxlen, batch, WORD_SIZE)

        # We need to provide the current sentences's embedding or "thought" at every timestep.
        thoughts = thoughts.repeat(MAXLEN, 1, 1)  # (maxlen, batch, THOUGHT_SIZE)

        # Prepare Thought Vectors for Prev. and Next Decoders.
        prev_thoughts = thoughts[:, :-1, :]  # (maxlen, batch-1, THOUGHT_SIZE)
        next_thoughts = thoughts[:, 1:, :]   # (maxlen, batch-1, THOUGHT_SIZE)

        # Teacher Forcing.
        #   1.) Prepare Word embeddings for Prev and Next Decoders.
        prev_word_embeddings = word_embeddings[:, :-1, :]  # (maxlen, batch-1, WORD_SIZE)
        next_word_embeddings = word_embeddings[:, 1:, :]  # (maxlen, batch-1, WORD_SIZE)
        #   2.) delay the embeddings by one timestep
        delayed_prev_word_embeddings = torch.cat([0 * prev_word_embeddings[-1:, :, :], prev_word_embeddings[:-1, :, :]])
        delayed_next_word_embeddings = torch.cat([0 * next_word_embeddings[-1:, :, :], next_word_embeddings[:-1, :, :]])

        # Supply current "thought" and delayed word embeddings for teacher forcing.
        prev_pred_embds, _ = self.prev_lstm(torch.cat([next_thoughts, delayed_prev_word_embeddings], dim=2))  # (maxlen, batch-1, embd_size)
        next_pred_embds, _ = self.next_lstm(torch.cat([prev_thoughts, delayed_next_word_embeddings], dim=2))  # (maxlen, batch-1, embd_size)

        # predict actual words
        a, b, c = prev_pred_embds.size()
        prev_pred = self.worder(prev_pred_embds.view(a*b, c)).view(a, b, -1)  # (maxlen, batch-1, VOCAB_SIZE)
        a, b, c = next_pred_embds.size()
        next_pred = self.worder(next_pred_embds.view(a*b, c)).view(a, b, -1)  # (maxlen, batch-1, VOCAB_SIZE)

        prev_pred = prev_pred.transpose(0, 1).contiguous()  # (batch-1, maxlen, VOCAB_SIZE)
        next_pred = next_pred.transpose(0, 1).contiguous()  # (batch-1, maxlen, VOCAB_SIZE)

        return prev_pred, next_pred


class UniSkip(nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoders = DuoDecoder()

    def create_mask(self, var, lengths):
        mask = var.data.new().resize_as_(var.data).fill_(EOS)
#         print("lengths", lengths)
        for i, l in enumerate(lengths):
            for j in range(l):
                mask[i, j] = 1

        mask = Variable(mask)
        if USE_CUDA:
            mask = mask.cuda(var.get_device())

        return mask

    def forward(self, sentences, lengths):
        # sentences = (B, maxlen)
        # lengths = (B)

        # Compute Thought Vectors for each sentence. Also get the actual word embeddings for teacher forcing.
        thoughts, word_embeddings = self.encoder(sentences)  # thoughts = (B, THOUGHT_SIZE), word_embeddings = (B, maxlen, WORD_SIZE)

        # Predict the words for previous and next sentences.
        prev_pred, next_pred = self.decoders(thoughts, word_embeddings)  # both = (batch-1, maxlen, VOCAB_SIZE)

        # mask the predictions, so that loss for beyond-EOS word predictions is cancelled.
        prev_mask = self.create_mask(prev_pred, lengths[:-1])
        next_mask = self.create_mask(next_pred, lengths[1:])
        
        masked_prev_pred = prev_pred * prev_mask
        masked_next_pred = next_pred * next_mask
        
        prev_loss = F.cross_entropy(masked_prev_pred.view(-1, VOCAB_SIZE), sentences[:-1, :].view(-1))
        next_loss = F.cross_entropy(masked_next_pred.view(-1, VOCAB_SIZE), sentences[1:, :].view(-1))

        loss = prev_loss + next_loss
        
        _, prev_pred_ids = prev_pred[0].max(1)
        _, next_pred_ids = next_pred[0].max(1)

        return loss, sentences[0], sentences[1], prev_pred_ids, next_pred_ids













