# coding=utf-8

"""
A very basic implementation of neural machine translation

Usage:
    nmt.py train --train-src=<file> --train-tgt=<file> --dev-src=<file> --dev-tgt=<file> --vocab=<file> [options]
    nmt.py decode [options] MODEL_PATH TEST_SOURCE_FILE OUTPUT_FILE
    nmt.py decode [options] MODEL_PATH TEST_SOURCE_FILE TEST_TARGET_FILE OUTPUT_FILE

Options:
    -h --help                               show this screen.
    --cuda                                  use GPU
    --train-src=<file>                      train source file
    --train-tgt=<file>                      train target file
    --dev-src=<file>                        dev source file
    --dev-tgt=<file>                        dev target file
    --vocab=<file>                          vocab file
    --seed=<int>                             seed [default: 0]
    --batch-size=<int>                      batch size [default: 32]
    --embed-size=<int>                      embedding size [default: 256]
    --hidden-size=<int>                     hidden size [default: 256]
    --clip-grad=<float>                     gradient clipping [default: 5.0]
    --log-every=<int>                       log every [default: 10]
    --max-epoch=<int>                       max epoch [default: 30]
    --patience=<int>                        wait for how many iterations to decay learning rate [default: 5]
    --max-num-trial=<int>                   terminate training after how many trials [default: 5]
    --lr-decay=<float>                      learning rate decay [default: 0.5]
    --beam-size=<int>                       beam size [default: 5]
    --lr=<float>                            learning rate [default: 0.001]
    --uniform-init=<float>                  uniformly initialize all parameters [default: 0.1]
    --save-to=<file>                        model save path
    --valid-niter=<int>                     perform validation after how many iterations [default: 500]
    --dropout=<float>                       dropout [default: 0.2]
    --max-decoding-time-step=<int>          maximum number of decoding time steps [default: 70]
"""

import math
import pickle
import sys
import time
import random
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
import torch.nn.functional as F
from torch.nn.functional import log_softmax

import numpy as np
from typing import List, Tuple, Dict, Set, Union
from docopt import docopt
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction

from utils import read_corpus, batch_iter, input_transpose
from vocab import Vocab, VocabEntry



Hypothesis = namedtuple('Hypothesis', ['value', 'score'])

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class NMT(nn.Module):

    def __init__(self, embed_size, hidden_size, vocab, dropout_rate=0.2):
        super(NMT, self).__init__()

        embed_size = embed_size
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.vocab = vocab
        self.teach_rate = 0.95

        # initialize neural network layers...
        self.src_embedding = nn.Embedding(len(self.vocab.src), embed_size)
        self.tgt_embedding = nn.Embedding(len(self.vocab.tgt), embed_size)

        self.encode_lstm = nn.LSTM(embed_size, hidden_size, bidirectional = True)
        self.decode_lstm = nn.LSTMCell(embed_size + hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size * 2, hidden_size)
        self.value = nn.Linear(hidden_size * 2, hidden_size)

        self.init = nn.Linear(hidden_size * 2, hidden_size)

        self.dropout = nn.Dropout(dropout_rate)
        self.output = nn.Linear(2 * hidden_size, len(self.vocab.tgt))

        self.criterion = nn.CrossEntropyLoss(ignore_index = 0, reduction = 'sum')


    def forward(self, src_sents, tgt_sents, is_training = True):
        """
        take a mini-batch of source and target sentences, compute the log-likelihood of
        target sentences.

        Args:
            src_sents: list of source sentence tokens
            tgt_sents: list of target sentence tokens, wrapped by `<s>` and `</s>`

        Returns:
            scores: a variable/tensor of shape (batch_size, ) representing the
                log-likelihood of generating the gold-standard target sentence for
                each example in the input batch
        """
        src_encodings, src_lengths, decoder_init_state = self.encode(src_sents)
        scores, tgt_padded = self.decode(src_encodings, src_lengths, decoder_init_state, tgt_sents) # (L, B, V)
        #print(scores.shape)
        scores = scores.permute(1, 2, 0)
        loss = self.criterion(scores, tgt_padded[:, 1:])
        return loss

    def make_mask(self, lens):
        lens = torch.tensor(lens)
        tmp_lens = lens.unsqueeze(1)
        max_lens = lens[0]
        size = lens.shape
        t = torch.arange(0, max_lens)
        mask = t < tmp_lens
        return mask.float().to('cuda')

    def encode(self, src_sents):
        """
        Use a GRU/LSTM to encode source sentences into hidden states

        Args:
            src_sents: list of source sentence tokens

        Returns:
            src_encodings: hidden states of tokens in source sentences, this could be a variable
                with shape (batch_size, source_sentence_length, encoding_dim), or in orther formats
            decoder_init_state: decoder GRU/LSTM's initial state, computed from source encodings
        """

        src_padded, src_lens = self.pre_parse(src_sents, self.vocab.src)

        x = self.src_embedding(src_padded)
        x = rnn.pack_padded_sequence(x, torch.tensor(src_lens))
        x, (h, c) = self.encode_lstm(x)
        src_encodings, _ = rnn.pad_packed_sequence(x) # (L, B, H)
        decoder_init_cell = self.init(torch.cat([c[0], c[1]], dim=1))
        decoder_init_hidden = F.tanh(decoder_init_cell)

        return src_encodings, src_lens, (decoder_init_hidden, decoder_init_cell)

    def decode(self, src_encodings, src_lengths, decoder_init_state, tgt_sents, is_training = True):
        """
        Given source encodings, compute the log-likelihood of predicting the gold-standard target
        sentence tokens

        Args:
            src_encodings: hidden states of tokens in source sentences (L, B, 2 *H)
            decoder_init_state: decoder GRU/LSTM's initial state (B, H)
            tgt_sents: list of gold-standard target sentences, wrapped by `<s>` and `</s>`(B, )

        Returns:
            scores: could be a variable of shape (batch_size, ) representing the
                log-likelihood of generating the gold-standard target sentence for
                each example in the input batch
        """
        hidden_states = decoder_init_state
        tgt_padded, tgt_lens = self.pre_parse(tgt_sents, self.vocab.tgt)
        tgt_padded = tgt_padded.transpose(1, 0)
        B, L = tgt_padded.shape

        keys, values = self.key(src_encodings), self.value(src_encodings)
        keys = keys.transpose(0, 1) # (B, L, H)
        emb = self.tgt_embedding(tgt_padded)
        context = torch.zeros((B, self.hidden_size)).to(DEVICE)
        
        scores = []
        mask = self.make_mask(src_lengths)

        for i in range(L - 1):
            use_teacher_forcing = True if is_training and random.random() < self.teach_rate else False
            dec_inp = emb[:, i, :] if use_teacher_forcing or not scores else self.tgt_embedding(gen)
            x = torch.cat((dec_inp, context), dim = 1)
            h, c = self.decode_lstm(x, (hidden_states))
            query = h.unsqueeze(2) # (B, H, 1)
            context, att = self.dot_attention(keys, query, values, mask)
            score = self.output(att) #B, V
            hidden_states = h, c
            scores.append(score)
            gen = torch.argmax(score, dim=1)

        return torch.stack(scores), tgt_padded

    def dot_attention(self, keys, query, values, mask):
        energy = torch.bmm(keys,query).squeeze(2) * mask #B,L
        attention = F.softmax(energy, dim = 1)
        attention = F.normalize(attention, p = 1, dim = 1)
        context = torch.bmm(attention.unsqueeze(1), values.permute(1, 0, 2))
        context = context.squeeze(1)
        att = torch.cat((context, query.squeeze(2)), dim = 1)  #B, H
        return context, att
    
    def pre_parse(self, sents, vocab):
        data = vocab.words2indices(sents)
        lens = [len(s) for s in sents]
        data = [torch.tensor(s) for s in data]
        padded = rnn.pad_sequence(data).to(DEVICE)
        return padded, lens
    
    def beam_search(self, src_sent, beam_size=5, max_decoding_time_step=70):
        """
        Given a single source sentence, perform beam search

        Args:
            src_sent: a single tokenized source sentence
            beam_size: beam size
            max_decoding_time_step: maximum number of time steps to unroll the decoding RNN

        Returns:
            hypotheses: a list of hypothesis, each hypothesis has two fields:
                value: List[str]: the decoded target sentence, represented as a list of words
                score: float: the log-likelihood of the target sentence
        """
        pass
    
    def greedy_search(self, src_sent, max_decoding_time_step = 70):
        src_sent = src_sent
        src_encodings, src_lengths, hidden_states = self.encode([src_sent])

        keys, values = self.key(src_encodings), self.value(src_encodings)
        keys = keys.transpose(0, 1) # (B, L, H)
        context = torch.zeros((1, self.hidden_size)).to(DEVICE)
        
        gen = torch.tensor([1]).to(DEVICE)
        mask = self.make_mask(src_lengths)
        generations = []

        for _ in range(max_decoding_time_step):
            dec_inp = self.tgt_embedding(gen)
            x = torch.cat((dec_inp, context), dim = 1)
            h, c = self.decode_lstm(x, hidden_states)
            query = h.unsqueeze(2) # (B, H, 1)
            context, att = self.dot_attention(keys, query, values, mask)
            out = self.output(att)
            gen = torch.argmax(out, dim = 1)
            hidden_states = h, c
            if gen.item() == 2:
                break
            generations.append(self.vocab.tgt.id2word[gen.item()])
    
        return generations

    def evaluate_ppl(self, dev_data, batch_size=32):
        """
        Evaluate perplexity on dev sentences

        Args:
            dev_data: a list of dev sentences
            batch_size: batch size

        Returns:
            ppl: the perplexity on dev sentences
        """

        cum_loss = 0.
        cum_tgt_words = 0.

        # you may want to wrap the following code using a context manager provided
        # by the NN library to signal the backend to not to keep gradient information
        # e.g., `torch.no_grad()`
        with torch.no_grad():

            for src_sents, tgt_sents in batch_iter(dev_data, batch_size):
                loss = self.forward(src_sents, tgt_sents, is_training = False)

                cum_loss += loss.item()
                tgt_word_num_to_predict = sum(len(s[1:]) for s in tgt_sents)  # omitting the leading `<s>`
                cum_tgt_words += tgt_word_num_to_predict

            ppl = np.exp(cum_loss / cum_tgt_words)

        return ppl


def compute_corpus_level_bleu_score(references, hypotheses):
    """
    Given decoding results and reference sentences, compute corpus-level BLEU score

    Args:
        references: a list of gold-standard reference target sentences
        hypotheses: a list of hypotheses, one for each reference

    Returns:
        bleu_score: corpus-level BLEU score
    """
    if references[0][0] == '<s>':
        references = [ref[1:-1] for ref in references]
    bleu_score = corpus_bleu([[ref] for ref in references],
                             [hyp for hyp in hypotheses])

    return bleu_score


def train(args):
    train_data_src = read_corpus(args['--train-src'], source='src')
    train_data_tgt = read_corpus(args['--train-tgt'], source='tgt')

    dev_data_src = read_corpus(args['--dev-src'], source='src')
    dev_data_tgt = read_corpus(args['--dev-tgt'], source='tgt')

    train_data = list(zip(train_data_src, train_data_tgt))
    dev_data = list(zip(dev_data_src, dev_data_tgt))

    train_batch_size = int(args['--batch-size'])
    clip_grad = float(args['--clip-grad'])
    valid_niter = int(args['--valid-niter'])
    log_every = int(args['--log-every'])
    model_save_path = args['--save-to']
    lr = float(args['--lr'])
    lr_decay = float(args['--lr-decay'])

    vocab = pickle.load(open(args['--vocab'], 'rb'))

    model = NMT(embed_size=int(args['--embed-size']),
                hidden_size=int(args['--hidden-size']),
                dropout_rate=float(args['--dropout']),
                vocab=vocab)

    model = model.to(DEVICE)
    for p in model.parameters():
        p.data.uniform_(-0.1, 0.1)

    optimizer = torch.optim.Adam(model.parameters(), lr = lr)

    num_trial = 0
    train_iter = patience = cum_loss = report_loss = cumulative_tgt_words = report_tgt_words = 0
    cumulative_examples = report_examples = epoch = valid_num = 0
    hist_valid_scores = []
    train_time = begin_time = time.time()
    print('begin Maximum Likelihood training')

    while True:
        epoch += 1

        for src_sents, tgt_sents in batch_iter(train_data, batch_size=train_batch_size, shuffle=True):
            train_iter += 1
            optimizer.zero_grad()

            batch_size = len(src_sents)

            # (batch_size)
            loss = model(src_sents, tgt_sents)

            report_loss += loss.item()
            cum_loss += loss.item()

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm(model.parameters(), 0.8)
            optimizer.step()

            tgt_words_num_to_predict = sum(len(s[1:]) for s in tgt_sents)  # omitting leading `<s>`
            report_tgt_words += tgt_words_num_to_predict
            cumulative_tgt_words += tgt_words_num_to_predict
            report_examples += batch_size
            cumulative_examples += batch_size

            if train_iter % log_every == 0:
                print('epoch %d, iter %d, avg. loss %.2f, avg. ppl %.2f ' \
                      'cum. examples %d, speed %.2f words/sec, time elapsed %.2f sec' % (epoch, train_iter,
                                                                                         report_loss / report_examples,
                                                                                         math.exp(report_loss / report_tgt_words),
                                                                                         cumulative_examples,
                                                                                         report_tgt_words / (time.time() - train_time),
                                                                                         time.time() - begin_time), file=sys.stderr)

                train_time = time.time()
                report_loss = report_tgt_words = report_examples = 0.

            # the following code performs validation on dev set, and controls the learning schedule
            # if the dev score is better than the last check point, then the current model is saved.
            # otherwise, we allow for that performance degeneration for up to `--patience` times;
            # if the dev score does not increase after `--patience` iterations, we reload the previously
            # saved best model (and the state of the optimizer), halve the learning rate and continue
            # training. This repeats for up to `--max-num-trial` times.
            if train_iter % valid_niter == 0:
                print('epoch %d, iter %d, cum. loss %.2f, cum. ppl %.2f cum. examples %d' % (epoch, train_iter,
                                                                                         cum_loss / cumulative_examples,
                                                                                         np.exp(cum_loss / cumulative_tgt_words),
                                                                                         cumulative_examples), file=sys.stderr)

                cum_loss = cumulative_examples = cumulative_tgt_words = 0.
                valid_num += 1

                print('begin validation ...', file=sys.stderr)

                # compute dev. ppl and bleu
                dev_ppl = model.evaluate_ppl(dev_data, batch_size=128)   # dev batch size can be a bit larger
                valid_metric = -dev_ppl

                print('validation: iter %d, dev. ppl %f' % (train_iter, dev_ppl), file=sys.stderr)

                is_better = len(hist_valid_scores) == 0 or valid_metric > max(hist_valid_scores)
                hist_valid_scores.append(valid_metric)

                if is_better:
                    patience = 0
                    print('save currently the best model to [%s]' % model_save_path, file=sys.stderr)
                    torch.save(model, model_save_path)

                    # You may also save the optimizer's state
                elif patience < int(args['--patience']):
                    patience += 1
                    print('hit patience %d' % patience, file=sys.stderr)
                    
                    if patience == int(args['--patience']):
                        num_trial += 1
                
                        print('hit #%d trial' % num_trial, file=sys.stderr)
                        if num_trial == int(args['--max-num-trial']):
                            print('early stop!', file=sys.stderr)
                            exit(0)
                        model.teach_rate *= 0.9
                        # decay learning rate, and restore from previously best checkpoint
                        lr = lr * float(args['--lr-decay'])
                        print('load previously best model and decay learning rate to %f' % lr, file=sys.stderr)

                        # load model
                        model = torch.load(model_save_path)
                        optimzer = torch.optim.Adam(model.parameters(), lr = lr)

                        print('restore parameters of the optimizers', file=sys.stderr)
                        # You may also need to load the state of the optimizer saved before
                        # reset patience
                        patience = 0

                if epoch == int(args['--max-epoch']):
                    print('reached maximum number of epochs!', file=sys.stderr)
                    exit(0)


def beam_search(model, test_data_src, beam_size, max_decoding_time_step):

    hypotheses = []
    for src_sent in tqdm(test_data_src, desc='Decoding', file=sys.stdout):
        example_hyps = model.greedy_search(src_sent, max_decoding_time_step=max_decoding_time_step)

        hypotheses.append(example_hyps)

    return hypotheses


def decode(args):
    """
    performs decoding on a test set, and save the best-scoring decoding results.
    If the target gold-standard sentences are given, the function also computes
    corpus-level BLEU score.
    """
    test_data_src = read_corpus(args['TEST_SOURCE_FILE'], source='src')
    if args['TEST_TARGET_FILE']:
        test_data_tgt = read_corpus(args['TEST_TARGET_FILE'], source='tgt')

    print(f"load model from {args['MODEL_PATH']}", file=sys.stderr)
    model = torch.load(args['MODEL_PATH'])
    model = model.to(DEVICE)

    hypotheses = beam_search(model, test_data_src, 5, max_decoding_time_step=int(args['--max-decoding-time-step']))

    if args['TEST_TARGET_FILE']:
        top_hypotheses = [hyps for hyps in hypotheses]
        bleu_score = compute_corpus_level_bleu_score(test_data_tgt, top_hypotheses)
        print(f'Corpus BLEU: {bleu_score}', file=sys.stderr)

    with open(args['OUTPUT_FILE'], 'w') as f:
        for src_sent, hyps in zip(test_data_src, hypotheses):
            top_hyp = hyps
            hyp_sent = ' '.join(top_hyp)
            f.write(hyp_sent + '\n')


def main():
    args = docopt(__doc__)

    # seed the random number generator (RNG), you may
    # also want to seed the RNG of tensorflow, pytorch, dynet, etc.
    seed = int(args['--seed'])
    np.random.seed(seed * 13 // 7)

    if args['train']:
        train(args)
    elif args['decode']:
        decode(args)
    else:
        raise RuntimeError(f'invalid mode')


if __name__ == '__main__':
    main()
