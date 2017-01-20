#!/usr/bin/env python
#coding:utf-8
import sys
import os
import random
import math
from collections import OrderedDict

import numpy as np
import theano
from theano import tensor as T
theano.config.floatX = "float32"

import f_measure_on_entity

vocab = {}
inv_vocab ={}

tag_vocab = {}
inv_tag_vocab = {}

#word -> ID に変換
def load_data(filename):
    global vocab, n_vocab
    global tag_vocab, n_tag

    doc = open(filename,"r")
    words = []
    tags = []

    for line in doc:
        if line != "\n":
            word, tag =line.split("\t")
            #print(word)
            words.append(word)
            tags.append(tag[:-1])
    #words = open(filename).read().replace('\n','<eos>').strip().split()
    dataset = np.ndarray((len(words),), dtype = np.int32)
    tag_dataset = np.ndarray((len(tags),), dtype = np.int32)

    #word -> IDdic
    for i, word in enumerate(words):
        if word not in vocab:
            vocab[word] = len(vocab)
            inv_vocab[len(vocab) -1] = word
        dataset[i] = vocab[word]

    #NEtag -> IDdic
    for i, tag in enumerate(tags):
        if tag not in tag_vocab:
            tag_vocab[tag] = len(tag_vocab)
            inv_tag_vocab[len(tag_vocab) -1] = tag
        tag_dataset[i] = tag_vocab[tag]
    return dataset, tag_dataset

#word -> ID
def char2id(token,dataset):
    for data in vocab:
        if data == token:
            #print("found : "+str(vocab[data]))
            return str(vocab[data])
    #print("not found")
    return

#NEtag -> ID
def tag2id(token,tag_dataset):
    for data in tag_vocab:
        if data == token:
            #print("found : "+str(tag_vocab[data]))
            return str(tag_vocab[data])
    #print("not found")
    return

def load_all(filename, files):
    sentences = []
    for _file in files:
        dataset, tag_dataset = load_data(filename+_file)

        doc = open(filename+_file,"r")
        tmp = []
        #文ごとにまとめてから追加する
        for line in doc:
            if line != "\n":
                word, tag = line.split("\t")
                tmp.append((word,tag[:-1]))
            else:
                sentences.append(tmp)
                tmp =[]
    return sentences, dataset, tag_dataset

def load_all_from_doc(filename):
    sentences = []
    dataset, tag_dataset = load_data(filename)
    doc = open(filename,"r")
    tmp = []
    #文ごとにまとめてから追加する
    for line in doc:
        if line != "\n":
            word, tag = line.split("\t")
            tmp.append((word,tag[:-1]))
        else:
            sentences.append(tmp)
            tmp =[]
    return sentences, dataset, tag_dataset

#データ分割
def data_split(sentences):
    sum_of_sencences =len(sentences)
    random.shuffle(sentences)   #->変数代入するのではなく、元のリストがソートされる
    test_size = math.floor(sum_of_sencences*0.1)
    dev_size = math.floor(sum_of_sencences*0.2)

    test_sents  = sentences[:test_size]
    dev_sents   = sentences[test_size+1:dev_size]
    train_sents = sentences[dev_size+1:]

    return test_sents, dev_sents ,train_sents

#前後二単語をとってきたベクトルを作る（二次元配列）
def contextwin(l, win):
    assert (win % 2) == 1
    assert win >= 1
    l = list(l)

    lpadded = win // 2 * [-1] + l + win // 2 * [-1]
    out = [lpadded[i:(i + win)] for i in range(len(l))]

    assert len(out) == len(l)
    return out

def change_to_id():
    sentence = []
    labels = []
    example = "model train . phrase-based translation models ."
    example_tag = "S-model O O B-model I-model E-model O"

    for token, tag in zip(example.split(" "),example_tag.split(" ")):
        #print(token)
        #print(tag)
        sentence.append(str(char2id(token,dataset))+" ")
        labels.append(str(tag2id(tag,tag_dataset))+" ")
    #print(example)
    """
    for token, tag in zip(sentence, labels):
        print(token,tag)
    """
    #print(sentence)
    #print(labels)
    return

def to_id(sentences,dataset,tag_dataset):
    #id化
    sent_id = []
    tag_id  = []
    ided_sentence = []
    for sentences in sentences:
        sent_id = []
        tag_id  = []
        for line in sentences:
            #print(line)
            token, tag = line[0],line[1]
            sent_id.append(int(char2id(token,dataset)))
            tag_id.append(int(tag2id(tag,tag_dataset)))
        #print(sent_id)
        #print(tag_id)
        #print()
        ided_sentence.append([sent_id, tag_id])
    return ided_sentence

def main():
    argv = sys.argv
    dirname = argv[1]
    epoch = int(argv[2])
    if ".txt" not in argv[1]:
        files = os.listdir(dirname)
        sentences, dataset, tag_dataset = load_all(dirname, files)
    else:
        sentences, dataset, tag_dataset = load_all_from_doc(dirname)


    ided_sentence = to_id(sentences,dataset,tag_dataset)

    test_sents, dev_sents, train_sents = data_split(ided_sentence)
    #学習データとテストデータとdevデータの作成

    nh = 100 # dimension of the hidden layer中間層のサイズ
    nc = len(tag_vocab) # number of classes分類数
    ne = len(vocab) # number of word embeddings in the vocabulary
    de = 50 # dimension of the word embeddings
    cs = 5 # word window context size

    emb = theano.shared(name='embeddings', value=0.2 * np.random.uniform(-1.0, 1.0, (ne+1, de)).astype(theano.config.floatX))

    print(np.random.uniform(-1.0, 1.0, (de * cs, nh)))
    #(250,100)コンテクストウィンドウと中間層のサイズのウィンドウにする
    wx = theano.shared(name='wx', value=0.2 * np.random.uniform(-1.0, 1.0, (de * cs, nh)).astype(theano.config.floatX))

    wh = theano.shared(name='wh', value=0.2 * np.random.uniform(-1.0, 1.0, (nh, nh)) .astype(theano.config.floatX))

    w = theano.shared(name='w', value=0.2 * np.random.uniform(-1.0, 1.0, (nh, nc)).astype(theano.config.floatX))

    #バイアスはs全て０でよい
    bh = theano.shared(name='bh', value=np.zeros(nh, dtype=theano.config.floatX))
    b  = theano.shared(name='b', value=np.zeros(nc, dtype=theano.config.floatX))
    h0 = theano.shared(name='h0', value=np.zeros(nh, dtype=theano.config.floatX))

    params = [emb, wx, wh, w, bh, b, h0]

    idxs = T.imatrix()

    x = emb[idxs].reshape((idxs.shape[0], de*cs))
    y_sentence = T.ivector('y_sentence')

    def recurrence(x_t, h_tm1):
    	h_t = T.nnet.sigmoid(T.dot(x_t, wx) + T.dot(h_tm1, wh) + bh)
        #h_t = T.nnet.relu(T.dot(x_t, wx) + T.dot(h_tm1, wh) + bh)とか
        #x[::-1]で並び替えししてLSTM
    	s_t = T.nnet.softmax(T.dot(h_t, w) + b)
    	return [h_t, s_t]

    [h, s], _ = theano.scan(fn=recurrence, sequences=x, outputs_info=[h0, None], n_steps=x.shape[0])
    #fn = 反復処理をする関数
    #sequence = 逐次処理を行う際、要素を進めながら入力をおこなうList.Matrixタイプの変数　ここでは単語(のウインドウをとってembしたもの)x
    #output_info = 逐次処理の初期値
    #n_steps =　繰り返し回数(文の長さ分)
    #いらない次元を消す
    p_y_given_x_sentence = s[:, 0, :]
    y_pred = T.argmax(p_y_given_x_sentence, axis=1)
    #softmaxしたもののうち、もっとも確率が高いもの
    ##学習
    lr = T.scalar('lr') #学習率

    #negative log likelyhood 負の対数尤度
    #t.arangeはi for i in range(16)
    #y_sentenceはゴールド
    #文全体の誤差をクロスエントロピーで算出
    sentence_nll = -T.mean(T.log(p_y_given_x_sentence) [T.arange(x.shape[0]), y_sentence])

    #自動微分して、勾配を求める
    sentence_gradients = T.grad(sentence_nll, params)
    f = theano.function([idxs, y_sentence], sentence_gradients)

    #現在の値から学習率をかけた勾配をかける
    sentence_updates = OrderedDict((p, p - lr*g) for p, g in zip(params, sentence_gradients))

    f = theano.function([idxs, y_sentence], sentence_nll)

    classify = theano.function(inputs=[idxs], outputs=y_pred)
    sentence_train = theano.function(inputs=[idxs, y_sentence, lr], outputs=sentence_nll, updates=sentence_updates)

    #train model
    for i in range(epoch):
        print("Epoch: "+str(i))
        random.shuffle(train_sents)
        for train in train_sents:
            x, y = train[0],train[1]
            #print(x)
            #print(y)
            cn_x = contextwin(x, cs)
            sentence_train(cn_x, y,0.1)
            #print(i, sentence_train(cn_x, y,0.1))

    tp = 0
    fp = 0
    fn = 0
    tn = 0
    #BIOES用
    seq_tp = False

    #BIO用
    m_correct = False
    g_correct = False
    m_line = False
    g_line = False

    for test in test_sents:
        x, y = test[0], test[1]
        cn_x = contextwin(x, cs)
        print("sentence : "+str([inv_vocab[id_x] for id_x in x ]))
        print("gold : "+str([inv_tag_vocab[tag] for tag in y]))
        print("predict : "+str([inv_tag_vocab[tag] for tag in classify(cn_x)]))
        for gold_id, pred_id in zip(y, classify(cn_x)):
            gold = inv_tag_vocab[gold_id]
            predict = inv_tag_vocab[pred_id]
            #BIOES
            """
            if gold[0] == "S":
                #print(gold[0])
                if gold == predict:
                    tp += 1
                else:
                    fn += 1
            elif gold[0] == "B":
                #print(token+"\t||\t"+gold+"\t"+predict)
                if gold == predict:
                    seq_tp = True
                    #print(seq_tp)
                else:
                    pass
                    #print(seq_tp)
            elif gold[0] == "I":
                #print(token+"\t||\t"+gold+"\t"+predict)
                if gold == predict and seq_tp == True:
                    seq_tp = True
                    #print(seq_tp)
                else:
                    seq_tp =False
                    #print(seq_tp)
            elif gold[0] == "E":
                #print(token+"\t||\t"+gold+"\t"+predict)
                if gold == predict and seq_tp == True:
                    tp += 1
                    #print(seq_tp)
                    #print("<<<True>>>")
                    seq_tp = False
                else:
                    fn += 1
                    #print(seq_tp)
                    seq_tp = False
                #print("")
            elif gold[0] == "O":
                if gold == predict:
                    tn += 1
                else:
                    fp += 1
            """
            if "B" in predict:
                m_line =True
                if gold == predict:
                    m_correct = True
                else:
                    m_correct = False
            elif "I" in predict:
                if gold == predict:
                    m_correct = True
                else:
                    m_correct = False

            elif "O" in predict:
                if m_line ==True:
                    if m_correct == True:
                        tp += 1
                        m_correct = False
                    else:
                        fp += 1
                        m_correct = False
                    m_line = False

            if "B" in gold:
                g_line = True
                if gold == predict:
                    g_correct = True
                else:
                    g_correct = False
            elif "I" in gold:
                if gold == predict:
                    g_correct = True
                else:
                    g_correct = False

            elif "O" in gold:
                if g_line == True:
                    if g_correct == True:
                        pass
                        #tp += 1
                        #g_correct = False
                    else:
                        fn += 1
                        g_correct = False
                    g_line = False

    #文の数、学習率エポック数などの設定ファイルを出す
    f_measure_on_entity.print_result(tp, fp, fn, tn)
    #classごとの分布を出すmatplotlibとかで分布
    print()

if __name__ == '__main__':
    main()
