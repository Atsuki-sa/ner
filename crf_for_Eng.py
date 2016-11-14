#crfで英文に固有表現認識（タグ付け）をする

#usr/bin/python3
#coding:utf-8
from itertools import chain
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer
import sklearn
import pycrfsuite
from sklearn import cross_validation
import sys
import os
import math
import random

def word2features(sent, i):
    word = sent[i][0]
    tag = sent[i][1]
    features = [
        'bias',
        'word.lower=' + word.lower(),
        'word[-3:]=' + word[-3:],
        'word[-2:]=' + word[-2:],
        'word.isupper=%s' % word.isupper(),
        'word.istitle=%s' % word.istitle(),
        'word.isdigit=%s' % word.isdigit(),
        #'tag=' + tag,
        #'tag[:2]=' + tag[:2],
    ]
    if i > 0:
        word1 = sent[i-1][0]
        tag1 = sent[i-1][1]
        features.extend([
            '-1:word.lower=' + word1.lower(),
            '-1:word.istitle=%s' % word1.istitle(),
            '-1:word.isupper=%s' % word1.isupper(),
            '-1:tag=' + tag1,
            '-1:tag[:2]=' + tag1[:2],
        ])
    else:
        features.append('BOS')

    if i < len(sent)-1:
        word1 = sent[i+1][0]
        tag1 = sent[i+1][1]
        features.extend([
            '+1:word.lower=' + word1.lower(),
            '+1:word.istitle=%s' % word1.istitle(),
            '+1:word.isupper=%s' % word1.isupper(),
            #'+1:tag=' + tag1,
            #s'+1:tag[:2]=' + tag1[:2],
        ])
    else:
        features.append('EOS')

    return features

def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, label in sent]

def sent2tokens(sent):
    return [token for token, label in sent]

def bio_classification_report(y_true, y_pred):
    """
    Classification report for a list of BIO-encoded sequences.
    It computes token-level metrics and discards "O" labels.

    Note that it requires scikit-learn 0.15+ (or a version from github master)
    to calculate averages properly!
    """
    lb = LabelBinarizer()
    y_true_combined = lb.fit_transform(list(chain.from_iterable(y_true)))
    y_pred_combined = lb.transform(list(chain.from_iterable(y_pred)))

    tagset = set(lb.classes_) - {'O'}
    #tagset = set(lb.classes_)
    tagset = sorted(tagset, key=lambda tag: tag.split('-', 1)[::-1])
    class_indices = {cls: idx for idx, cls in enumerate(lb.classes_)}

    return classification_report(
        y_true_combined,
        y_pred_combined,
        labels = [class_indices[cls] for cls in tagset],
        target_names = tagset,
    )

def main():
    argv = sys.argv
    dirname = argv[1]
    model = str(argv[2])
    #全部読む
    files = os.listdir(dirname)
    sentences = []
    #sentenceごとに読み込み。
    for filename in files:
        doc = open(dirname+filename,"r")
        tmp = []
        #文ごとにまとめてから追加する
        for line in doc:
            if line != "\n":
                word, tag = line.split("\t")
                tmp.append((word,tag[:-1]))
            else:
                sentences.append(tmp)
                tmp =[]

    #for line in sentences:
        #print(line)
    sum_of_sencences =len(sentences)
    #print("sentences: "+str(len(sentences)))
    #print(sentences[0])
    #sentenceをランダムに入れ替え
    random.shuffle(sentences)   #->変数代入するのではなく、元のリストがソートされる
    #print(sentences[0])
    #sentenceをtrain, testに分割

    test_size = math.floor(sum_of_sencences*0.9)
    test = sentences[0]
    test_sents = sentences[1:test_size]
    train_sents = sentences[test_size+1:]

    """
    #クロスバリデーションしてみる
    #モデルの問題？
    """

    X_train = [sent2features(s) for s in train_sents]
    y_train = [sent2labels(s) for s in train_sents]

    X_test = [sent2features(s) for s in test_sents]
    y_test = [sent2labels(s) for s in test_sents]

    #学習データと同一のトークンータグが含まれているかチェック
    trainer = pycrfsuite.Trainer(verbose=False)

    for xseq, yseq in zip(X_train, y_train):
        trainer.append(xseq, yseq)
    trainer.set_params({
    'c1': 1.0,   # coefficient for L1 penalty
    'c2': 1e-3,  # coefficient for L2 penalty
    'max_iterations': 100,  # stop earlier

    # include transitions that are possible, but not observed
    'feature.possible_transitions': True
    })
    #print(trainer.params())
    trainer.train(model)

    #print(len(trainer.logparser.iterations), trainer.logparser.iterations[-1])

    tagger = pycrfsuite.Tagger()
    tagger.open(model)

    example_sent = test
    for sentence in test_sents:
        for token, correct, predict in zip(sent2tokens(sentence), sent2labels(sentence), tagger.tag(sent2features(sentence))):
            print(token+"\t"+correct+"\t"+predict)
        print()
    """
    print(' '.join(sent2tokens(example_sent)), end='\n\n')

    print("Predicted:", ' '.join(tagger.tag(sent2features(example_sent))))
    print("Correct:  ", ' '.join(sent2labels(example_sent)))
    """
    #テストの中に、トレインのものがあるかチェック
    y_pred = [tagger.tag(xseq) for xseq in X_test]
    #print(bio_classification_report(y_test, y_pred))

if __name__ == '__main__':
    main()
