#coding:utf-8
from copy import deepcopy
from xml.etree import ElementTree as et
from lxml import etree as ET
import os
import sys
import re

#sentence -> [token, token, ...., token] and [tag, tag, ..., tag]
def xml2seq2(input_file):
    abst = ET.iterparse(input_file,tag='S')
    for event, sentence in abst:
        #print("<--sentence-->")
        #print(et.tostring(sentence).rstrip()[3:-4])
        #そのままだとバイナリ形式なので、デコードして前後の<S>タグを取り除いて出力
        string = et.tostring(sentence).rstrip()[3:-4].decode("utf-8")
        parse(string)
        print()
    return

def parse(sentence):
    BIOES = []
    parsed = []
    #文中の固有表現ごとに分析
    len_parsed = 0
    if re.search('<.*?>.*?</.*?>',sentence):
        ne = re.search('<.*?>.*?</.*?>',sentence)
        #print(ne)
        ne_word = re.compile(">.*?<")
        #tag =re.compile("</.*?>")
        tag =re.compile("\".*?\"")
        n = ne_word.search(ne.group(0),0)
        #print(n.group(0)[1:-1])
        t = tag.search(ne.group(0),0)
        #print(t.group(0)[2:-1])
        length = int(n.end()-1) - int(n.start()+1)

        prefix = sentence[:ne.start()]

        for i in prefix.split():
            print(i+'\t'+"O")
            BIOES.append("O")
        middle = sentence[ne.start() + n.start()+1 : ne.start() + n.start()+1+length]
        #print(middle)
        #print("S-"+t.group(0)[2:-1])
        BIOES.append("S-"+t.group(0)[2:-1])
        if " " in n.group(0)[1:-1]:
            length = len(n.group(0)[1:-1].split(" "))
            num = 0
            for token in n.group(0)[1:-1].split(" "):
                if length != 1:
                    if num == 0:
                        print(token+"\t"+"B-"+t.group(0)[1:-1])
                    elif num == length -1:
                        print(token+"\t"+"E-"+t.group(0)[1:-1])
                    else:
                        print(token+"\t"+"I-"+t.group(0)[1:-1])
                    num += 1

        else:
            print(n.group(0)[1:-1]+"\t"+"S-"+t.group(0)[1:-1])
        #print(prefix)
        if prefix != "\n":
            parsed.append(prefix[:-1])
        parsed.append(middle)
        len_parsed = len(prefix + middle)
        sentence = sentence[ne.end():]
        surfix = sentence
        #文残りをパーズする
        #print(parsed)
        #print(BIOES)
        """
        for token, tag in zip(parsed, BIOES):
            print(token+"\t"+tag)
        """
        return parse(surfix), BIOES

    else:
        #print(sentence)
        parsed = sentence.split()
        for i in parsed:
            BIOES.append("O")
        #print(parsed)
        #print(BIOES)
        #token分のNE_Tag配列を作る。
        for token, tag in zip(parsed, BIOES):
            print(token+"\t"+tag)
        return


def main():
    #print 形式で出力するので、|tee で保存
    argv = sys.argv
    input_file = argv[1]
    xml2seq2(input_file)

if __name__ == '__main__':
    main()
