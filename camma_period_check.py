#in: トークンと単語区切り形式のNERデータ
#out: トークンにコンマとピリオドがあった場合分割して整形して再出力

#coding:utf-8
import sys
import os

def _files(path):
    files = os.listdir(path)
    for _file in files:
        doc = open(path+_file,"r")
        for line in doc:
            if line != "\n":
                #print(line)
                word, tag = line.split("	")
                if len(word) > 2:
                    if "." in word or "," in word:
                        new_word1 = word[:-1]
                        new_word2 = word[-1:]
                        #if tag == "O":
                        if tag.strip() == "O":
                            pass
                            print(new_word1+"\t"+tag[:-1])
                            print(new_word2+"\t"+tag[:-1])
                        else:
                            new_tag1 = tag[0:1]
                            #print("Before : "+new_tag1)
                            new_tag2 = tag[1:]
                            if new_tag1 == "S":
                                new_word_tag1 = new_word1+"\t"+"S"+new_tag2
                                new_word_tag2 = new_word2+"\t"+"O"+new_tag2
                                print(new_word_tag1[:-1])
                                print(new_word_tag2[:-1])
                            elif new_tag1 == "B":
                                new_word_tag1 = new_word1+"\t"+"B"+new_tag2
                                new_word_tag2 = new_word2+"\t"+"I"+new_tag2
                                print(new_word_tag1[:-1])
                                print(new_word_tag2[:-1])
                            elif new_tag1 == "I":
                                new_word_tag1 = new_word1+"\t"+"I"+new_tag2
                                new_word_tag2 = new_word2+"\t"+"I"+new_tag2
                                print(new_word_tag1[:-1])
                                print(new_word_tag2[:-1])
                            elif new_tag1 == "E":
                                new_word_tag1 = new_word1+"\t"+"I"+new_tag2
                                new_word_tag2 = new_word2+"\t"+"E"+new_tag2
                                print(new_word_tag1[:-1])
                                print(new_word_tag2[:-1])
                            else:
                                print("error")

                    else:
                        pass
                        print(line[:-1])
                else:
                    pass
                    print(line[:-1])
            else:
                pass
                print()

#word,word -> word , word
def split_three_camma(word,tag):
    find = word.find(",")
    word_len = len(word)
    new_word1 = word[:find]
    new_word2 = word[find]
    new_word3 = word[find+1:]
    if tag.strip() == "O":
        pass
        print(new_word1+"\t"+tag[:-1])
        print(new_word2+"\t"+tag[:-1])
        print(new_word3+"\t"+tag[:-1])
    else:
        new_tag1 = tag[0:1]
        #print("Before : "+new_tag1)
        new_tag2 = tag[1:]
        if new_tag1 == "S":
            new_word_tag1 = new_word1+"\t"+"B"+new_tag2
            new_word_tag2 = new_word2+"\t"+"I"+new_tag2
            new_word_tag3 = new_word3+"\t"+"E"+new_tag2
            print(new_word_tag1[:-1])
            print(new_word_tag2[:-1])
            print(new_word_tag3[:-1])

        elif new_tag1 == "B":
            new_word_tag1 = new_word1+"\t"+"B"+new_tag2
            new_word_tag2 = new_word2+"\t"+"I"+new_tag2
            new_word_tag3 = new_word3+"\t"+"I"+new_tag2
            print(new_word_tag1[:-1])
            print(new_word_tag2[:-1])
            print(new_word_tag3[:-1])

        elif new_tag1 == "I":
            new_word_tag1 = new_word1+"\t"+"I"+new_tag2
            new_word_tag2 = new_word2+"\t"+"I"+new_tag2
            new_word_tag3 = new_word3+"\t"+"I"+new_tag2
            print(new_word_tag1[:-1])
            print(new_word_tag2[:-1])
            print(new_word_tag3[:-1])

        elif new_tag1 == "E":
            new_word_tag1 = new_word1+"\t"+"I"+new_tag2
            new_word_tag2 = new_word2+"\t"+"I"+new_tag2
            new_word_tag3 = new_word3+"\t"+"E"+new_tag2
            print(new_word_tag1[:-1])
            print(new_word_tag2[:-1])
            print(new_word_tag3[:-1])

        else:
            print("error")

#word.word -> word . word
def split_three_period(word, tag):
    find = word.find(".")
    word_len = len(word)
    new_word1 = word[:find]
    new_word2 = word[find]
    new_word3 = word[find+1:]
    if tag.strip() == "O":
        pass
        print(new_word1+"\t"+tag[:-1])
        print(new_word2+"\t"+tag[:-1])
        print(new_word3+"\t"+tag[:-1])
    else:
        pass
#word, -> word ,
def split_two(word,tag):
    new_word1 = word[:-1]
    new_word2 = word[-1:]
    #if tag == "O":
    if tag.strip() == "O":
        pass
        print(new_word1+"\t"+tag[:-1])
        print(new_word2+"\t"+tag[:-1])
    else:
        new_tag1 = tag[0:1]
        #print("Before : "+new_tag1)
        new_tag2 = tag[1:]
        if new_tag1 == "S":
            new_word_tag1 = new_word1+"\t"+"S"+new_tag2
            new_word_tag2 = new_word2+"\t"+"O"+new_tag2
            print(new_word_tag1[:-1])
            print(new_word_tag2[:-1])
        elif new_tag1 == "B":
            new_word_tag1 = new_word1+"\t"+"B"+new_tag2
            new_word_tag2 = new_word2+"\t"+"I"+new_tag2
            print(new_word_tag1[:-1])
            print(new_word_tag2[:-1])
        elif new_tag1 == "I":
            new_word_tag1 = new_word1+"\t"+"I"+new_tag2
            new_word_tag2 = new_word2+"\t"+"I"+new_tag2
            print(new_word_tag1[:-1])
            print(new_word_tag2[:-1])
        elif new_tag1 == "E":
            new_word_tag1 = new_word1+"\t"+"I"+new_tag2
            new_word_tag2 = new_word2+"\t"+"E"+new_tag2
            print(new_word_tag1[:-1])
            print(new_word_tag2[:-1])
        else:
            print("error")
    return

def word_tag(word,tag):
    #. , が文字の最後かどうか調べる
    if word.find(","):
        if word.find(",") == -1 or word.find(",") == len(word)-1:
            split_two(word, tag)
        else:
            #print(word)
            split_three_camma(word,tag)

    elif word.find("."):
        if word.find(".") == -1 or word.find(".") == len(word)-1:
            split_two(word, tag)
        else:
            #print(word)
            split_three_period(word, tag)
    else:
        pass
    return

def _file(path):
    doc = open(path,"r")
    for line in doc:
        if line != "\n":
            #print(line)
            word, tag = line.split("	")
            if len(word) > 2:
                if "." in word or "," in word:
                    word_tag(word, tag)
                else:
                    print(line[:-1])
            else:
                print(line[:-1])
        else:
            print()
def main():
    argv = sys.argv
    path = argv[1]
    _file(path)
if __name__ == '__main__':
    main()
