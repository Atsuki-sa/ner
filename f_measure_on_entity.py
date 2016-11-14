#エンティティー単位で認識結果のF値を計測

import sys

def get_tagset(out_file):
    tagset = set()
    for line in out_file:
        #print(line[:-1])
        if line != "\n":
            token, gold, predict = line.split("\t")
            if gold not in tagset:
                tagset.add(gold)
    return tagset

def get_entity_class(tagset):
    entity = set()
    for tag in tagset:
        if tag == "O":
            entity.add(tag)
        else:
            if tag[2:] not in entity:
                entity.add(tag[2:])
    #print(entity)
    return entity

def get_entity_counter(entities):
    entity_counter = dict()

    """
    #[tp, fn, fp, fn]
    tp = gold: True model: True
    fn = gold: True model: False
    fp = gold: False model: True
    fn = gold: False model: False
    """

    for entity in entities:
        #print(entity)
        entity_counter[entity] = [0,0,0,0]
    return entity_counter

def count_whole_f_score(doc, en_counter):
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    seq_tp = False
    for line in doc:
        #print(line)
        if line != "\n":
            token, gold, predict = line[:-1].split("\t")
            #print(gold)
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

    print_result(tp, fp, fn)
    return en_counter

def precision(tp, fp):
    return tp / (tp + fp)

def recall(tp, fn):
    return tp / (tp + fn)

def f_measure(tp,fp,fn):
    return float(2 * recall(tp,fn) * precision(tp, fp) / (recall(tp, fn) + precision(tp, fp)))

def print_result(tp, fp, fn):
    print("TP: "+str(tp))
    print("FP: "+str(fp))
    print("FN: "+str(fn))
    print("")
    print("Precision: "+str(precision(tp, fp)))
    print("Recall: "+str(recall(tp, fn)))
    print("F measure: "+str(f_measure(tp,fp, tp)))
    return
def main():
    argv = sys.argv
    path = argv[1]

    #file load
    doc = open(path, "r")

    #make tagset
    tagset = get_tagset(doc)

    #make entity class set
    entities = get_entity_class(tagset)

    #make entity counter
    en_counter = get_entity_counter(entities)

    #file load
    doc = open(path, "r")

    #check exch entity (B-O, S)
    count_whole_f_score(doc, en_counter)

if __name__ == '__main__':
    main()
