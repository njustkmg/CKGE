import torch
from torch import nn
import numpy as np
import six
import collections
import logging
#import batching

def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    fin = open(vocab_file)
    for num, line in enumerate(fin):
        items = line.strip().split("\t")
        if len(items) > 2:
            break
        token = items[0]
        index = items[1] if len(items) == 2 else num
        token = token.strip()
        vocab[token] = int(index)
    return vocab

def load_txt(txt_path):
    text = open(txt_path)
    data_list=[]
    for num,triple in enumerate(text):
        items = triple.strip().split("\t")
        triples_before=[]
        for vac in items:
            triples_before.append(vac)
        data_list.append(triples_before)
    return data_list

#每条路径长度为4
def load_path(path_file):
    entity_path = {}
    
    '''with open("path_rand_34.txt", "r") as f:
        for line in f:
            if len(line) < 3 or line == '':
                continue
            else:
                line = line.strip().split("|")
                for path in line:
                    path = path.strip().split(" ")
                    if entity_path.get((path[0],path[-1])) is None:
                        entity_path[(path[0],path[-1])] = []
                    entity_path[(path[0],path[-1])].append([path[1],path[2]])'''
    
    with open(path_file, "r") as f:
        for line in f:
            if len(line) < 3 or line == '':
                continue
            else:
                line = line.strip().split("|")
                for path in line:
                    path = path.strip().split(" ")
                    if entity_path.get((path[0],path[-1])) is None:
                        entity_path[(path[0],path[-1])] = []
                    entity_path[(path[0],path[-1])].append([path[1],path[2]])
                    
    return entity_path
# def load_train_txt(txt_path):
#     text = open(txt_path)
#     train_list_tail=[]
#     train_list_head=[]
#     for num,triple in enumerate(text):
#         items = triple.strip().split("\t")
#         triples_before=[]
#         triples_back = []
#         for vac in items:
#             triples_before.append(vac)
#             triples_back.insert(0,vac)
#         train_list_tail.append(triples_before)
#         train_list_head.append(triples_back)
#     return train_list_head,train_list_tail
