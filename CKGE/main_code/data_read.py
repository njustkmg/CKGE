from __future__ import absolute_import #3
from __future__ import division
from __future__ import print_function

import argparse
import collections
import multiprocessing
import os
import time as times
import logging
import json
import random
#from time import time as times
import numpy as np
import torch
import torch.nn
from torch import nn
from reader_data import load_vocab,load_txt,load_path

#长度为14,用于transformer中的mask
def getmask():
    mask = [[1,1,1,1,0,0,0,0,1,1,1,1],
[1,1,1,1,0,0,0,0,1,1,1,1],
[1,1,1,1,0,0,0,0,1,1,1,1],
[1,1,1,1,0,0,0,0,1,1,1,1],
[0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0],
[1,1,1,1,0,0,0,0,1,1,1,1],
[1,1,1,1,0,0,0,0,1,1,1,1],
[1,1,1,1,0,0,0,0,1,1,1,1],
[1,1,1,1,0,0,0,0,1,1,1,1]]
    mask = np.ones((12,12))
    return mask


#查找path节点的距离，构建距离矩阵
def distance_matric(path,vocab,short_dis):

    path_len = len(path)
    distance = np.zeros((path_len,path_len))
    for rows in range(path_len):
        entity_1 = vocab[path[rows]]
        for columns in range(path_len):
            entity_2 = vocab[path[columns]]
            if short_dis[entity_1][entity_2] > 10:
                distance[rows][columns] = 10
            else:
                if short_dis[entity_1][entity_2] != 0:
                    distance[rows][columns] = short_dis[entity_1][entity_2]
                else:
                    distance[rows][columns] = short_dis[entity_1][entity_2]
    return distance

#查找path节点的关系，构建关系矩阵
def relation_matric(path,vocab,entity_relation):

    path_len = len(path)
    rel_matric = np.zeros((path_len,path_len))
    for rows in range(path_len):
        entity_1 = path[rows]
        for columns in range(path_len):
            entity_2 = path[columns]
            if entity_relation.get((entity_1,entity_2)) is None:
                continue
            else:
                index = random.randint(0,len(entity_relation[(entity_1,entity_2)])-1)
                rel_matric[rows][columns] = vocab[entity_relation[(entity_1,entity_2)][index]]-23155
                
    return rel_matric

#查找用户以及课程之间的路径;路径中包含除用户及课程以外的2个节点，路径数量为2
def path_series(user,cou,neighbor_sample,path_bias_dict):
    mask = np.array(getmask())
    entity_paths = []
    if path_bias_dict.get((user,cou)) is not None:
        entity_paths = path_bias_dict[(user,cou)]
    series = []
    path_sample = 2; path_entity_sample = 2
    for path_index in range(path_sample):
        path = []
        if path_index < len(entity_paths):
            path = entity_paths[path_index]
        #series.append("[CLS]")
        
        for entity_index in range(path_entity_sample):
            if entity_index < len(path):
                series.append(path[entity_index])
            else:
                series.append("[PAD]")
                mask[:,(neighbor_sample+len(series))] = 0
    #series.append("[SEP]")
    return series, mask


#读取数据
def data_load():
    amount = 0
    vocab = load_vocab('e2c_vocab.txt')
    ntokens = len(vocab)
    print(ntokens)
    train = load_txt('e2c_train.txt')
    #print(len(train_data))
    val_data = load_txt('e2c_valid.txt')
    test = load_txt('e2c_test.txt')
    kg = load_txt('e2c_kg.txt')
    short_dis = np.load("short_dis.npy")
    path_bias_dict = load_path("path_rand_34.txt")
    
    print("===============================data information======================================")
    print("vocab_len : {}, train_len : {}, val_len : {}, test_len : {}, kg_len : {}".format(len(vocab), len(train), len(val_data), len(test), len(kg)))
    
    
    '''print("===============================short path======================================")
    start_time = times.time()
    node_short_dis = get_shortpath(adj_matrix)
    end_time = times.time()
    print("short_dis ( costtime : {}, dis_shape : {})".format((end_time-start_time), node_short_dis.shape))'''

    print("===============================user item neighbor======================================")
    entity_neighbor, course = {}, []
    entity_relation = {}
    for index in range(len(kg)):
        entity_0 = kg[index][0]
        rel = kg[index][1]
        entity_1 = kg[index][2]
        if entity_relation.get((entity_0,entity_1)) is None:
            entity_relation[(entity_0,entity_1)] = []
        entity_relation[(entity_0,entity_1)].append(rel)
        if rel >= '0' and rel <= '9':
            continue
        if rel[0] == 'P':
            if entity_1 not in course:
                course.append(entity_1)
        if entity_neighbor.get(entity_0) is None:
            entity_neighbor[entity_0] = []
        if entity_neighbor.get(entity_1) is None:
            entity_neighbor[entity_1] = []
        entity_neighbor[entity_0].append(entity_1)
        entity_neighbor[entity_1].append(entity_0)
    print("entity_amount : {}, course_amount : {}".format(len(entity_neighbor), len(course)))
    

    print("===============================train neighbor sample and spatial dataset======================================") 
    train_data,  train_pn_data, pn_label = [], [], []
    train_dis_data, train_pn_dis_data = [], []
    train_relation_data, train_pn_relation_data = [], []
    train_mask_data, train_pn_mask_data = [], []
    neighbor_sample = 3
    for index in range(len(train)):
        user,cou = [], []
        pos, neg = [], []
        user.append(train[index][0])
        cou.append(train[index][2])
        user_neighbor, cou_neighbor = [], []
        neighbor_number = neighbor_sample
        '''learnsamecourse_neighbor = []
        for node in entity_neighbor[train[index][0]]:
            if node in entity_neighbor[train[index][2]]:
                learnsamecourse_neighbor.append(node)'''
        while neighbor_number > 0:
            '''if len(learnsamecourse_neighbor) > 0:
                neighbor = learnsamecourse_neighbor[0]
                del learnsamecourse_neighbor[0]
                user_neighbor.append(neighbor)
                neighbor_number -= 1
                continue'''
            neighbor = entity_neighbor[train[index][0]][random.randint(0,len(entity_neighbor[train[index][0]])-1)]
            user_neighbor.append(neighbor)
            neighbor_number -= 1

        neighbor_number = neighbor_sample
        '''learnsamecourse_neighbor = []
        for node in entity_neighbor[train[index][2]]:
            if node in entity_neighbor[train[index][0]]:
                learnsamecourse_neighbor.append(node)'''
        while neighbor_number > 0:
            '''if len(learnsamecourse_neighbor) > 0:
                neighbor = learnsamecourse_neighbor[0]
                del learnsamecourse_neighbor[0]
                user_neighbor.append(neighbor)
                neighbor_number -= 1
                continue'''
            neighbor = entity_neighbor[train[index][2]][random.randint(0,len(entity_neighbor[train[index][2]])-1)]
            cou_neighbor.append(neighbor)
            neighbor_number -= 1

        user = np.concatenate((user,user_neighbor),axis=0)
        path, mask = path_series(train[index][0],train[index][2],neighbor_sample,path_bias_dict)
        #user = np.concatenate((user,path),axis=0)
        user = np.concatenate((user,path),axis=0)
        user = np.concatenate((user,cou_neighbor),axis=0)
        pos = np.concatenate((user,cou),axis=0)
        '''if cou[0] in path: 
            amount += 1
            print(pos,amount,len(train_data))'''

        while True:
            neg_index = random.randint(0,len(course)-1)
            if course[neg_index] not in entity_neighbor[train[index][0]]:
                neg.append(course[neg_index])
                break
        neg = np.concatenate((user,neg),axis=0)

        train_data.append(pos)
        train_pn_data.append(pos)
        train_pn_data.append(neg)
        pn_label.append(1)
        pn_label.append(0)
        
        pos_dis = distance_matric(pos,vocab,short_dis)
        train_dis_data.append(pos_dis); train_pn_dis_data.append(pos_dis)
        neg_dis = distance_matric(neg,vocab,short_dis)
        train_pn_dis_data.append(neg_dis)
        
        pos_relation = relation_matric(pos,vocab,entity_relation);
        train_relation_data.append(pos_relation);train_pn_relation_data.append(pos_relation)
        neg_relation = relation_matric(neg,vocab,entity_relation)
        train_pn_relation_data.append(neg_relation)
        
        train_mask_data.append(mask)
        train_pn_mask_data.append(mask);train_pn_mask_data.append(mask)
        
    print("train_len : {}".format(len(train_data)))
    print("train_data shape : ", np.array(train_data).shape,'\n',
          "train_pn_data shape : ", np.array(train_pn_data).shape,'\n',
          "train_dis_data shape : ", np.array(train_dis_data).shape,'\n',
          "train_pn_dis_data shape : ", np.array(train_pn_dis_data).shape,'\n',
          "train_relation_data shape : ", np.array(train_relation_data).shape,'\n',
          "train_pn_relation_data shape : ", np.array(train_pn_relation_data).shape,'\n',
          "train_mask_data shape : ", np.array(train_mask_data).shape,'\n',
          "train_pn_mask_data shape : ", np.array(train_pn_mask_data).shape)
    
    print("===============================test neighbor sample dataset======================================") 
    test_data,  test_pn_data = [], []
    test_dis_data, test_pn_dis_data = [], []
    test_relation_data, test_pn_relation_data = [], []
    test_mask_data, test_pn_mask_data = [], []
    for index in range(len(test)):
        user,cou = [], []
        user.append(test[index][0])
        cou.append(test[index][1])
        user_neighbor, cou_neighbor = [], []
        neighbor_number = neighbor_sample
        while neighbor_number > 0:
            neighbor = entity_neighbor[test[index][0]][random.randint(0,len(entity_neighbor[test[index][0]])-1)]
            user_neighbor.append(neighbor)
            neighbor_number -= 1

        neighbor_number = neighbor_sample
        while neighbor_number > 0:
            neighbor = entity_neighbor[test[index][1]][random.randint(0,len(entity_neighbor[test[index][1]])-1)]
            cou_neighbor.append(neighbor)
            neighbor_number -= 1

        user = np.concatenate((user,user_neighbor),axis=0)
        path, mask = path_series(test[index][0],test[index][1],neighbor_sample,path_bias_dict)
        #print("path_series : ", user,path)
        user = np.concatenate((user,path),axis=0)
        user = np.concatenate((user,cou_neighbor),axis=0)
        user = np.concatenate((user,cou),axis=0)
        test_data.append(user)
        test_pn_data.append(user)
        
        user_dis = distance_matric(user,vocab,short_dis)
        test_dis_data.append(user_dis)
        
        user_relation = relation_matric(user,vocab,entity_relation)
        test_relation_data.append(user_relation)
        
        test_mask_data.append(mask)
        
    print("test_len : {}".format(len(test_data))) 
    print("test_data shape : ", np.array(test_data).shape,'\n',
          "test_pn_data shape : ", np.array(test_pn_data).shape,'\n',
          "test_dis_data shape : ", np.array(test_dis_data).shape,'\n',
          "test_relation_data shape : ", np.array(test_relation_data).shape,'\n',
          "test_mask_data shape : ", np.array(test_mask_data).shape)

    return vocab, ntokens, train, val_data, test, kg, entity_neighbor, course, \
train_data, train_pn_data, pn_label, test_data, test_pn_data, train_dis_data, train_pn_dis_data, test_dis_data,\
train_relation_data, train_pn_relation_data, test_relation_data, train_mask_data, train_pn_mask_data, test_mask_data


#数据处理，变为张量
def data_process(data, vocab):
    data_list=[]
    for triplet in data:
        triplet_list=[]
        for element in triplet:
            triplet_list.append(vocab[element])
        triplet_list[11] = triplet_list[11]-16538
        data_list.append(triplet_list)
    data = torch.tensor(data_list,dtype = torch.long)
    return data

def data_pn_process(data, vocab):
    data_list=[]
    for triplet in data:
        triplet_list=[]
        for element in triplet:
            triplet_list.append(vocab[element])
        data_list.append(triplet_list)
    data = torch.tensor(data_list,dtype = torch.long)
    return data


                                                    
                                   
                                   
                                   
                                   
                                   