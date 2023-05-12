from __future__ import absolute_import #6
from __future__ import division
from __future__ import print_function

import argparse
import collections
import multiprocessing
import os
import time
import logging
import json
import random
from time import time as times

import numpy as np
import torch
import torch.nn
from torch import nn
from coke1 import CoKEModel
from reader_data import load_vocab,load_txt
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from data_read import *
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO)
logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger(__name__)

#读取数据
vocab,ntokens,train,val_data,test,kg,entity_neighbor,course,train_data,train_pn_data,pn_label,test_data,test_pn_data, \
train_dis_data, train_pn_dis_data, test_dis_data, train_relation_data, train_pn_relation_data, test_relation_data, train_mask, train_pn_mask, test_mask = data_load()


#预热学习
class GradualWarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier < 1.:
            raise ValueError('multiplier should be greater thant or equal to 1.')
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super(GradualWarmupScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_last_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch if epoch != 0 else 1  # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
        if self.last_epoch <= self.total_epoch:
            warmup_lr = [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group['lr'] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.total_epoch)

    def step(self, epoch=None, metrics=None):
        if type(self.after_scheduler) != ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.total_epoch)
                self._last_lr = self.after_scheduler.get_last_lr()
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)

class EMA():
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
 
    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
 
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
 
    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]
 
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
print(device)

#将数据转换成tensor
train_data= data_process(train_data, vocab).to(device)
#val_data = data_process(val_data)  .to(device)
test_data = data_process(test_data, vocab) .to(device)
train_pn_data = data_pn_process(train_pn_data, vocab).to(device)
test_pn_data = data_pn_process(test_pn_data, vocab).to(device)
pn_label = torch.tensor(pn_label,dtype = torch.float).to(device)
kg = data_pn_process(kg, vocab).to(device)

train_dis_data = torch.tensor(train_dis_data,dtype = torch.long).to(device)
train_pn_dis_data = torch.tensor(train_pn_dis_data,dtype = torch.long).to(device)
test_dis_data = torch.tensor(test_dis_data,dtype = torch.long).to(device)

train_relation_data = torch.tensor(train_relation_data,dtype = torch.long).to(device)
train_pn_relation_data = torch.tensor(train_pn_relation_data,dtype = torch.long).to(device)
test_relation_data = torch.tensor(test_relation_data,dtype = torch.long).to(device)

train_mask = torch.tensor(train_mask,dtype = torch.bool).to(device)
train_pn_mask = torch.tensor(train_pn_mask,dtype = torch.bool).to(device)
test_mask = torch.tensor(test_mask,dtype = torch.bool).to(device)




lr = 0.001#0.0005  # learning rate
#定义模型参数
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.5)
batch_size = 512
#model = CoKEModel(voc_size=ntokens, emb_size =256 , nhead = 4, nhid = 512, nlayers = 12 , dropout = 0.1,soft_label=0.8,batch_size = batch_size).to(device)
model = CoKEModel(voc_size=ntokens, emb_size =256 , nhead = 1, nhid = 512, nlayers = 1 , dropout = 0.1,soft_label=0.8,batch_size = batch_size).to(device)
#optimizer = torch.optim.Adam(model.parameters(), lr=lr)
optimizer = torch.optim.Adam(model.parameters(),lr =lr)
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer,  2,gamma=0.9)

warmup_epoch = 60
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,  50-warmup_epoch)
#scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epoch,after_scheduler=scheduler)
scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epoch)

#ema = EMA(model, 0.9999)
#ema.register()



print(train_data.size(0))

print("===============================mask data======================================")
mask_tail_train = torch.ByteTensor([[0,0,0,0,0,0,0,0,0,0,0,1]])
mask_head_train = torch.ByteTensor([[1,0,0,0,0,0,0,0,0,0,0,0]])
for i in range(int(np.log2(batch_size))):
    mask_tail_train = torch.cat((mask_tail_train,mask_tail_train),0).to(device)
    mask_head_train = torch.cat((mask_head_train,mask_head_train),0).to(device)
    
mask_tail_test = torch.ByteTensor([[0,1,1,1,1,1,1,1,1,1,1,1]])
mask_head_test = torch.ByteTensor([[1,1,1,1,1,1,1,1,1,1,1,0]])
for i in range(int(np.log2(batch_size))):
    mask_tail_test = torch.cat((mask_tail_test,mask_tail_test),0).to(device)
    mask_head_test = torch.cat((mask_head_test,mask_head_test),0).to(device)


#按batch取出数据
def get_batch(train_data, i,flag, split,dis_data=None, rel_data=None, mask_data=None):

    if i+batch_size>  train_data.size(0):
       i = 0
    if flag == 'tail'  :
        if split == 'train':
            data_mask_tail = train_data[i:i+batch_size,:]
            data_mask_tail = data_mask_tail.masked_fill(mask=mask_tail_train.bool(), value=99)
            target_mask_tail = train_data[i:i+batch_size,11:12]
            distance_matrix = dis_data[i:i+batch_size,:]
            realtion_matrix = rel_data[i:i+batch_size,:]
            mask_matrix = mask_data[i:i+batch_size,:]
            return  data_mask_tail.to(device),\
                    target_mask_tail.to(device),\
                    distance_matrix.to(device),\
                    realtion_matrix.to(device),\
                    mask_matrix.to(device)
        elif split == 'eval':
            data_mask_tail = train_data[i:i+batch_size,:]
            data_mask_tail = data_mask_tail.masked_fill(mask=mask_tail_test.bool(), value=99)
            target_mask_tail = train_data[i:i+batch_size,11:12]
            distance_matrix = dis_data[i:i+batch_size,:]
            realtion_matrix = rel_data[i:i+batch_size,:]
            return  data_mask_tail.to(device),target_mask_tail.to(device),distance_matrix.to(device),realtion_matrix.to(device)


def get_pn_batch(train_data, i, pn_label, dis_data=None, rel_data=None, mask_data=None):

    if i+batch_size>  train_data.size(0):
       i = 0
    data_pn_1 = train_data[i:(i+batch_size),:]
    target_pn_1 = pn_label[i:(i+batch_size)]
    pn_distance_matrix_1 = dis_data[i:(i+batch_size),:]
    pn_relation_matrix_1 = rel_data[i:(i+batch_size),:]
    pn_mask_matrix_1 = mask_data[i:(i+batch_size),:]
    data_pn_2 = train_data[i+int(len(train_data)/2):(i+batch_size)+int(len(train_data)/2),:]
    target_pn_2 = pn_label[i+int(len(train_data)/2):(i+batch_size)+int(len(train_data)/2)]
    pn_distance_matrix_2 = dis_data[i+int(len(train_data)/2):(i+batch_size)+int(len(train_data)/2),:]
    pn_relation_matrix_2 = rel_data[i+int(len(train_data)/2):(i+batch_size)+int(len(train_data)/2),:]
    pn_mask_matrix_2 = mask_data[i+int(len(train_data)/2):(i+batch_size)+int(len(train_data)/2),:]
    return  data_pn_1.to(device),\
            target_pn_1.to(device),\
            data_pn_2.to(device),\
            target_pn_2.to(device),\
            pn_distance_matrix_1.to(device),\
            pn_distance_matrix_2.to(device),\
            pn_relation_matrix_1.to(device),\
            pn_relation_matrix_2.to(device),\
            pn_mask_matrix_1.to(device),\
            pn_mask_matrix_2.to(device)


def train(epoch):
    model.train()
    print('开始训练')
    total_loss = 0.
    start_time = time.time()
    nokens =len (vocab)
    embedding, embedding_times = {}, {}
    for batch, i in enumerate(range(0, train_data.size(0) - 1, batch_size)):
        if i<=train_data.size(0) - 1-batch_size:
            
            data, target, distance_matrix, relation_matrix, mask_matrix = get_batch(train_data, 
                                                                                    i, 
                                                                                    'tail', 
                                                                                    'train', 
                                                                                     train_dis_data,
                                                                                     train_relation_data,
                                                                                     train_mask)
            pn_data_1,pn_target_1,\
            pn_data_2,pn_target_2,\
            pn_distance_matrix_1,pn_distance_matrix_2,\
            pn_relation_matrix_1,pn_relation_matrix_2,\
            pn_mask_matrix_1, pn_mask_matrix_2 = get_pn_batch(train_pn_data, 
                                                              i, 
                                                              pn_label, 
                                                              train_pn_dis_data,
                                                              train_pn_relation_data,
                                                              train_pn_mask)
            
            optimizer.zero_grad()
            '''
            #只添加maskloss计算
            loss,_,embedding, embedding_times = model.forward(data,
                                                              target,
                                                              'tail',
                                                              None,
                                                              None,
                                                              None,
                                                              None,
                                                              None,
                                                              None,
                                                              None,
                                                              embedding,
                                                              embedding_times,
                                                              epoch)'''
            '''
            #添加maskloss以及becloss计算
            loss,_,embedding, embedding_times = model.forward(data,
                                                              target,
                                                              'tail',
                                                              pn_data_1,
                                                              pn_target_1,
                                                              pn_data_2,
                                                              pn_target_2,
                                                              None,
                                                              None,
                                                              None,
                                                              embedding,
                                                              embedding_times,
                                                              epoch)'''
            
            #添加maskloss以及bceloss计算，融入节点最短距离矩阵，融入关系矩阵
            loss,_,embedding, embedding_times = model.forward(data,
                                                              target,
                                                              'tail',
                                                              pn_data_1,
                                                              pn_target_1,
                                                              pn_data_2,
                                                              pn_target_2,
                                                              distance_matrix,
                                                              pn_distance_matrix_1,
                                                              pn_distance_matrix_2,
                                                              relation_matrix,
                                                              pn_relation_matrix_1,
                                                              pn_relation_matrix_2,
                                                              mask_matrix,
                                                              pn_mask_matrix_1, 
                                                              pn_mask_matrix_2,
                                                              embedding,
                                                              embedding_times,
                                                              epoch)
            #print(loss)
            loss.backward()
            optimizer.step()
            #ema.update()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            total_loss = loss.item()
            optimizer.step()
            #ema.update()
            
            
    '''#测试
    test_user_cou = {}
    recall10 = []
    hit_amount, test_amount = 0.0, 0.0
    hit10_user, test_user = {}, {}
    for index in range(len(test_pn_data)):
        user = test_pn_data[index][0].item()
        cou = test_pn_data[index][-1].item()
        if test_user_cou.get(user) is None:
            test_user_cou[user] = []
            hit10_user[user] = 0.0
            test_user[user] = 0.0
        test_user_cou[user].append(cou)
        
    amount = 0
    for (k,v) in test_user_cou.items():
        score = []
        if embedding.get(k) is None:
            amount += 1
            continue
        user_emb = embedding[k]/embedding_times[k]
        for cou in course:
            cou = vocab[cou]
            cou_emb = embedding[cou]/embedding_times[cou]
            score.append(np.dot(user_emb,cou_emb))
        rank_index = np.argsort(np.array(score))[::-1][:10]
        #if k == 100:
            #print(k, rank_index, score)
        for predict_index in rank_index:
            if vocab[course[predict_index]] in v:
                hit10_user[k] += 1
        
        test_user[k] = len(v)
        #print(k, hit10_user[k], len(v))
    
    for (user,hit) in hit10_user.items():
        if test_user[user] == 0.0:
            continue
        recall10.append(hit/test_user[user])
        hit_amount += hit
        test_amount += test_user[user]
    print("emb_Recall10 : {}, emb_Hit10 : {}".format(np.mean(recall10), hit_amount/test_amount),amount)
    end_time = time.time()
    print("cost time : {}".format(end_time-start_time))'''
    
    return total_loss


def kbc_batch_evaluation(model,data_source):
    model.eval()
    ntokens = len(vocab)
   # ema.apply_shadow()
   # ema.restore()
    total_num = data_source.size(0)
    total_h10 = 0
    recall10 = []
    hit_amount, test_amount = 0.0, 0.0
    hit10_user, test_user = {}, {}
    for i in range(0, data_source.size(0) - 1, batch_size):
        #print(i,data_source.size(0) - 1)
        if i<=data_source.size(0) - 1-batch_size:
            data, targets, _, _ = get_batch(data_source, i,'tail','eval',test_dis_data,test_relation_data)
            _, pridect, _, _ = model.forward(data,targets,flag='tail')
            targets = targets.view(batch_size).cpu().numpy().tolist()
            for j,item in enumerate(pridect):
                rank_10 = torch.topk(item,10).indices.cpu().numpy().tolist()
                
                if hit10_user.get(data[j][0].item()) is None:
                    hit10_user[data[j][0].item()] = 0
                if test_user.get(data[j][0].item()) is None:
                    test_user[data[j][0].item()] = 0
                    
                if targets[j] in rank_10:
                    total_h10=total_h10+1
                    hit10_user[data[j][0].item()] += 1
                test_user[data[j][0].item()] += 1
                    
    for (user,hit) in hit10_user.items():
        recall10.append(hit/test_user[user])
        hit_amount += hit
        test_amount += test_user[user]
    print("mask_Recall10 : {}, mask_Hit10 : {}".format(np.mean(recall10), hit_amount/test_amount))
    return  hit_amount/test_amount


def dcg_at_k(r, k, method=1):
    """Score is discounted cumulative gain (dcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Returns:
        Discounted cumulative gain
    """
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.


def ndcg_at_k(r, k, ground_truth, method=1):
    """Score is normalized discounted cumulative gain (ndcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Returns:
        Normalized discounted cumulative gain

        Low but correct defination
    """
    GT = set(ground_truth)
    if len(GT) > k :
        sent_list = [1.0] * k
    else:
        sent_list = [1.0]*len(GT) + [0.0]*(k-len(GT))
    dcg_max = dcg_at_k(sent_list, k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max


def kbc_batch_evaluation_full(model,data_source):
    model.eval()
    ntokens = len(vocab)
   # ema.apply_shadow()
   # ema.restore()
    total_num = data_source.size(0)
    total_h10 = 0
    recall1, recall2, recall5, recall10, recall20, recall50 = [], [], [], [], [], []
    precision1, precision2, precision5, precision10, precision20, precision50 = [], [], [], [], [], []
    ndcg1, ndcg2, ndcg5, ndcg10, ndcg20, ndcg50 = [], [], [], [], [], []
    rr1, rr2, rr5, rr10, rr20, rr50 = {}, {}, {}, {}, {}, {}
    hit_amount1, hit_amount2, hit_amount5, hit_amount10, hit_amount20, hit_amount50, test_amount = 0.0,0.0,0.0,0.0,0.0,0.0,0.0
    hit1_user, hit2_user, hit5_user, hit10_user, hit20_user, hit50_user, test_user, test_u_list = {}, {}, {}, {}, {}, {}, {}, {}
    for i in range(0, data_source.size(0) - 1, batch_size):
        #print(i,data_source.size(0) - 1)
        if i<=data_source.size(0) - 1-batch_size:
            data, targets, _, _ = get_batch(data_source, i,'tail','eval',test_dis_data,test_relation_data)
            strat_time_test = times.time() 
            _, pridect, _, _ = model.forward(data,targets,flag='tail')
            end_time_test = times.time()   
            print("test cost time : ", end_time_test-strat_time_test)

            targets = targets.view(batch_size).cpu().numpy().tolist()
            for j,item in enumerate(pridect):
                rank_1 = torch.topk(item,1).indices.cpu().numpy().tolist()
                rank_2 = torch.topk(item,2).indices.cpu().numpy().tolist()
                rank_5 = torch.topk(item,5).indices.cpu().numpy().tolist()
                rank_10 = torch.topk(item,10).indices.cpu().numpy().tolist()
                rank_20 = torch.topk(item,20).indices.cpu().numpy().tolist()
                rank_50 = torch.topk(item,50).indices.cpu().numpy().tolist()
                
                if hit10_user.get(data[j][0].item()) is None:
                    hit1_user[data[j][0].item()] = 0;rr1[data[j][0].item()] = []
                    hit2_user[data[j][0].item()] = 0;rr2[data[j][0].item()] = []
                    hit5_user[data[j][0].item()] = 0;rr5[data[j][0].item()] = []
                    hit10_user[data[j][0].item()] = 0;rr10[data[j][0].item()] = []
                    hit20_user[data[j][0].item()] = 0;rr20[data[j][0].item()] = []
                    hit50_user[data[j][0].item()] = 0;rr50[data[j][0].item()] = []
                if test_user.get(data[j][0].item()) is None:
                    test_user[data[j][0].item()] = 0;test_u_list[data[j][0].item()] = []
                    
                if targets[j] in rank_10:
                    total_h10=total_h10+1
                    hit10_user[data[j][0].item()] += 1
                    rr10[data[j][0].item()].append(1)
                else:
                    rr10[data[j][0].item()].append(0)
                    
                if targets[j] in rank_1:
                    hit1_user[data[j][0].item()] += 1
                    rr1[data[j][0].item()].append(1)
                else:
                    rr1[data[j][0].item()].append(0)
                    
                if targets[j] in rank_2:
                    hit2_user[data[j][0].item()] += 1
                    rr2[data[j][0].item()].append(1)
                else:
                    rr2[data[j][0].item()].append(0)
                    
                if targets[j] in rank_5:
                    hit5_user[data[j][0].item()] += 1
                    rr5[data[j][0].item()].append(1)
                else:
                    rr5[data[j][0].item()].append(0)
                    
                if targets[j] in rank_20:
                    hit20_user[data[j][0].item()] += 1
                    rr20[data[j][0].item()].append(1)
                else:
                    rr20[data[j][0].item()].append(0)
                    
                if targets[j] in rank_50:
                    hit50_user[data[j][0].item()] += 1
                    rr50[data[j][0].item()].append(1)
                else:
                    rr50[data[j][0].item()].append(0)
                    
                test_user[data[j][0].item()] += 1;test_u_list[data[j][0].item()].append(targets[j])
                    
    for (user,hit) in hit10_user.items():
        recall10.append(hit/test_user[user])
        precision10.append(hit/10.0)
        ndcg10.append(ndcg_at_k(rr10[user],10,test_u_list[user]))
        hit_amount10 += hit
        test_amount += test_user[user]
        
    for (user,hit) in hit1_user.items():
        recall1.append(hit/test_user[user])
        precision1.append(hit/1.0)
        ndcg1.append(ndcg_at_k(rr1[user],1,test_u_list[user]))
        hit_amount1 += hit
        
    for (user,hit) in hit2_user.items():
        recall2.append(hit/test_user[user])
        precision2.append(hit/2.0)
        ndcg2.append(ndcg_at_k(rr2[user],2,test_u_list[user]))
        hit_amount2 += hit
        
    for (user,hit) in hit5_user.items():
        recall5.append(hit/test_user[user])
        precision5.append(hit/5.0)
        ndcg5.append(ndcg_at_k(rr5[user],5,test_u_list[user]))
        hit_amount5 += hit
        
    for (user,hit) in hit20_user.items():
        recall20.append(hit/test_user[user])
        precision20.append(hit/20.0)
        ndcg20.append(ndcg_at_k(rr20[user],20,test_u_list[user]))
        hit_amount20 += hit
        
    for (user,hit) in hit50_user.items():
        recall50.append(hit/test_user[user])
        precision50.append(hit/50.0)
        ndcg50.append(ndcg_at_k(rr50[user],50,test_u_list[user]))
        hit_amount50 += hit
        
    
    print("mask_Recall : ",np.mean(recall1),
          np.mean(recall2),
          np.mean(recall5),
          np.mean(recall10),
          np.mean(recall20),
          np.mean(recall50))
    
    print("mask_Precision : ",np.mean(precision1),
          np.mean(precision2),
          np.mean(precision5),
          np.mean(precision10),
          np.mean(precision20),
          np.mean(precision50))
    print("mask_ndcg : ",np.mean(ndcg1),
          np.mean(ndcg2),
          np.mean(ndcg5),
          np.mean(ndcg10),
          np.mean(ndcg20),
          np.mean(ndcg50))
    print("mask_hit : ",np.mean(hit_amount1/test_amount),
          np.mean(hit_amount2/test_amount),
          np.mean(hit_amount5/test_amount),
          np.mean(hit_amount10/test_amount),
          np.mean(hit_amount20/test_amount),
          np.mean(hit_amount50/test_amount))
    
    print("mask_Hit10 : {}".format(hit_amount10/test_amount),'\n')
    return  hit_amount10/test_amount


last_h10 = 0

strat_time = times.time()       
for epoch   in  range(1,25):
    loss = train(epoch)

    print('epoch', epoch, '|loss', loss, '|lr', scheduler_warmup.get_lr()[0])
    scheduler_warmup.step()
    if epoch % 1 == 0:
        # print('验证集',train_data[:10240:4,:].shape)
        h10 = kbc_batch_evaluation(model, test_data)
        hit10 = kbc_batch_evaluation_full(model, test_data)
        if h10 > last_h10:
            best_model = model
            last_h10 = h10
end_time = times.time()   
print("all train cost time : ", end_time-strat_time)

torch.save(best_model, 'wn18_model_test_warmup60.pkl')














            
         



