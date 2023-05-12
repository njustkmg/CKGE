from __future__ import absolute_import #3
from __future__ import division
from __future__ import print_function

import six
import json
import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
#from torch.nn import TransformerEncoder,TransformerEncoderLayer
from transformer1 import Encoder
class CoKEModel(nn.Module):

    def __init__(self,voc_size,emb_size,nhead,nhid,nlayers,dropout,soft_label,batch_size):
        super(CoKEModel, self).__init__()

        #print("emb_size:",emb_size,"nhead:",nhead,"nhid:",nhid,"nlayers:",nlayers,"dropout:",dropout)
        self._emb_size = emb_size
        self._n_layer = nlayers
        self._n_head =nhead
        self._voc_size = voc_size
        self._dropout = dropout
        self._batch_size = batch_size
        self._nhid = nhid
        self._soft_label = soft_label
        self.model_type = 'CoKE'
        self._position_ids =torch.tensor([[0,1,2]])
        self._device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
        for time in range(int(np.log2(self._batch_size))):
            self._position_ids = torch.cat((self._position_ids,self._position_ids),0)
        self._position_ids = self._position_ids.to(self._device)
        # encoder_layers = TransformerEncoderLayer(d_model =self._emb_size ,nhead=self._n_head,dim_feedforward=self._nhid,dropout = self._dropout,activation='gelu')
        # #encoder_layers.wight.data.
        # self.transformer_encoder = TransformerEncoder(encoder_layers,num_layers=self._n_layer)
        self.transformer_encoder = Encoder(num_layers=self._n_layer,num_heads=self._n_head,dropout=self._dropout,ffn_dim=self._nhid)

        self.ele_encoder = nn.Embedding(num_embeddings =self._voc_size,embedding_dim = self._emb_size)
        self.pos_encoder = nn.Embedding(num_embeddings= 3,embedding_dim= self._emb_size)
        
        #节点距离编码
        self.spatial_pos_encoder = nn.Embedding(20, self._n_head,padding_idx=0)#去除padding_idx=0项
        self.graph_token_virtual_distance = nn.Embedding(1, self._n_head)
        #节点关系编码
        self.relation_encoder = nn.Embedding(30, self._n_head, padding_idx=0)
        self.relation_weight_encoder = nn.Embedding(self._batch_size*self._n_head*self._n_head, 1, padding_idx=0)
        self.relation_weight = nn.Linear(self._emb_size, self._n_head)
        
        #归一化矩阵
        self.pre_scale = torch.nn.Parameter(torch.FloatTensor(self._batch_size,3,self._emb_size), requires_grad=True)
        self.pre_bias = torch.nn.Parameter(torch.FloatTensor(self._batch_size,3,self._emb_size),requires_grad=True)
        self.post_scale = torch.nn.Parameter(torch.FloatTensor(self._batch_size,  self._emb_size), requires_grad=True)
        self.post_bias = torch.nn.Parameter(torch.FloatTensor(self._batch_size,  self._emb_size), requires_grad=True)

        self.dropoutl = torch.nn.Dropout(p = self._dropout)
        self.FC1 = nn.Linear(self._emb_size, self._emb_size)
        self.FC2 = nn.Linear(self._emb_size,5389)
        self.gelu = nn.GELU()
        self.init_weights()
        self.criterion = LabelSmoothingCELoss(self._voc_size,self._soft_label)
        self.bceloss = nn.BCELoss()
        self.alpha = 0.4#0.01
        print("emb_size:",emb_size,"nhead:",nhead,"nhid:",nhid,"nlayers:",nlayers,"dropout:",dropout,"self.alpha:",self.alpha)


    def layer_norm(self,src):
        begin_norm_axis = len(src.shape)-1
        #begin_norm_axis = 1
        #print('coke1',begin_norm_axis)
        mean = torch.mean(src,dim=begin_norm_axis,keepdim=True)
        shift_x = src - mean
        variance = torch.mean(shift_x*shift_x,dim=begin_norm_axis,keepdim=True)
        r_stdev = torch.sqrt(1/(variance+1e-12))

        norm_x = shift_x*r_stdev
        return norm_x


    #初始化函数，之后考虑要不要改。
    def init_weights(self):
        initrange = 0.02
        self.ele_encoder.weight.data.normal_(0, 0.02)
        self.pos_encoder.weight.data.normal_(0, 0.02)
        self.spatial_pos_encoder.weight.data.normal_(0, 0.02)
        self.graph_token_virtual_distance.weight.data.normal_(0, 0.02)
        self.relation_encoder.weight.data.normal_(0, 0.02)
        self.FC2.bias.data.zero_()
        self.FC1.bias.data.zero_()
        #self.transformer_encoder.weight.data.normal_(0, 0.02)
        self.FC1.weight.data.normal_(0, 0.02)
        self.FC2.weight.data.normal_(0, 0.02)

        self.pre_scale.data.fill_(1.)
        self.pre_bias.data.fill_(0.)
        self.post_scale.data.fill_(1.)
        self.post_bias.data.fill_(0.)




    def forward(self,
                src,
                target,
                flag=None,
                pn_data_1=None,
                pn_target_1=None,
                pn_data_2=None,
                pn_target_2=None,
                distance_matrix=None,
                pn_distance_matrix_1=None,
                pn_distance_matrix_2=None,
                relation_matrix=None,
                pn_relation_matrix_1=None,
                pn_relation_matrix_2=None,
                mask_matrix=None,
                pn_mask_matrix_1=None, 
                pn_mask_matrix_2=None,
                embedding=None,
                embedding_times=None,
                epoch=None):

        #user_id = src[:,0].detach().cpu().numpy()
        #cou_id = target[:,-1].detach().cpu().numpy()
        
        src = self.ele_encoder(src)
        #src_pos = self.pos_encoder(self._position_ids)
        #src = src + src_pos
        #print('transformer前',src.shape)
        src = self.layer_norm(src)
        src = self.dropoutl(src)
        
        if distance_matrix != None:
            #计算空间距离矩阵
            # [n_graph, n_node, n_node, n_head] -> [n_graph, n_head, n_node, n_node]
            spatial_pos_bias = self.spatial_pos_encoder(distance_matrix).permute(0, 3, 1, 2)
            # reset spatial pos here
            t = self.graph_token_virtual_distance.weight.view(1, self._n_head, 1)
            spatial_pos_bias[:, :, 1:, 0] = spatial_pos_bias[:, :, 1:, 0] + t
            spatial_pos_bias[:, :, 0, :] = spatial_pos_bias[:, :, 0, :] + t
            spatial_pos_bias = self.layer_norm(spatial_pos_bias)
            spatial_pos_bias = self.dropoutl(spatial_pos_bias)
            
            #计算关系矩阵
            # [n_graph, n_node, n_node, n_head] -> [n_graph, n_head, n_node, n_node]
            relation_bias = self.relation_encoder(relation_matrix)#.permute(0, 3, 1, 2)
            batch = relation_bias.size(0); node = relation_bias.size(1)
            relation_bias = relation_bias.reshape(batch, -1, self._n_head)
            relation_bias = torch.bmm(relation_bias,self.relation_weight_encoder.weight.reshape(batch, -1, self._n_head))
            relation_bias = relation_bias.reshape(batch,node,node,-1).permute(0, 3, 1, 2)
            
            relation_bias = self.layer_norm(relation_bias)
            relation_bias = self.dropoutl(relation_bias)
            
            path_1_mask = mask_matrix.clone();path_2_mask = mask_matrix.clone()
            path_1_mask[:,1:4,:] = 0;path_1_mask[:,:,6:11] = 0;path_1_mask[:,:,1:4] = 0;path_1_mask[:,6:11,:] = 0
            path_2_mask[:,1:6,:] = 0;path_2_mask[:,:,8:11] = 0;path_2_mask[:,:,1:6] = 0;path_2_mask[:,8:11,:] = 0
            
            output,_ = self.transformer_encoder(src, [spatial_pos_bias,relation_bias,mask_matrix])
            output_path_1,_ = self.transformer_encoder(src, [spatial_pos_bias,relation_bias,path_1_mask])
            output_path_2,_ = self.transformer_encoder(src, [spatial_pos_bias,relation_bias,path_2_mask])
        else:
            output,_ = self.transformer_encoder(src)
        
        if flag == 'tail':
            pridect = output[:,11:12,:]
        else:
            pridect = output[:,0:1,:]

        #print('目标',pridect.shape)

        #pridect = self.FC1(pridect)
        #print('第一层FC',pridect.shape)

        pridect = pridect.contiguous().view(self._batch_size, self._emb_size)
        pridect = self.FC1(pridect)
        pridect = self.gelu(pridect)
        pridect = self.layer_norm(pridect)
        #print('归一化后',pridect.shape)
        pridect = pridect * self.post_scale
        pridect = pridect + self.post_bias

        '''user = output[:,0:1,:]
        cou = output[:,7:8,:]
        
        user = user.contiguous().view(self._batch_size, self._emb_size)
        user = self.FC1(user)
        user = self.gelu(user)
        user = self.layer_norm(user)
        user = user * self.post_scale
        user = user + self.post_bias

        cou = cou.contiguous().view(self._batch_size, self._emb_size)
        cou = self.FC1(cou)
        cou = self.gelu(cou)
        cou = self.layer_norm(cou)
        cou = cou * self.post_scale
        cou = cou + self.post_bias
            
        #记录训练过程中得到的embedding
        if embedding != None:
            for index in range(len(src)):
                if embedding.get(user_id[index]) is None:
                    embedding[user_id[index]] = np.zeros(self._emb_size)
                    embedding_times[user_id[index]] = 0
                if embedding.get(cou_id[index]+16538) is None:
                    embedding[cou_id[index]+16538] = np.zeros(self._emb_size)
                    embedding_times[cou_id[index]+16538] = 0
                embedding[user_id[index]] += user[index].detach().cpu().numpy()
                embedding_times[user_id[index]] += 1
                embedding[cou_id[index]+16538] += cou[index].detach().cpu().numpy()
                embedding_times[cou_id[index]+16538] += 1'''
        
        #print('FC1后', pridect)
        #pridect = self.FC2(pridect)
        ##print('FC后', pridect)
        #pridect = self.gelu(pridect)
        pridect = self.FC2(pridect)
        #print('后',pridect.shape)
        target = target.view(self._batch_size)
        #loss = self.criterion(pridect,target.long())
        
        loss = 0.0
        if distance_matrix != None:
            #mask除path以外的信息进行预测
            pridect_path_1 = output_path_1[:,11:12,:]
            pridect_path_1 = pridect_path_1.contiguous().view(self._batch_size, self._emb_size)
            pridect_path_1=self.layer_norm(self.gelu(self.FC1(pridect_path_1)))
            pridect_path_1 = pridect_path_1 * self.post_scale
            pridect_path_1 = pridect_path_1 + self.post_bias
            pridect_path_1 = self.FC2(pridect_path_1)
            loss_path_1 = self.criterion(pridect_path_1,target.long())

            pridect_path_2 = output_path_2[:,11:12,:]
            pridect_path_2 = pridect_path_2.contiguous().view(self._batch_size, self._emb_size)
            pridect_path_2=self.layer_norm(self.gelu(self.FC1(pridect_path_2)))
            pridect_path_2 = pridect_path_2 * self.post_scale
            pridect_path_2 = pridect_path_2 + self.post_bias
            pridect_path_2 = self.FC2(pridect_path_2)
            loss_path_2 = self.criterion(pridect_path_2,target.long())
            loss = loss + loss_path_1 + loss_path_2
        
        if pn_data_1 != None and pn_target_1 != None:
            user_id = pn_data_1[:,0].detach().cpu().numpy()
            cou_id = pn_data_1[:,-1].detach().cpu().numpy()
            
            pn_data_1 = self.ele_encoder(pn_data_1)
            pn_data_1 = self.layer_norm(pn_data_1)
            pn_data_1 = self.dropoutl(pn_data_1)
            
            if pn_distance_matrix_1 != None:
                #计算空间距离矩阵
                # [n_graph, n_node, n_node, n_head] -> [n_graph, n_head, n_node, n_node]
                spatial_pos_bias = self.spatial_pos_encoder(pn_distance_matrix_1).permute(0, 3, 1, 2)
                # reset spatial pos here
                t = self.graph_token_virtual_distance.weight.view(1, self._n_head, 1)
                spatial_pos_bias[:, :, 1:, 0] = spatial_pos_bias[:, :, 1:, 0] + t
                spatial_pos_bias[:, :, 0, :] = spatial_pos_bias[:, :, 0, :] + t
                spatial_pos_bias = self.layer_norm(spatial_pos_bias)
                spatial_pos_bias = self.dropoutl(spatial_pos_bias)
                
                #计算关系矩阵
                # [n_graph, n_node, n_node, n_head] -> [n_graph, n_head, n_node, n_node]
                relation_bias = self.relation_encoder(pn_relation_matrix_1)#.permute(0, 3, 1, 2)
                batch = relation_bias.size(0); node = relation_bias.size(1)
                relation_bias = relation_bias.reshape(batch, -1, self._n_head)
                relation_bias = torch.bmm(relation_bias,self.relation_weight_encoder.weight.reshape(batch, -1, self._n_head))
                relation_bias = relation_bias.reshape(batch,node,node,-1).permute(0, 3, 1, 2)
                
                relation_bias = self.layer_norm(relation_bias)
                relation_bias = self.dropoutl(relation_bias)
            
                output,_ = self.transformer_encoder(pn_data_1, [spatial_pos_bias,relation_bias,pn_mask_matrix_1])
            else:
                output,_ = self.transformer_encoder(pn_data_1)
               
            user = output[:,0:1,:]
            cou = output[:,11:12,:]

            user = user.contiguous().view(self._batch_size, self._emb_size)
            user = self.FC1(user)
            user = self.gelu(user)
            user = self.layer_norm(user)
            user = user * self.post_scale
            user = user + self.post_bias

            cou = cou.contiguous().view(self._batch_size, self._emb_size)
            cou = self.FC1(cou)
            cou = self.gelu(cou)
            cou = self.layer_norm(cou)
            cou = cou * self.post_scale
            cou = cou + self.post_bias

            score = torch.sum(torch.mul(user,cou),axis=-1)
            score = nn.Sigmoid()(score)
            bceloss = self.bceloss(score,pn_target_1)
            loss = loss+self.alpha*bceloss
            
            #记录训练过程中得到的embedding
            for index in range(len(pn_data_1)):
                if embedding.get(user_id[index]) is None:
                    embedding[user_id[index]] = np.zeros(self._emb_size)
                    embedding_times[user_id[index]] = 0
                if embedding.get(cou_id[index]) is None:
                    embedding[cou_id[index]] = np.zeros(self._emb_size)
                    embedding_times[cou_id[index]] = 0
                embedding[user_id[index]] += user[index].detach().cpu().numpy()
                embedding_times[user_id[index]] += 1
                embedding[cou_id[index]] += cou[index].detach().cpu().numpy()
                embedding_times[cou_id[index]] += 1
            
        if pn_data_2 != None and pn_target_2 != None:
            user_id = pn_data_2[:,0].detach().cpu().numpy()
            cou_id = pn_data_2[:,-1].detach().cpu().numpy()
            
            pn_data_2 = self.ele_encoder(pn_data_2)
            pn_data_2 = self.layer_norm(pn_data_2)
            pn_data_2 = self.dropoutl(pn_data_2)
            
            if pn_distance_matrix_2 != None:
                #计算空间距离矩阵
                # [n_graph, n_node, n_node, n_head] -> [n_graph, n_head, n_node, n_node]
                spatial_pos_bias = self.spatial_pos_encoder(pn_distance_matrix_2).permute(0, 3, 1, 2)
                # reset spatial pos here
                t = self.graph_token_virtual_distance.weight.view(1, self._n_head, 1)
                spatial_pos_bias[:, :, 1:, 0] = spatial_pos_bias[:, :, 1:, 0] + t
                spatial_pos_bias[:, :, 0, :] = spatial_pos_bias[:, :, 0, :] + t
                spatial_pos_bias = self.layer_norm(spatial_pos_bias)
                spatial_pos_bias = self.dropoutl(spatial_pos_bias)
                
                #计算关系矩阵
                # [n_graph, n_node, n_node, n_head] -> [n_graph, n_head, n_node, n_node]
                relation_bias = self.relation_encoder(pn_relation_matrix_2)#.permute(0, 3, 1, 2)
                batch = relation_bias.size(0); node = relation_bias.size(1)
                relation_bias = relation_bias.reshape(batch, -1, self._n_head)
                relation_bias = torch.bmm(relation_bias,self.relation_weight_encoder.weight.reshape(batch, -1, self._n_head))
                relation_bias = relation_bias.reshape(batch,node,node,-1).permute(0, 3, 1, 2)
                
                relation_bias = self.layer_norm(relation_bias)
                relation_bias = self.dropoutl(relation_bias)
            
                output,_ = self.transformer_encoder(pn_data_2, [spatial_pos_bias,relation_bias,pn_mask_matrix_2])
            else:
                output,_ = self.transformer_encoder(pn_data_2)
                
            user = output[:,0:1,:]
            cou = output[:,11:12,:]

            user = user.contiguous().view(self._batch_size, self._emb_size)
            user = self.FC1(user)
            user = self.gelu(user)
            user = self.layer_norm(user)
            user = user * self.post_scale
            user = user + self.post_bias

            cou = cou.contiguous().view(self._batch_size, self._emb_size)
            cou = self.FC1(cou)
            cou = self.gelu(cou)
            cou = self.layer_norm(cou)
            cou = cou * self.post_scale
            cou = cou + self.post_bias

            score = torch.sum(torch.mul(user,cou),axis=-1)
            score = nn.Sigmoid()(score)
            bceloss = self.bceloss(score,pn_target_2)
            loss = loss+self.alpha*bceloss
            
            #记录训练过程中得到的embedding
            for index in range(len(pn_data_2)):
                if embedding.get(user_id[index]) is None:
                    embedding[user_id[index]] = np.zeros(self._emb_size)
                    embedding_times[user_id[index]] = 0
                if embedding.get(cou_id[index]) is None:
                    embedding[cou_id[index]] = np.zeros(self._emb_size)
                    embedding_times[cou_id[index]] = 0
                embedding[user_id[index]] += user[index].detach().cpu().numpy()
                embedding_times[user_id[index]] += 1
                embedding[cou_id[index]] += cou[index].detach().cpu().numpy()
                embedding_times[cou_id[index]] += 1
            

        return loss,pridect,embedding, embedding_times




class LabelSmoothingCELoss(nn.Module):
    def __init__(self, classes, smoothing=0.1, dim=-1):
    # smoothing  标签平滑的百分比
        super(LabelSmoothingCELoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim
        self._device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        batch_size = pred.shape[0]
        true_dist = pred.data.clone()
        entity_indicator = torch.zeros(batch_size,5389).to(self._device)
        #entity_indicator.fill_(self.smoothing / (self.cls - 1-18)).to(self._device)
        #relation_indicator= torch.zeros(batch_size,18).to(self._device)
        #print(pred.shape, true_dist.shape, entity_indicator.shape, relation_indicator.shape, batch_size, self.cls - 18)
        true_dist = entity_indicator.to(self._device)
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))









