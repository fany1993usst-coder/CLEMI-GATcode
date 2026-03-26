import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from nltk.corpus import stopwords
import nltk
from nltk.wsd import lesk
from nltk.corpus import wordnet as wn
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
import re
import collections
from collections import Counter
import numpy as np
from multiprocessing import Process, Queue
import pandas as pd
import os
import random
import torch
import pickle


def split_validation(train_set, valid_portion):
    train_set_x, train_set_y = train_set
    n_samples = len(train_set_x)
    sidx = np.arange(n_samples, dtype='int32')
    np.random.shuffle(sidx)
    n_train = int(np.round(n_samples * (1. - valid_portion)))
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]

    return (train_set_x, train_set_y), (valid_set_x, valid_set_y)

def translation(data, item_dic):

    datax = []
    for i in range(len(data[0])):
        datax.append([item_dic[s] for s in data[0][i]])
    datay = [item_dic[s] for s in data[1]]

    return (datax, datay)

#对局部图的数据进行处理
def handle_data(inputData, train_len=None):
    len_data = [len(nowData) for nowData in inputData]
    if train_len is None:
        max_len = max(len_data)
    else:
        max_len = train_len
    # reverse the sequence
    us_pois = [list(reversed(upois)) + [0] * (max_len - le) if le < max_len else list(reversed(upois[-max_len:]))
               for upois, le in zip(inputData, len_data)]
    us_msks = [[1] * le + [0] * (max_len - le) if le < max_len else [1] * max_len
               for le in len_data]
    return us_pois, us_msks, max_len

class Data():
    def __init__(self, data, window, train_len=None):
        inputs = data[0]
        self.inputs = np.asarray(inputs) 
        self.targets = np.asarray(data[1])
        self.length = len(inputs)
        self.window = window
        self.train_len = train_len

        # LDA相关数据的处理
        self.LDA = True
        item_set = set()
        for session in self.inputs:
            for item in session:
                item_set.add(item)
        vocab_dic = {}
        for i in item_set:
            vocab_dic[i] = len(vocab_dic) + 1  # vocab_dic记录item所在的位置索引

        keywords_dic = {}
        if self.LDA:
            keywords_dic_original = pickle.load(open('datasets/yoochoose1_64/_LDA.p', "rb"))
            for i in keywords_dic_original:
                if int(i) in vocab_dic:
                    keywords_dic[vocab_dic[int(i)]] = keywords_dic_original[i]

        self.keywords_dic = keywords_dic




    def generate_batch(self, batch_size, shuffle = False):
        if shuffle:
            shuffled_arg = np.arange(self.length)
            np.random.shuffle(shuffled_arg)
            self.inputs = self.inputs[shuffled_arg]
            self.targets = self.targets[shuffled_arg]
        n_batch = int(self.length / batch_size)
        if self.length % batch_size != 0:
            n_batch += 1
        slices = np.split(np.arange(n_batch * batch_size), n_batch)
        slices[-1] = slices[-1][:(self.length - batch_size * (n_batch - 1))]
        return slices

    def get_slice(self, iList):
        inputs, targets = self.inputs[iList], self.targets[iList]
        #关于超图的构建
        items, n_node, H, HT, G, EG, alias_inputs, node_masks, node_dic = [], [], [], [], [], [], [], [], []
        num_edge, edge_mask, edge_inputs = [], [], []
        node_sum = [] #求非去重的节点数量

        for u_input in inputs:
            temp_s = u_input

            # temp_l = list(set(temp_s)) #这里把去重的步骤去掉了，方便与局部图的维度对齐
            temp_l = list(set(temp_s))
            temp_dic = {temp_l[i]: i for i in range(len(temp_l))}
            n_node.append(temp_l)

            alias_inputs.append([temp_dic[i] for i in temp_s])
            node_dic.append(temp_dic)


            min_s = min(self.window, len(u_input))
            # min_s = len(u_input)
            num_edge.append(int((1 + min_s) * len(u_input) - (1 + min_s) * min_s / 2))


        max_n_node = np.max([len(i) for i in n_node])
        # max_n_node = 199
        # num_edge = [len(i) for i in inputs]

        if self.LDA:
            num_edge = [i + 50  for i in num_edge]

        max_n_edge = max(num_edge)

        max_se_len = max([len(i) for i in alias_inputs])

        edge_mask = [[1] * len(le) + [0] * (max_n_edge - len(le)) for le in alias_inputs]

        for idx in range(len(inputs)):
            u_input = inputs[idx]
            effect_len = len(alias_inputs[idx])
            node = n_node[idx]
            items.append(node + (max_n_node - len(node)) * [0])

            effect_list = alias_inputs[idx]
            ws = np.ones(max_n_edge)
            cols = []
            rows = []
            edg = []
            e_idx = 0

            #1+ min(self.window, effect_len - 1)
            for w in range(1+ min(self.window, effect_len - 1)):
                edge_idx = list(np.arange(e_idx, e_idx + effect_len-w))
                edg += edge_idx
                for ww in range(w + 1):
                    rows += effect_list[ww:ww+effect_len-w]
                    cols += edge_idx

                e_idx += len(edge_idx)

            if len(cols) == 0:
                s = 0
            else:
                s = max(cols) + 1

            if self.LDA:
                for i in node:
                    if i in self.keywords_dic:
                        temp = self.keywords_dic[i]

                        rows += [node_dic[idx][i]] * len(temp)
                        cols += [topic + s for topic in temp]


            u_H = sp.coo_matrix(([1.0]*len(rows), (rows, cols)), shape=(max_n_node, max_n_edge))
            HT.append(np.asarray(u_H.T.todense()))


            node_masks.append((max_se_len - len(alias_inputs[idx])) * [0] + [1]*len(alias_inputs[idx]))
            alias_inputs[idx] = (max_se_len - len(alias_inputs[idx])) * [0] + alias_inputs[idx]


            edge_inputs.append(edg + (max_n_edge - len(edg))*[0])

        #关于局部图的构建

        Adj_local, Alias_inputs_local, Items_local = [], [], []

        inputs_local, mask_local, max_len = handle_data(inputs, self.train_len)
        for index in range(len(inputs_local)):
            #表示单个的会话数据
            u_input = inputs_local[index]
            # max_n_node = 39
            #x3/10修改处
            max_n_node = max_len
            node = np.unique(u_input)
            items_local = node.tolist() + (max_n_node - len(node)) * [0]
            adj = np.zeros((max_n_node, max_n_node))
            #3/9修改
            for i in np.arange(len(u_input) - 1):
                u = np.where(node == u_input[i])[0][0]
                adj[u][u] = 1
                if u_input[i + 1] == 0:
                    break
                v = np.where(node == u_input[i + 1])[0][0]
                adj[v][v] = 1
                adj[u][v] = 1
            # for i in np.arange(len(u_input) - 1):
            #     u = np.where(node == u_input[i])[0][0]
            #     adj[u][u] = 1
            #     if u_input[i + 1] == 0:
            #         break
            #     v = np.where(node == u_input[i + 1])[0][0]
            #     if u == v or adj[u][v] == 4:
            #         continue
            #     adj[v][v] = 1
            #     if adj[v][u] == 2:
            #         adj[u][v] = 4
            #         adj[v][u] = 4
            #     else:
            #         adj[u][v] = 2
            #         adj[v][u] = 3
            alias_inputs_local = [np.where(node == i)[0][0] for i in u_input]


            #收集需要返回的值
            Adj_local.append(adj)
            Alias_inputs_local.append(alias_inputs_local)
            Items_local.append(items_local)

        #将需要返回的值转为tensor类型
        Adj_local = torch.tensor(Adj_local)
        Alias_inputs_local = torch.tensor(Alias_inputs_local)
        Items_local = torch.tensor(Items_local)
        mask_local = torch.tensor(mask_local)

#修改处
        return alias_inputs, H, HT, G, EG, items, targets, node_masks, edge_mask, edge_inputs, Adj_local, Alias_inputs_local, Items_local, mask_local#adj-local 100,39,39 inputs-local 100,24  mask local100*24 Items_local 100,39

