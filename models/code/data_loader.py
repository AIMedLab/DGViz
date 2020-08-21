#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2017 www.drcubic.com, Inc. All Rights Reserved
#
"""
File: data.py
Author: shileicao(shileicao@stu.xjtu.edu.cn)
Date: 2017-06-20 14:56:54

**Note.** This code absorb some code from following source.
1. [DSB2017](https://github.com/lfz/DSB2017)
"""

import json
import collections
import os
import random
import time
import warnings
from copy import deepcopy

import numpy as np
import torch
from torch.utils.data import Dataset

import sys
sys.path.append('../tools')
import py_op

class DataBowl(Dataset):
    def __init__(self, args, phase='train'):
        assert (phase == 'train' or phase == 'valid' or phase == 'test', phase == 'DGVis')
        self.args = args
        self.phase = phase
        self.vocab = json.load(open(os.path.join(args.data_dir, args.dataset, args.dataset.lower().replace('json','') + 'vocab.json')))
        if phase == 'DGVis':
            self.inputs = json.load(open(os.path.join(args.data_dir, args.dataset, 'test.json')))
        else:
            self.inputs = json.load(open(os.path.join(args.data_dir, args.dataset, phase + '.json')))
        if phase == 'train':
            group_drug = np.load(os.path.join(args.data_dir, args.dataset, 'group_drug.npy'))
            self.index_to_group = dict()
            groups = set(group_drug[0])
            print('Groups: ', groups)
            self.group_to_index = { g: { 'with_drug': [], 'without_drug': [] } for g in groups }
            self.indices_without_drug = []
            for index in range(len(group_drug[0])):
                group, drug = group_drug[:, index]
                self.index_to_group[index] = group
                if drug == 0:
                    self.group_to_index[group]['without_drug'].append(index)
                    self.indices_without_drug.append(index)
                else:
                    self.group_to_index[group]['with_drug'].append(index)
        self.seq = args.seq_length
        # print 'seq lenght', args.seq_length

        self.id_icd9_dict = py_op.myreadjson(os.path.join(args.file_dir, 'id_icd9_dict.json'))
        self.icd9_cui_dict = py_op.myreadjson(os.path.join(args.file_dir, 'icd9_cui_dict.json'))

        self.entity_id = dict()
        self.id_entity = []
        for line in open(os.path.join(args.file_dir, 'entity2id.txt')):
            data = line.strip().split()
            if len(data) == 2:
                cui, id = data[0], int(data[1])
                self.entity_id[cui] = id
                self.id_entity.append(cui)

        # infomation for knowledge graph
        cui_set = set()
        for id in self.vocab:
            id = str(id)
            if id in self.id_icd9_dict:
                icd9 = self.id_icd9_dict[id]
                if icd9 in self.icd9_cui_dict:
                    cui = self.icd9_cui_dict[icd9]
                    cui_set.add(cui)
        # print 'start', len(cui_set)

        relation_set = set()
        for line in open(os.path.join(args.file_dir, 'graph.txt')):
            node_f,node_s,relation = line.strip().split('\t')
            node_f = self.id_entity[int(node_f)]
            node_s = self.id_entity[int(node_s)]
            cui_set.add(node_f)
            cui_set.add(node_s)
            relation_set.add(relation)
        self.vocab = ['null'] + self.vocab + list(set(cui_set) - set(self.vocab))
        # self.vocab = ['null'] + sorted(set(self.vocab) | cui_set)
        # self.vocab_index = { w:i for i,w in enumerate(self.vocab) }
        self.vocab_index = { }
        for i,w in enumerate(self.vocab):
            if w not in self.vocab_index:
                self.vocab_index[w] = i
        self.relation = sorted(relation_set)



        # stati number of events in knowledge graph
        if args.phase == 'DGVis':
            self.id_icd9_dict = py_op.myreadjson(os.path.join(args.file_dir, 'id_icd9_dict.json'))
            self.icd9_cui_dict = py_op.myreadjson(os.path.join(args.file_dir, 'icd9_cui_dict.json'))
            self.edge_dict = { }
            for line in open(os.path.join(args.file_dir, 'graph.txt')):
                data = line.strip().split('\t')
                node_f,node_s,relation_type = int(data[0]), int(data[1]), int(data[2])
                self.edge_dict[node_f] = self.edge_dict.get(node_f, []) + [[node_s, int(relation_type)]]
            # ..
            in_kg_set = set()
            for id in self.vocab:
                if id in self.id_icd9_dict:
                    icd = self.id_icd9_dict[id]
                    if icd in self.icd9_cui_dict:
                        cui = self.icd9_cui_dict[icd]
                        if cui in self.entity_id:
                            cui = self.entity_id[cui]
                            if cui in self.edge_dict:
                                in_kg_set.add(id)
            print('There are :', len(in_kg_set), len(self.vocab))
            # ..
            n_in_event = 0
            n_out_event = 0
            for pid_data in self.inputs:
                dp, lp = pid_data
                for vs in dp.values():
                    for fid in vs:
                        if fid in in_kg_set:
                            n_in_event += 1
                        else:
                            n_out_event += 1
            print('in/out', n_in_event, n_out_event)


    def series(self, data, r=0.9):

        keys = sorted(data.keys(), key=lambda k:int(k))
        data = [data[k] for k in keys]

        new_data = []
        n_of_visit = []
        for i, day_data in enumerate(data):
            new_data = new_data + day_data
            n_of_visit = n_of_visit + [i+1 for _ in day_data]

        data = new_data

        return data, n_of_visit

    def augment(self, data, r=0.9):
        if np.random.random() < r:
            new_data = []
            base_r = 0.7
            days = range(len(data))
            for i in sorted(days[: int(1 + (base_r + np.random.random() * (1 - base_r))* len(days))]):
                new_data.append(data[i])
            data = new_data
        return data


    def __getitem__(self, idx, split=None):
        if self.phase == 'train':
            idx_withou_drug = self.indices_without_drug[idx]
            data_without_drug = self.get_an_item(idx_withou_drug, split)

            group = self.index_to_group[idx_withou_drug]
            indices_with_drug = self.group_to_index[group]['with_drug']

            # assign a patient with drug
            indices = list(range(len(indices_with_drug)))
            np.random.shuffle(indices)
            idx = (idx + indices[0]) % len(indices)
            idx_with_drug = indices_with_drug[idx]

            data_with_drug = self.get_an_item(idx_with_drug, split)

            data = [torch.stack((a,b), 0) for a,b in zip(data_without_drug, data_with_drug)]
            return data


        else:
            return self.get_an_item(idx, split)

    def get_an_item(self, idx, split):
        if self.phase != 'DGVis':
            # in training process
            data, label = self.inputs[idx]
        else:
            # in inference process
            data, label = idx
        data, n_of_visit = self.series(data)
        index = []
        for d in data:
            idx = self.vocab_index[str(d)]
            # if idx > 1000:
            #     print(d, idx)


            if 0 and self.args.use_kg and str(d) in self.id_icd9_dict:
                icd9 = self.id_icd9_dict[str(d)]
                if icd9 in self.icd9_cui_dict:
                    cui = self.icd9_cui_dict[icd9]
                    if cui in self.vocab_index:
                        idx = self.vocab_index[cui]

            index.append(idx)


        if self.phase != 'DGVis':
            if len(index) <= self.seq:
                index = [0 for _ in range(self.seq)] + index
                n_of_visit = [0 for _ in range(self.seq)] + n_of_visit

            '''
            elif self.phase == 'train':
                margin = range(len(index) - self.seq)
                np.random.shuffle(margin)
                start = margin[0]
                index = index[start: start + self.seq]
                n_of_visit = n_of_visit[start: start + self.seq]
            '''

            index = index[- self.seq:]
            n_of_visit = n_of_visit[- self.seq:]
        else:
            index = [0] + index
            n_of_visit = [0] + n_of_visit

        data = np.array(index).astype(np.int64)
        label = np.array([label]).astype(np.float32)
        n_of_visit = np.array(n_of_visit).astype(np.int32)

        mask = data > 0
        mask = mask.astype(np.float32)

        if self.phase == 'DGVis':
            return torch.from_numpy(data), torch.from_numpy(label), torch.from_numpy(mask), torch.from_numpy(n_of_visit)
        else:
            return torch.from_numpy(data), torch.from_numpy(label), torch.from_numpy(mask)

    def __len__(self):
        if self.phase == 'train':
            return len(self.indices_without_drug)
        else:
            return len(self.inputs)


def augment(sample,
            target,
            bboxes,
            coord,
            ifflip=True,
            ifrotate=True,
            ifswap=True):
    return sample, target, bboxes, coord




def collate(batch):
    if torch.is_tensor(batch[0]):
        return [b.unsqueeze(0) for b in batch]
    elif isinstance(batch[0], np.ndarray):
        return batch
    elif isinstance(batch[0], int):
        return torch.LongTensor(batch)
    elif isinstance(batch[0], collections.Iterable):
        transposed = zip(*batch)
        return [collate(samples) for samples in transposed]
