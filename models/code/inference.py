#!/usr/bin/env python
# coding=utf-8

__author__ = "Changchang.Yin"


import sys
# reload(sys)
# sys.setdefaultencoding('utf8')

import os
import sys
import time
import numpy as np
from sklearn import metrics
import random
import json
from collections import OrderedDict
from tqdm import tqdm


import torch
from torch.autograd import Variable
from torch.backends import cudnn
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from flask import Flask, jsonify


sys.path.append('../tools')
sys.path.append('models/tools')
import parse, py_op

sys.path.append('models/code')
import model
import data_loader


args = parse.args
if torch.cuda.is_available():
    args.gpu = 1
else:
    args.gpu = 0

def second_to_date(second):
    return time.localtime(second)

def _cuda(tensor):
    if args.gpu:
        return tensor.cuda()
    else:
        return tensor

def get_model(model_file, use_kg):
    dataset = data_loader.DataBowl(args, phase='valid')
    args.vocab = dataset.vocab
    args.relation = dataset.relation

    net, _ = model.FCModel(args, use_kg), model.Loss()
    net = _cuda(net)
    # return net
    try:
        net.load_state_dict(torch.load(model_file))
    except:
        # print(os.path.exists(model_file))
        d = torch.load(model_file, map_location=torch.device('cpu'))
        for k,v in d.items():
            d[k] = v.cpu()
            # print(k, type(v))
        net.load_state_dict(d)
    return net

def get_data():
    vocab_list = py_op.myreadjson(os.path.join(args.data_dir, args.dataset, args.dataset[:-4].lower() + 'vocab.json'))
    aid_year_dict = py_op.myreadjson(os.path.join(args.data_dir, args.dataset, 'aid_year_dict.json'))
    pid_aid_did_dict = py_op.myreadjson(os.path.join(args.data_dir, args.dataset, 'pid_aid_did_dict.json'))
    pid_demo_dict = py_op.myreadjson(os.path.join(args.data_dir, args.dataset, 'pid_demo_dict.json'))
    case_control_data = py_op.myreadjson(os.path.join(args.data_dir, args.dataset, 'case_control_data.json'))
    case_test = list(set(case_control_data['case_test'] + case_control_data['case_valid']))
    case_control_dict = case_control_data['case_control_dict']
    dataset = data_loader.DataBowl(args, phase='DGVis')
    id_name_dict = py_op.myreadjson(os.path.join(args.file_dir, 'id_name_dict.json'))

    # select patients with higher ration in knowledge graph
    selected_pid_set = set(pid_aid_did_dict)
    test_set = set()
    for case in case_test:
        test_set.add(case)
        for con in case_control_dict[case]:
            test_set.add(con)

    graph_dict = { 'edge': { }, 'node': { } }
    for line in open(os.path.join(args.file_dir, 'relation2id.txt')):
        data = line.strip().split()
        if len(data) == 2:
            relation, id = data[0], int(data[1])
            graph_dict['edge'][id] = relation
    for line in open(os.path.join(args.file_dir, 'entity2id.txt')):
        data = line.strip().split()
        if len(data) == 2:
            cui, id = data[0], int(data[1])
            graph_dict['node'][id] = cui

    if 1:
        selected_pid_set = set()
        graph = { 'nodes': { }, 'edges': [] }
        id_icd9_dict = py_op.myreadjson(os.path.join(args.file_dir, 'id_icd9_dict.json'))
        icd9_cui_dict = py_op.myreadjson(os.path.join(args.file_dir, 'icd9_cui_dict.json'))
        edge_dict = { }
        vocab_dict = py_op.myreadjson(os.path.join(args.file_dir, 'id_name_dict.json'))
        no_name = 0
        for line in open(os.path.join(args.file_dir, 'graph.txt')):
            data = line.strip().split('\t')
            node_f,node_s,relation_type = int(data[0]), int(data[1]), int(data[2])
            edge_dict[node_f] = edge_dict.get(node_f, []) + [[node_s, int(relation_type)]]

            # build graph
            cui_f = graph_dict['node'][node_f]
            cui_s = graph_dict['node'][node_s]
            if cui_f not in vocab_dict:
                vocab_dict[cui_f] = cui_f
                no_name += 1
            if cui_s not in vocab_dict:
                vocab_dict[cui_s] = cui_s
                no_name += 1

            relation = graph_dict['edge'][relation_type]
            # graph['nodes'][cui_f] = { 'id': cui_f, 'label': vocab_dict[cui_f], 'x': str(np.random.random() * 10), 'y': str(np.random.random() * 10)}
            # graph['nodes'][cui_s] = { 'id': cui_s, 'label': vocab_dict[cui_s], 'x': str(np.random.random() * 10), 'y': str(np.random.random() * 10)}
            graph['nodes'][cui_f] = { 'id': cui_f, 'label': vocab_dict[cui_f], 'x': int(np.random.random() * 10)+1, 'y': int(np.random.random() * 10)+1, 'size':4, 'color':'#2c3e50'}
            graph['nodes'][cui_s] = { 'id': cui_s, 'label': vocab_dict[cui_s], 'x': int(np.random.random() * 10) + 1, 'y': int(np.random.random() * 10) + 1, 'size': 4, 'color': '#2c3e50'}
            edge = { 'id': len(graph['edges']), 'source': cui_f, 'target': cui_s, 'relation': relation, 'type':'line', 'label': relation, 'size':float(0.5), 'color' :'rgba(128, 140, 141, 0.8)'}
            graph['edges'].append(edge)
        graph['nodes'] = list(graph['nodes'].values())
        print('no name', no_name)

        entity_id = dict()
        id_entity = []
        for line in open(os.path.join(args.file_dir, 'entity2id.txt')):
            data = line.strip().split()
            if len(data) == 2:
                cui, id = data[0], int(data[1])
                entity_id[cui] = id
                id_entity.append(cui)
        # ..
        in_kg_set = set()
        for id in vocab_list:
            if id in id_icd9_dict:
                icd = id_icd9_dict[id]
                if icd in icd9_cui_dict:
                    cui = icd9_cui_dict[icd]
                    if cui in entity_id:
                        cui = entity_id[cui]
                        if cui in edge_dict:
                            in_kg_set.add(id)
        print('There are :', len(in_kg_set), len(vocab_list))
        drug_list = [ "96017", "60394", "96079", "95717", "95804", "95698", "96258", "95511", "60314", "60392", "60362", "60361", "95666", "96103", "95929", "96341", "95521", "95972", "60320", "96266", "95803", "96472", "60111", "60157", "60111", "60110", "96045", "95666"]
        drug_set = set(drug_list)
        n_drug = 0
        n_in_event = 0
        n_out_event = 0
        ratio_list = []
        new_test_set = set()
        for pid in test_set:
            if pid not in pid_aid_did_dict:
                continue
            dp = pid_aid_did_dict[pid]
            n_pid_event = 0
            n_pid_event_in = 0
            for vs in dp.values():
                for fid in vs:
                    n_pid_event += 1
                    if fid in drug_set:
                        n_drug += 1
                    if fid in in_kg_set:
                        n_pid_event_in += 1
                        n_in_event += 1
                    else:
                        n_out_event += 1
            if 1.0 * n_pid_event_in / n_pid_event > 0.13:
                new_test_set.add(pid)
                ratio_list.append(1.0 * n_pid_event_in / n_pid_event)
        print('drug/in/out', n_drug, n_in_event, n_out_event)
        print('avg ratio', np.mean(ratio_list))
        print('case: ', len(new_test_set & set(case_test)))
        print('control: ', len(new_test_set) - len(new_test_set & set(case_test)))
        test_set = new_test_set





    aid_second_dict = py_op.myreadjson(os.path.join(args.data_dir, args.dataset, 'aid_second_dict.json'))

    for pid, aid_did_dict in pid_aid_did_dict.items():
        n = 0 
        aids = sorted(aid_did_dict.keys(), key=lambda aid:int(aid), reverse=True)
        for ia, aid in enumerate(aids):
            n += len(aid_did_dict[aid])
            if n > 120:
                pid_aid_did_dict[pid] = { aid: aid_did_dict[aid] for aid in aids[:ia] }
                break



    new_pid_demo_dict = dict()
    # pid_list = case_test + [c for case in case_test for c in case_control_dict[str(case)]]
    pid_list = list(test_set)
    pid_list = [str(pid) for pid in pid_list]
    for pid in pid_list:
        pid = str(pid)
        demo = pid_demo_dict[pid]
        gender = demo[0]
        yob = int(demo[2:])
        if pid not in pid_aid_did_dict:
            continue
        aids = pid_aid_did_dict[pid].keys()
        year = max([aid_year_dict[aid] for aid in aids])
        age = year - yob
        assert age < 120 and age > 0
        new_pid_demo_dict[pid] = [gender, age]


    # return data
    # case_control_dict = { case: [c for c in case_control_dict[case] if c in new_pid_demo_dict] for case in case_test if case in new_pid_demo_dict}
    case_set = list(set(test_set) & set(case_test))


    pid_demo_dict = new_pid_demo_dict
    pid_aid_did_dict = { pid: pid_aid_did_dict[pid] for pid in new_pid_demo_dict }

    return pid_demo_dict, pid_aid_did_dict, aid_second_dict, dataset, case_set, vocab_list, graph_dict, id_name_dict, graph


def get_pids_data(pids, pid_aid_did_dict, aid_second_dict, dataset, case_set):
    def _tensor(data):
        data = np.array(data)
        data = torch.from_numpy(data)
        data = Variable(_cuda(data))
        return data
    data_list, mask_list, label_list, visit_list  = [], [], [], []
    for pid in pids:
        aid_did_dict = pid_aid_did_dict[pid]
        aids = sorted(aid_did_dict.keys(), key=lambda aid: aid_second_dict[aid])
        max_date = aid_second_dict[aids[-1]]
        date_did_dict = dict()
        if pid in case_set:
            is_case = 1
        else:
            is_case = 0
        for aid in aids:
            delta_date = int((aid_second_dict[aid] - max_date) / 3600 / 24)
            date_did_dict[delta_date] = aid_did_dict[aid]
        data, label, mask, n_of_visit = dataset.__getitem__([date_did_dict, is_case])
        data_list.append(data.numpy())
        mask_list.append(mask.numpy())
        label_list.append(label.numpy())
        visit_list.append(n_of_visit.numpy())
    data_list = _tensor(data_list)
    mask_list = _tensor(mask_list)
    label_list = _tensor(label_list)
    visit_list = np.array(visit_list)
    return data_list, mask_list, label_list, visit_list

def get_att(relation_att, graph_dict):
    relation, weight = relation_att
    weight = list(np.array(weight).reshape(-1))
    assert len(weight) == len(relation)

    src_edge_tgt_w = []
    for rel, w in zip(relation, weight):
        src = graph_dict['node'][rel[0]]
        edge = graph_dict['edge'][rel[2]]
        tgt = graph_dict['node'][rel[1]]
        src_edge_tgt_w.append([src, edge, tgt, w])

    # print(src_edge_tgt_w[0])

    return src_edge_tgt_w

def inference(net, data, mask, label, n_of_visit, vocab_list, pids, pid_aid_did_dict, graph_dict):
    # print('input.size', data.size())
    output, contributions, output_vectors, fc_weight, graph_att_res, seq = net(data, mask, label)

    data = data.data.cpu().numpy()
    output_vectors = output_vectors
    output = output.data.cpu().numpy()

    label = label.data.cpu().numpy().reshape(-1)
    pred = (output > 0.5).reshape(-1)
    # print(sum(pred == label), len(pred))
    # n_correct, n_all =  sum(pred == label), len(pred)
    res =  sum(pred + label == 0), sum(label==0), sum(pred + label == 2), sum(label==1)

    pid_aid_did_cr_dict = { pid: { } for pid in pids }
    pid_aid_risk_dict = { pid: { } for pid in pids }
    pid_aid_did_att_dict = { pid: { } for pid in pids }
    for ipid, pid in enumerate(pids):
        indices = data[ipid]
        sindices = seq[ipid]
        crs = contributions[ipid]
        visits = n_of_visit[ipid]
        vectors = output_vectors[ipid]
        aids = sorted(pid_aid_did_dict[pid].keys(), key = lambda k:int(k))

        pid_graph_att_res = [gar[ipid] for gar in graph_att_res]


        i_visit = 1
        for iv, n in enumerate(visits):
            index = indices[iv]
            sindex = sindices[iv]
            # EHR data start when index > 0
            if index > 0:
                aid = aids[n - 1]
                # print(index)
                did = vocab_list[index - 1]
                sdid = vocab_list[sindex - 1]
                assert did == sdid
                cr = list(crs[iv*2: iv*2 + 2])

                if aid not in pid_aid_did_att_dict[pid]:
                    pid_aid_did_att_dict[pid][aid] = dict()
                if aid not in pid_aid_did_cr_dict[pid]:
                    pid_aid_did_cr_dict[pid][aid] = dict()
                pid_aid_did_cr_dict[pid][aid][did] = cr

                # calculate the contribution of a visit
                if i_visit != n:
                    i_visit = aid
                    last_aid = aids[n - 2]
                    vector = vectors[:, : iv*2 - 2].max(1) # the output vectors before current visit
                    risk = np.dot(vector, fc_weight)
                    pid_aid_risk_dict[pid][last_aid] = risk

                # attention weight
                if len(pid_graph_att_res[iv][0]):
                    pid_aid_did_att_dict[pid][aid][did] = get_att(pid_graph_att_res[iv], graph_dict)
                    # if iv == 12:
                    #     print(did, pid_graph_att_res[iv], get_att(pid_graph_att_res[iv], graph_dict))
                else:
                    pid_aid_did_att_dict[pid][aid][did] = []

        # last visit's risk
        pid_aid_risk_dict[pid][aids[-1]] = output[ipid][0]

    # print risk, whether the risk is correct
    for pid in pids:
        break
        # print(len(pid_aid_did_dict[pid]))
        # print(len(pid_aid_risk_dict[pid]))
        init_risk = 0
        out_risk = 0
        for aid in sorted(pid_aid_did_dict[pid], key=lambda a:int(a)):
            aid_risk = pid_aid_risk_dict[pid][aid]
            did_cr = pid_aid_did_cr_dict[pid][aid]
            sum_risk = 0
            for did in did_cr:
                sum_risk += sum(did_cr[did])
            print('{:3.4f}, {:3.4f}'.format(aid_risk , sum_risk))
            # print(aid_risk)
            # print(type(aid_risk))
            init_risk = aid_risk
            out_risk += sum_risk
        # print(pid_aid_risk_dict[pid][aid], out_risk)
        # print()


        # analyze graph attentin result
        # pid_aid_did_att_dict[pid]

    # print("output sisk:", pid_aid_risk_dict[pid][aids[-1]])

    return pid_aid_did_cr_dict, pid_aid_risk_dict, pid_aid_did_att_dict, res


class DGRNN(object):
    def __init__(self):

        # build model
        model_file = os.path.join(args.result_dir, '{:s}-kg-gp.ckpt'.format(args.dataset.lower().split('_')[0]))
        self.net = get_model(model_file, 1)
        model_file = os.path.join(args.result_dir, '{:s}-no-kg.ckpt'.format(args.dataset.lower().split('_')[0]))
        self.net_nokg = get_model(model_file, 0)

        # prepare all the test data
        test_data = get_data()
        self.pid_demo_dict, self.pid_aid_did_dict, self.aid_second_dict, self.dataset, self.case_set, \
                self.vocab_list, self.graph_dict, self.id_name_dict, self.graph = test_data
        # assert len(self.pid_demo_dict) == len(self.pid_aid_did_dict)
        self.pids = list(self.pid_demo_dict.keys())
        self.icd_name_dict = py_op.myreadjson(os.path.join(args.file_dir, 'icd_name_dict.json'))

    # def test(self):
    def get_test_data(self):
        '''
        return all the data needed for visualization:
            pid_aid_did_dict: 
                pid: patient id
                aid: admission id
                did: diagnosis id
            aid_date_dict:
                aid: admission id
                date: int, admission's time
            vocab_dict: icd9 -> diagnosis  dict
        '''
        aid_date_dict = { aid: second_to_date(second) for aid, second in self.aid_second_dict.items() }
        vocab_list = []
        for vocab in self.vocab_list:
            if vocab in self.id_name_dict:
                vocab_list.append(self.id_name_dict[vocab] )
            else:
                if vocab in self.icd_name_dict:
                    vocab_list.append(self.icd_name_dict[vocab])
                else:
                    vocab = vocab.strip('0')
                    if vocab in self.icd_name_dict:
                        vocab_list.append(self.icd_name_dict[vocab])
                    else:
                        vocab = vocab[:-1]
                        try:
                            vocab_list.append(self.icd_name_dict[vocab.strip('0')])
                        except:
                            # vocab_list.append(self.icd_name_dict[vocab.strip('0')])
                            vocab_list.append(vocab)
                        assert len(vocab) >= 3
        # vocab_dict = { k:v for k,v in zip(self.vocab_list, vocab_list) }
        vocab_dict = py_op.myreadjson(os.path.join(args.file_dir, 'id_name_dict.json'))
        # py_op.mywritejson(os.path.join(args.file_dir, 'graph.json'), self.graph)
        try:
            return jsonify(self.pid_aid_did_dict, aid_date_dict, vocab_dict, self.graph)
        except:
            return self.pid_aid_did_dict, aid_date_dict, vocab_dict, self.graph

    # def test(self, pid_aid_did_dict = { }):
    def predict(self, pid_aid_did_dict):
        '''
        predict the risk for given patients
        intput: 
            pid_aid_did_dict
                pid: patient id
                aid: admission id
                did: diagnosis id
        output:
            pid_aid_did_cr_dict:
            pid_aid_risk_dict:
            pid_aid_did_att_dict:
        '''

        # pid_aid_did_dict = { k:v for k,v in list(self.pid_aid_did_dict.items())[:10] }

        pid_aid_did_cr_dict, pid_aid_risk_dict, pid_aid_did_att_dict, pid_aid_risk_dict_nokg = { }, { }, { }, { }
        results = [[], []]
        # for pid in tqdm(pid_aid_did_dict):
        for pid in pid_aid_did_dict:
            # pids_batch = pid_aid_did_dict.keys()
            pids_batch = [pid]
            data, mask, label, n_of_visit = get_pids_data(pids_batch, pid_aid_did_dict, \
                self.aid_second_dict, self.dataset, self.case_set) 
            cr_dict, risk_dict, att_dict, res_kg = inference(self.net, data, \
                mask, label, n_of_visit, self.vocab_list, pids_batch, pid_aid_did_dict, self.graph_dict)
            _, risk_dict_nokg, _, res_nokg = inference(self.net_nokg, data, \
                mask, label, n_of_visit, self.vocab_list, pids_batch, pid_aid_did_dict, self.graph_dict)
            results[0].append(res_kg)
            results[1].append(res_nokg)
            pid_aid_did_cr_dict.update(cr_dict)
            pid_aid_risk_dict.update(risk_dict)
            pid_aid_risk_dict_nokg.update(risk_dict_nokg)
            pid_aid_did_att_dict.update(att_dict)
        results = np.array(results)
        # print('kg', results[0].mean(0))
        # print('nokg', results[1].mean(0))

        for pid, aids in pid_aid_risk_dict.items():
            for aid, risk in aids.items():
                pid_aid_risk_dict[pid][aid] = str(risk)

        for pid, aids in pid_aid_risk_dict_nokg.items():
            for aid, risk in aids.items():
                pid_aid_risk_dict_nokg[pid][aid] = str(risk)

        for pid, aids in pid_aid_did_att_dict.items():
            for aid, dids in aids.items():
                for did, att in dids.items():
                    pid_aid_did_att_dict[pid][aid][did] = [str(x) for x in att]

        for pid, aids in pid_aid_did_cr_dict.items():
            for aid, dids in aids.items():
                for did, cr in dids.items():
                    if len(pid_aid_did_att_dict[pid][aid][did]) == 0:
                        cr = [sum(cr), 0]
                    pid_aid_did_cr_dict[pid][aid][did] = [str(x) for x in cr]

        # return jsonify(pid_aid_did_cr_dict, pid_aid_risk_dict, pid_aid_did_att_dict, pid_aid_risk_dict_nokg )
        display_patients_with_helpful_kg = 0
        if display_patients_with_helpful_kg:
            assert len(pid_aid_risk_dict) == 1
            label = label.data.cpu().numpy().reshape(-1)[0]
            for pid, aids in pid_aid_risk_dict.items():
                aid = sorted(aids, key=lambda a:int(a))[-1]
                risk_kg = float(pid_aid_risk_dict[pid][aid])
                risk_nokg = float(pid_aid_risk_dict_nokg[pid][aid])

                if label>0.5 and risk_kg>0.43  and  risk_nokg<0.38:
                    crs = [pid_aid_did_cr_dict[pid][aid][did] for aid, dids in pid_aid_did_cr_dict[pid].items() for did in dids]
                    crs = [[float(cr[0]), float(cr[1])] for cr in crs]
                    crs = np.array(crs)
                    cr_sum = crs.sum()
                    cr_kg = max(crs.sum(0)[1], crs.max(0)[1])
                    kg_rate = cr_kg / (label - 0.5) 
                    # print(pid, risk_kg, risk_nokg, label, kg_rate)
                    if kg_rate > 0.1:
                        print(pid, risk_kg, risk_nokg, label, kg_rate)
                        pass


        try:
            return jsonify(pid_aid_did_cr_dict, pid_aid_risk_dict, pid_aid_did_att_dict, pid_aid_risk_dict_nokg )
        except:
            return pid_aid_did_cr_dict, pid_aid_risk_dict, pid_aid_did_att_dict, pid_aid_risk_dict_nokg

    def generate_csv(self):

        print(self.case_set)

        pids = list(self.pid_aid_did_dict.keys())
        vectors, outputs = [], []
        args.batch_size = 1
        for i in tqdm(range(0, len(pids), args.batch_size)):
            pids_batch = pids[i: i+args.batch_size]
            pid_aid_did_dict = { pid: self.pid_aid_did_dict[pid] for pid in pids_batch }
            data, mask, label, n_of_visit = get_pids_data(pids_batch, pid_aid_did_dict, \
                self.aid_second_dict, self.dataset, self.case_set) 
            output, _, output_vectors, _, _ = self.net(data, mask, label)
            vectors.append(output_vectors.max(2)) 
            outputs.append(output.data.numpy())
            # if i > 50:
            #     break

        vectors = np.concatenate(vectors, 0)
        outputs = np.array(outputs).reshape(-1)
        labels = [pid in self.case_set for pid in pids]
        risks = sorted(outputs, reverse=True)
        print(len(labels))
        print(sum(labels))
        print(int(1.0 * sum(labels) / len(labels) * len(outputs)))
        threshold = risks[int(1.0 * sum(labels) / len(labels) * len(outputs))]
        # assert len(pids) == len(vectors)

        # use tsne
        from sklearn.manifold import TSNE
        X_tsne = TSNE(n_components=2,learning_rate=100).fit_transform(vectors)

        wdemo = open(os.path.join(args.result_dir, args.dataset.lower().split('_')[0] + '_demo.csv'), 'w')
        wdemo.write('PID,GENDER,AGE,X,Y,RISK,THRESHOLD,PREDICT,LABEL\n')
        for pid, xy, output in zip(pids, vectors, outputs):
            if pid in self.case_set:
                label = 1
            else:
                label = 0
            demo = self.pid_demo_dict[pid]
            wdemo.write(pid + ',')
            wdemo.write(demo[0]+ ',')
            wdemo.write(str(demo[1])+ ',')
            wdemo.write(str(xy[0])+ ',')
            wdemo.write(str(xy[1])+ ',')
            wdemo.write(str(output)+ ',')
            wdemo.write(str(threshold)+ ',')
            wdemo.write(str(int(output > threshold))+ ',')
            wdemo.write(str(label)+ '\n')
            print(int(int(output > threshold) == label))
            





def main():
    dr = DGRNN()
    dr.get_test_data()
    # return
    # dr.generate_csv()
    # return
    # return
    # print(len(dr.pids))
    for pid in ['480006629']:
    # for pid in dr.pids:
        pids = [pid]
        pid_aid_did_dict = { k:dr.pid_aid_did_dict[k] for k in pids }
        data = dr.predict(pid_aid_did_dict)
    return

    # print(dr.get_test_data()[-1])
    print('start prediction')
    print(len(data))
    return
    for d in data:
        print(d)
        print()
        print()

if __name__ == '__main__':
    main()
