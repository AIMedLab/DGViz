
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
import random
import json
from collections import OrderedDict
from tqdm import tqdm

sys.path.append('../tools')
sys.path.append('models/tools')
import parse, py_op
args = parse.args

drug_dict = py_op.myreadjson(os.path.join(args.file_dir, 'drug_dict.json'))
drug_set = set(drug_dict)

def find_drug(patient_dict):
    keys = sorted([int(key) for key in patient_dict.keys()])
    keys = [str(k) for k in keys]
    new_patient_dict = { }
    for k in keys:
        visit_data = patient_dict[k]
        if len(set(visit_data) & drug_set) > 0:
            return 1, new_patient_dict
        else:
            new_patient_dict[k] = patient_dict[k]
    return 0, patient_dict


def stat_drug_effect():
    for fi in ['train.json', 'valid.json', 'test.json']:
        ehr_data = json.load(open(os.path.join(args.data_dir, args.dataset, fi))) 
        new_ehr_data = []
        has_drug = []
        has_hf = [[], []]
        for pdata in ehr_data:
            patient_dict = pdata[0]
            hf = pdata[1]
            vis, new_patient_dict = find_drug(patient_dict)
            if len(new_patient_dict):
                has_drug.append(vis)
                has_hf[vis].append(hf)
                new_ehr_data.append([new_patient_dict, hf])
        print('')
        print('In {:s}:'.format(fi.split('.')[0]))
        print('There are {:d} patients. {:d} patients has drug. {:d} patients hasn\'t drugs.'.format(len(new_ehr_data), sum(has_drug), len(ehr_data) - sum(has_drug)))
        print('Drug patients with hf: {:3.4f}. No drug patients with hf: {:3.4}.'.format(np.mean(has_hf[1]), np.mean(has_hf[0])))
        py_op.mywritejson(os.path.join(args.data_dir, args.dataset, 'new_'+fi), new_ehr_data)

def find_similar_patients():
    '''
    use 
    '''
    def map_data_to_vector(pdata):
        patient_dict, y = pdata
        x = np.zeros(len(ehr_vocab))
        for k,vs in patient_dict.items():
            for v in vs:
                x[vocab_dict[v]] = 1
        return x,y
    ehr_vocab = json.load(open(os.path.join(args.data_dir, args.dataset, 'ehr_vocab.json')))
    vocab_dict = { v:i for i,v in enumerate(ehr_vocab) }
    lr_train = json.load(open(os.path.join(args.data_dir, args.dataset, 'new_valid.json'))) + \
            json.load(open(os.path.join(args.data_dir, args.dataset, 'new_test.json'))) + \
            json.load(open(os.path.join(args.data_dir, args.dataset, 'new_train.json'))) 
    xs = []
    ys = []
    for pdata in lr_train:
        x,y = map_data_to_vector(pdata)
        xs.append(x)
        ys.append(y)
    xs = np.array(xs)
    ys = np.array(ys)
    print(xs.shape)
    print(ys.shape)

    print('\nStart to train LR...')
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(random_state=0).fit(xs, ys)
    ps = clf.predict_proba(xs)[:, 1]

    from sklearn import metrics
    print('\nStart to compute AUC...')
    fpr, tpr, threshholds = metrics.roc_curve(ys, ps)
    auc = metrics.auc(fpr, tpr)
    print('AUC: {:3.4f}'.format(auc))

    print('LR score: {:3.4f}'.format(clf.score(xs, ys)))

    print('\nStart to compute patients\' risks before drug usage...')
    risks = []
    train_data = json.load(open(os.path.join(args.data_dir, args.dataset, 'train.json'))) 
    risks, ys, has_drug = [], [], []
    for pdata in train_data:
        patient_dict, y = pdata
        drug_vis, new_patient_dict = find_drug(patient_dict)
        if len(new_patient_dict):
            x,y = map_data_to_vector([new_patient_dict, y])
            risk = clf.predict_proba(np.array([x]))[0, 1]
            risks.append(risk)
        else:
            risks.append(-1)
        ys.append(y)
        has_drug.append(drug_vis)
    risks = np.array(risks)
    ys = np.array(ys)
    has_drug = np.array(has_drug)
    risks[risks < 0] = risks[risks>0].mean()
    fpr, tpr, threshholds = metrics.roc_curve(ys, risks)
    auc = metrics.auc(fpr, tpr)
    print('Train AUC: {:3.4f}'.format(auc))


    print('Start to group train patients with risks...')
    srisks = sorted(risks)
    n_group = 4
    risk_split = [srisks[int(i * len(risks) / n_group)] for i in range(n_group)]
    groups = np.zeros(len(risks), dtype=np.int64)
    for ir,r in enumerate(risks):
        for ig, rg in enumerate(risk_split):
            if r < rg:
                break
        groups[ir] = ig
    print(groups.min())
    print(groups.max())
    print(groups.mean())


    print('\nStart to compute drug effect in each group')
    for g in range(groups.min(), groups.max()+1):
        g_has_drug = has_drug[groups==g]
        g_ys = ys[groups==g]
        gnd_ys = g_ys[g_has_drug==0]
        gd_ys = g_ys[g_has_drug==1]
        print('Group {:d}: HF rate: {:3.4f} / {:4d} patients with drug.   HF rate: {:3.4f} / {:4d} patients without drug.'.format(g, np.mean(gd_ys), len(gd_ys), np.mean(gnd_ys), len(gnd_ys)))

    # py_op.mywritejson(
    group_drug = np.array([groups, has_drug])
    print(group_drug.shape)
    np.save(os.path.join(args.data_dir, args.dataset, 'group_drug.npy'), group_drug)









def main():
    # stat_drug_effect()
    find_similar_patients()

if __name__ == '__main__':
    main()
