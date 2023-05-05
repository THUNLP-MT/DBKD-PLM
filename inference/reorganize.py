import pdb
import pickle
import sys
import torch
from tqdm import tqdm
NOISE_NUM = 10
from scipy.stats import norm
from operator import itemgetter

scale = 1
mynorm = norm(loc=0, scale=scale)

def reverse_gaussian(tensor, smooth=1e-5):
    tensor = smooth + tensor
    tmp = tensor / (tensor + torch.min(tensor))
    tmp = torch.tensor(mynorm.ppf(tmp))
    return torch.softmax(tmp, dim=-1)

idx = torch.load(sys.argv[1])
logits = torch.load(sys.argv[2])
counter = 0
if len(sys.argv) > 4:
    strategy = sys.argv[4]
    if strategy == "noise-old":
        new_logits = torch.zeros(int(logits.shape[0] / NOISE_NUM), logits.shape[1])
        label_num = new_logits.shape[1]
        id = 0
        tmp_id = 0
        for lg in logits:
            tmp_id += 1
            if tmp_id > NOISE_NUM:
                tmp_id = 1
                id += 1
            new_logits[id][lg.argmax()] += 1/NOISE_NUM
    elif strategy == "noise-mc":
        new_logits = torch.zeros(logits.shape[0], logits.shape[-1])
        ptable = pickle.load(open(f"utils/ptable_mc{str(NOISE_NUM)}.pkl", "rb"))
        new_decisions = [[0] * logits.shape[-1] for i in range(logits.shape[0])]
        label_num = new_logits.shape[-1]
        id = 0
        tmp_id = 0
        for id, lg_list in enumerate(logits):
            for lg in lg_list:
                new_decisions[id][lg.argmax()] += 1
        for i in range(len(new_decisions)):
            decision = new_decisions[i]
            indices, decision = zip(*sorted(enumerate(decision), key=itemgetter(1), reverse=True))
            decision = ",".join([str(i) for i in decision])
            if str(NOISE_NUM) in decision:
                counter += 1
            if decision not in ptable:
                pdb.set_trace()
            prob = ptable[decision]
            for j in range(len(new_logits[i])):
                new_logits[i][indices[j]] = prob[j]
    elif strategy == "noise":
        new_logits = torch.zeros(int(logits.shape[0] / NOISE_NUM), logits.shape[1])
        if logits.shape[1] == 4:
            ptable = pickle.load(open("utils/ptable.pkl", "rb"))
            # ptable = pickle.load(open("utils/ptable_mc20.pkl", "rb"))
        elif logits.shape[1] == 3:
            ptable = pickle.load(open("utils/ptable3D.pkl", "rb"))
        elif logits.shape[1] == 2:
            ptable = pickle.load(open("utils/ptable2D.pkl", "rb"))
        else:
            assert False
        new_decisions = [[0] * logits.shape[1] for i in range(int(logits.shape[0] / NOISE_NUM))]
        label_num = new_logits.shape[1]
        id = 0
        tmp_id = 0
        for lg in logits:
            tmp_id += 1
            if tmp_id > NOISE_NUM:
                tmp_id = 1
                id += 1
            new_decisions[id][lg.argmax()] += 1
        for i in range(len(new_decisions)):
            decision = new_decisions[i]
            indices, decision = zip(*sorted(enumerate(decision), key=itemgetter(1), reverse=True))
            decision = ",".join([str(i) for i in decision])
            if "10" in decision:
                counter += 1
            if decision not in ptable:
                pdb.set_trace()
            prob = ptable[decision]
            for j in range(len(new_logits[i])):
                new_logits[i][indices[j]] = prob[j]
    elif strategy == "standard" or strategy == "surrogate":
        new_logits = torch.zeros(int(logits.shape[0]), logits.shape[1])
        for id, lg in enumerate(logits):
            new_logits[id] = lg
    elif strategy == "ab-emp":
        new_logits = torch.zeros(int(logits.shape[0]), logits.shape[1])
        label_num = new_logits.shape[1]
        if label_num == 4:
            ptable = pickle.load(open("utils/ptable.pkl", "rb"))
        elif label_num == 2:
            ptable = pickle.load(open("utils/ptable2D.pkl", "rb"))
        elif label_num == 3:
            ptable = pickle.load(open("utils/ptable3D.pkl", "rb"))
        else:
            assert False
        default_lg = ptable['1'+",".join(['0']*label_num)]
        for id, lg in enumerate(logits):
            # pdb.set_trace()
            for lb_idx in range(label_num):
                new_logits[id][lb_idx] = default_lg[1]
            new_logits[id][lg.argmax().item()] = default_lg[0]
    else:
        assert False
else:
    new_logits = torch.zeros(int(logits.shape[0]), logits.shape[1])
    for id, lg in zip(idx, logits):
        new_logits[id] = lg
torch.save(new_logits, sys.argv[3])
print(counter/len(new_logits))