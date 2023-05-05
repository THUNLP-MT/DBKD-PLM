import pdb
import pickle
import sys
import torch
import random
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
predicts = torch.load(sys.argv[2])
counter = 0
if len(sys.argv) > 4:
    strategy = sys.argv[4]
    if strategy == "noise":
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        choice_idx = [tokenizer.encode(choice)[0] for choice in [' A', ' B', ' C', ' D']]
        new_logits = torch.zeros(int(len(predicts) / NOISE_NUM), 4)
        ptable = pickle.load(open("utils/ptable.pkl", "rb"))
        new_decisions = [[0] * new_logits.shape[1] for i in range(new_logits.shape[0])]
        label_num = new_logits.shape[1]
        id = 0
        tmp_id = 0
        for pred in predicts:
            tmp_id += 1
            if tmp_id > NOISE_NUM:
                tmp_id = 1
                id += 1
            if pred in choice_idx:
                new_decisions[id][choice_idx.index(pred)] += 1
            else:
                new_decisions[id][random.randint(0, 3)] += 1
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
    elif strategy == "standard":
        new_logits = torch.zeros(int(len(predicts)), 4)
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        choice_idx = [tokenizer.encode(choice)[0] for choice in [' A', ' B', ' C', ' D']]
        for id, pred in enumerate(predicts):
            if pred in choice_idx:
                new_logits[id][choice_idx.index(pred)] = 1
            else:
                new_logits[id][random.randint(0, 3)] = 1
    else:
        assert False
else:
    assert False
torch.save(new_logits, sys.argv[3])
print(counter/len(new_logits))