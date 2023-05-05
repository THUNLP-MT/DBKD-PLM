import pdb
import pickle

import math
import scipy.integrate as integrate
from scipy.stats import norm
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
SCALE = 1
ERR_BOUND = 1e-3

class Decision2Prob:
    def __init__(self, debug=False, scale=0.1):
        self.counter = 0
        self.debug = debug
        self.B = [[1, 0, 0],
                  [0.5, math.sqrt(3)/2, 0],
                  [0.5, 0.5/math.sqrt(3), math.sqrt(2/3)]]
        for i in range(len(self.B)):
            for j in range(len(self.B[i])):
                self.B[i][j] *= (scale * math.sqrt(2))

    def fixed_point_solver(self, x0, p, round, err):
        counter = 0
        cur_err = 1e5
        x = x0
        while counter < round and cur_err > err:
            x_new = []
            orthant_p_list = []
            for i in range(len(x)):
                bias = [x[i]-x[j] for j in range(len(x)) if i != j]
                orthant_p = self.cal_orthant_prob(bias)
                x_new.append(p[i] - orthant_p + x[i])
                orthant_p_list.append(orthant_p)
            cur_err = 0
            for i in range(len(x)):
                cur_err += abs(x_new[i] - x[i])
            x = x_new
            counter += 1
        return x

    def f2(self, mu, x1, x2):
        low = (-mu[2] - self.B[2][0] * x1 - self.B[2][1] * x2) / self.B[2][2]
        return norm.sf(low)

    def f1(self,mu, x):
        low = (-mu[1] - self.B[1][0] * x) / self.B[1][1]
        high = np.inf

        def density_function1(t):
            return norm.pdf(t) * self.f2(mu, x, t)

        return integrate.quad(density_function1, low, high, epsabs=1e-4)[0]

    def cal_orthant_prob(self,mu):
        low = -mu[0] / self.B[0][0]
        high = np.inf

        def density_function0(x):
            return norm.pdf(x) * self.f1(mu, x)

        return integrate.quad(density_function0, low, high, epsabs=1e-4)[0]

class Decision2Prob2D:
    def __init__(self, debug=False, scale=0.1):
        self.counter = 0
        self.debug = debug
        self.B = math.sqrt(2) * scale

    def fixed_point_solver(self, x0, p, round, err):
        counter = 0
        cur_err = 1e5
        x = x0
        while counter < round and cur_err > err:
            x_new = []
            orthant_p_list = []
            for i in range(len(x)):
                bias = [x[i]-x[j] for j in range(len(x)) if i != j]
                orthant_p = self.cal_orthant_prob(bias)
                x_new.append(p[i] - orthant_p + x[i])
                orthant_p_list.append(orthant_p)
            cur_err = 0
            for i in range(len(x)):
                cur_err += abs(x_new[i] - x[i])
            x = x_new
            counter += 1
        return x

    def cal_orthant_prob(self,mu):
        low = -mu[0] / self.B
        return norm.sf(low)

class Decision2Prob3D:
    def __init__(self, debug=False, scale=0.1):
        self.counter = 0
        self.debug = debug
        self.B = [[1, 0],
                  [0.5, math.sqrt(3)/2]]
        for i in range(len(self.B)):
            for j in range(len(self.B[i])):
                self.B[i][j] *= (scale * math.sqrt(2))

    def fixed_point_solver(self, x0, p, round, err):
        counter = 0
        cur_err = 1e5
        x = x0
        while counter < round and cur_err > err:
            x_new = []
            orthant_p_list = []
            for i in range(len(x)):
                bias = [x[i]-x[j] for j in range(len(x)) if i != j]
                orthant_p = self.cal_orthant_prob(bias)
                x_new.append(p[i] - orthant_p + x[i])
                orthant_p_list.append(orthant_p)
            cur_err = 0
            for i in range(len(x)):
                cur_err += abs(x_new[i] - x[i])
            x = x_new
            counter += 1
        return x

    def f1(self,mu, x):
        low = (-mu[1] - self.B[1][0] * x) / self.B[1][1]
        return norm.sf(low)

    def cal_orthant_prob(self,mu):
        low = -mu[0] / self.B[0][0]
        high = np.inf

        def density_function0(x):
            return norm.pdf(x) * self.f1(mu, x)

        return integrate.quad(density_function0, low, high, epsabs=1e-4)[0]

def test(mu_diff, n=1000, scale=0.1):
    counter = 0
    mu = [0, 0-mu_diff[0], 0-mu_diff[1], 0-mu_diff[2]]
    for i in range(n):
        x = np.random.normal(mu, scale=scale)
        if x[0] > x[1] and x[0] > x[2] and x[0] > x[3]:
            counter += 1
    print(counter/n)

def test2D(mu_diff, n=1000, scale=0.1):
    counter = 0
    mu = [0, 0-mu_diff[0]]
    for i in range(n):
        x = np.random.normal(mu, scale=scale)
        if x[0] > x[1]:
            counter += 1
    print(counter/n)

def test3D(mu_diff, n=1000, scale=0.1):
    counter = 0
    mu = [0, 0-mu_diff[0], 0-mu_diff[1]]
    for i in range(n):
        x = np.random.normal(mu, scale=scale)
        if x[0] > x[1]:
            counter += 1
    print(counter/n)

def process_func(p_list):
    counter = Decision2Prob(debug=False, scale=SCALE)
    return counter.fixed_point_solver(x0=p_list, p=p_list, round=20, err=ERR_BOUND)

def process_func2D(p_list):
    counter = Decision2Prob2D(debug=False, scale=SCALE)
    return counter.fixed_point_solver(x0=p_list, p=p_list, round=20, err=ERR_BOUND)

def process_func3D(p_list):
    counter = Decision2Prob3D(debug=False, scale=SCALE)
    return counter.fixed_point_solver(x0=p_list, p=p_list, round=20, err=ERR_BOUND)

def make_decision2prob_table(mc_times=10, fpath="utils/ptable.pkl"):
    table = {}
    tot_num = 0
    tot_p_lists = []
    tot_c_lists = []
    for p0 in range(mc_times+1):
        for p1 in range(min(mc_times-p0+1, p0+1)):
            for p2 in range(min(mc_times-p0-p1+1, p1+1)):
                p3 = mc_times - p0 - p1 - p2
                if p3 <= p2:
                    tot_num += 1
                    tot_p_lists.append([p0/mc_times, p1/mc_times, p2/mc_times, p3/mc_times])
                    tot_c_lists.append([p0, p1, p2, p3])

    tmp_p_list = []
    results_list_all = []
    for i in tqdm(range(len(tot_p_lists))):
        tmp_p_list.append(tot_p_lists[i])
        if len(tmp_p_list) >= 50:
            with Pool(len(tmp_p_list)) as p:
                results_list = p.map(process_func, tot_p_lists)
            results_list_all.append(results_list)
            tmp_p_list = []
    with Pool(len(tmp_p_list)) as p:
        results_list = p.map(process_func, tot_p_lists)
    results_list_all.append(results_list)
    final_results_list = []
    for l in results_list_all:
        final_results_list.extend(l)
    for c_list, res_list in zip(tot_c_lists, final_results_list):
        key = ",".join([str(c) for c in c_list])
        table[key] = res_list

    # pdb.set_trace()
    pickle.dump(table, open(fpath, "wb"))

def make_decision2prob_table2D(mc_times=10, fpath="utils/ptable2D.pkl"):
    table = {}
    tot_num = 0
    tot_p_lists = []
    tot_c_lists = []
    for p0 in range(mc_times+1):
        p1 = mc_times - p0
        if p1 <= p0:
            tot_num += 1
            tot_p_lists.append([p0 / 10, p1 / 10])
            tot_c_lists.append([p0, p1])
    with Pool(tot_num) as p:
        results_list = p.map(process_func2D, tot_p_lists)
    for c_list, res_list in zip(tot_c_lists, results_list):
        key = ",".join([str(c) for c in c_list])
        table[key] = res_list
    pickle.dump(table, open(fpath, "wb"))

def make_decision2prob_table3D(mc_times=10, fpath="utils/ptable3D.pkl"):
    table = {}
    tot_num = 0
    tot_p_lists = []
    tot_c_lists = []
    for p0 in range(mc_times+1):
        for p1 in range(min(mc_times-p0+1, p0+1)):
            p2 = mc_times - p0 - p1
            if p2 <= p1:
                tot_num += 1
                tot_p_lists.append([p0/10, p1/10, p2/10])
                tot_c_lists.append([p0, p1, p2])
    with Pool(tot_num) as p:
        results_list = p.map(process_func3D, tot_p_lists)
    for c_list, res_list in zip(tot_c_lists, results_list):
        key = ",".join([str(c) for c in c_list])
        table[key] = res_list
    pickle.dump(table, open(fpath, "wb"))

if __name__ == "__main__":
    make_decision2prob_table(mc_times=10, fpath="utils/ptable_mc10.pkl")
