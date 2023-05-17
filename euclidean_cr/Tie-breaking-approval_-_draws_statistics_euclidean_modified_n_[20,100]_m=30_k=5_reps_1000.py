# In[1]:
import math

import abcvoting
import mapel

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import collections
import itertools

import os
import datetime


# In[2]:


SEED = 47
random.seed(SEED)
np.random.seed(SEED)


# In[3]:


ns = list(range(20, 100+1, 1))
m = 30
k = 5
reps = 1000
MAX_NUM_OF_COMMITTEES = 2


# In[4]:


# one model supported
models = [{"model_id": 'euclidean_cr', "params_family": {"radius": [0.5*k/m/1.87, k/m/1.87, 2*k/m/1.87], "dim": [1], "space": ["uniform"]}}]
# models = [{"model_id": 'euclidean', "params_family": {"p": [0.5*k/m, k/m], "dim": [1, 2]}}]


# In[5]:


# rules = ['av', 'sav', 'cc', 'seqcc', 'pav', 'seqpav', 'slav', 'seqslav', 'geom2', 'seqphragmen', 'rule-x-phase-1', 'rule-x']
# rules = ['av', 'sav', 'cc', 'pav', 'seqpav', 'slav', 'seqslav', 'geom2', 'seqphragmen', 'rule-x-phase-1', 'rule-x']
rules = ['av', 'sav', 'cc', 'pav', 'seqpav', 'slav', 'seqslav', 'seqphragmen', 'rule-x-phase-1', 'rule-x']


# In[ ]:


# AUXILIARY
def random_ball(dimension, num_points=1, radius=1):
    random_directions = np.random.normal(size=(dimension, num_points))
    random_directions /= np.linalg.norm(random_directions, axis=0)
    random_radii = np.random.random(num_points) ** (1 / dimension)
    x = radius * (random_directions * random_radii).T
    return x


def random_sphere(dimension, num_points=1, radius=1):
    random_directions = np.random.normal(size=(dimension, num_points))
    random_directions /= np.linalg.norm(random_directions, axis=0)
    random_radii = 1.
    return radius * (random_directions * random_radii).T


def get_rand(model: str, cat: str = "voters") -> list:
    """ generate random values"""
    # print(model ==  "1d_uniform")

    point = [0]
    if model in {"1d_uniform",  "1d_interval"}:
        return np.random.rand()
    elif model in {'1d_asymmetric'}:
        if np.random.rand() < 0.3:
            return np.random.normal(loc=0.25, scale=0.15, size=1)
        else:
            return np.random.normal(loc=0.75, scale=0.15, size=1)
    elif model in {"1d_gaussian"}:
        point = np.random.normal(0.5, 0.15)
        while point > 1 or point < 0:
            point = np.random.normal(0.5, 0.15)
    elif model == "1d_one_sided_triangle":
        point = np.random.uniform(0, 1) ** 0.5
    elif model == "1d_full_triangle":
        point = np.random.choice([np.random.uniform(0, 1) ** 0.5, 2 - np.random.uniform(0, 1) ** 0.5])
    elif model == "1d_two_party":
        point = np.random.choice([np.random.uniform(0, 1), np.random.uniform(2, 3)])
    elif model in {"2d_disc", "2d_range_disc"}:
        phi = 2.0 * 180.0 * np.random.random()
        radius = math.sqrt(np.random.random()) * 0.5
        point = [0.5 + radius * math.cos(phi), 0.5 + radius * math.sin(phi)]
    elif model == "2d_range_overlapping":
        phi = 2.0 * 180.0 * np.random.random()
        radius = math.sqrt(np.random.random()) * 0.5
        if cat == "voters":
            point = [0.25 + radius * math.cos(phi), 0.5 + radius * math.sin(phi)]
        elif cat == "candidates":
            point = [0.75 + radius * math.cos(phi), 0.5 + radius * math.sin(phi)]
    elif model in {"2d_square", "2d_uniform"}:
        point = [np.random.random(), np.random.random()]
    elif model in {'2d_asymmetric'}:
        if np.random.rand() < 0.3:
            return np.random.normal(loc=0.25, scale=0.15, size=2)
        else:
            return np.random.normal(loc=0.75, scale=0.15, size=2)
    elif model == "2d_sphere":
        alpha = 2 * math.pi * np.random.random()
        x = 1. * math.cos(alpha)
        y = 1. * math.sin(alpha)
        point = [x, y]
    elif model in ["2d_gaussian", "2d_range_gaussian"]:
        point = [np.random.normal(0.5, 0.15), np.random.normal(0.5, 0.15)]
        while np.linalg.norm(point - np.array([0.5, 0.5])) > 0.5:
            point = [np.random.normal(0.5, 0.15), np.random.normal(0.5, 0.15)]
    elif model in ["2d_range_fourgau"]:
        r = np.random.randint(1, 4)
        size = 0.06
        if r == 1:
            point = [np.random.normal(0.25, size), np.random.normal(0.5, size)]
        if r == 2:
            point = [np.random.normal(0.5, size), np.random.normal(0.75, size)]
        if r == 3:
            point = [np.random.normal(0.75, size), np.random.normal(0.5, size)]
        if r == 4:
            point = [np.random.normal(0.5, size), np.random.normal(0.25, size)]
    elif model in ["3d_cube", "3d_uniform"]:
        point = [np.random.random(), np.random.random(), np.random.random()]
    elif model in {'3d_asymmetric'}:
        if np.random.rand() < 0.3:
            return np.random.normal(loc=0.25, scale=0.15, size=3)
        else:
            return np.random.normal(loc=0.75, scale=0.15, size=3)
    elif model == "4d_cube":
        dim = 4
        point = [np.random.random() for _ in range(dim)]
    elif model == "5d_cube":
        dim = 5
        point = [np.random.random() for _ in range(dim)]
    elif model == "10d_cube":
        dim = 10
        point = [np.random.random() for _ in range(dim)]
    elif model == "20d_cube":
        dim = 20
        point = [np.random.random() for _ in range(dim)]
    elif model == "3d_sphere":
        dim = 3
        point = list(random_sphere(dim)[0])
    elif model == "4d_sphere":
        dim = 4
        point = list(random_sphere(dim)[0])
    elif model == "5d_sphere":
        dim = 5
        point = list(random_sphere(dim)[0])
    else:
        print('unknown model_id', model)
        point = [0, 0]
    return point


def get_range(params):
    if params['p_dist'] == 'beta':
        return np.random.beta(params['a'], params['b'])
    elif params['p_dist'] == 'uniform':
        return np.random.uniform(low=params['a'], high=params['b'])
    else:
        return params['p_dist']


# In[ ]:
def generate_approval_euclidean_candidate_range_votes(num_voters: int = None, num_candidates: int = None,
                                params: dict = None) -> list:
    # v_a = 1.05  # params['v_a']
    # v_b = 10  # params['v_b']
    # c_a = 1.05  # params['c_a']
    # c_b = 10  # params['c_b']

    # max_range = params['max_range']

    dim = params['dim']

    votes = [set() for _ in range(num_voters)]

    # voters = np.random.rand(num_voters, dim)
    # candidates = np.random.rand(num_candidates, dim)

    name = f'{dim}d_{params["space"]}'
    # print(name)

    voters = np.array([get_rand(name) for _ in range(num_voters)])
    candidates = np.array([get_rand(name) for _ in range(num_candidates)])

    # v_range = [np.random.beta(v_a, v_b) for _ in range(num_voters)]
    # c_range = [np.random.beta(c_a, c_b) for _ in range(num_candidates)]
    # [np.random.uniform(low=0.05, high=max_range) ...

    v_range = np.zeros(num_voters)
    c_range = np.random.normal(params['radius'], params['radius']/2, num_candidates)

    for v in range(num_voters):
        for c in range(num_candidates):
            if v_range[v] + c_range[c] >= np.linalg.norm(voters[v] - candidates[c]):
                votes[v].add(c)

    return votes




# In[6]:


elections_dict = {"n": [], "m": [], "model": [], "params": [], "inner_id": [], "election": [], "p": []}


# In[7]:


for model in models:
    model_id = model["model_id"]
    params_family = model["params_family"]
    params_tuples = itertools.product(*list(params_family.values()))
    for params_tuple in params_tuples:
        params = {k: v for k, v in zip(list(params_family.keys()), params_tuple)}
        for n in ns:
            ps = []
            for i in range(reps):
                e = generate_approval_euclidean_candidate_range_votes(num_voters=n, num_candidates=m, params=params)
                elections_dict["n"].append(n)
                elections_dict["m"].append(m)
                elections_dict["model"].append(model_id)
                elections_dict["params"].append(params)
                elections_dict["inner_id"].append(i)
                elections_dict["election"].append(e)
                p = sum([len(v) for v in e]) / (n * m)
                ps.append(round(p, 4))
                elections_dict["p"].append(p)
                # print(p)
            print(params, sum(ps) / len(ps), ps)
            # print(params, p)

    print("\n")

# In[ ]:


elections_df = pd.DataFrame(elections_dict)


# In[ ]:


elections_df


# In[26]:


dir_csv = "../results_server/euclidean_cr/csv_euclidean_cr"
dir_png = "../results_server/euclidean_cr/png_euclidean_cr"
os.makedirs(dir_csv, exist_ok=True)
os.makedirs(dir_png, exist_ok=True)
ns_name_to_save = [ns[0], ns[-1], ns[1]-ns[0] if len(ns) > 1 else ns[0]]
elections_df.to_csv(f"{dir_csv}/{model_id}_ns_{ns_name_to_save}_m_{m}_k_{k}_reps_{reps}.csv")


# In[37]:


elections_results_map = {"n": [], "m": [], "k": [], "model": [], "params": [], "inner_id": [], "rule_id": [], "unique": [], "time": []}


# In[41]:


for row in elections_df.itertuples():
    profile = abcvoting.preferences.Profile(num_cand=m)
    profile.add_voters(row.election)
    if row.inner_id == 0:
        print(f"{row.Index} election:    proceeding {row.inner_id}-th: {row.model} {row.params}")
    for rule_id in rules:
        d1 = datetime.datetime.now()
        # print(rule_id, d1)
        if rule_id != 'rule-x-phase-1':
            committees = abcvoting.abcrules.compute(rule_id=rule_id, profile=profile, committeesize=k,
                                                    resolute=False, max_num_of_committees=MAX_NUM_OF_COMMITTEES)
        else:
            committees = abcvoting.abcrules.compute_rule_x(profile=profile, committeesize=k, resolute=False,
                                                           skip_phragmen_phase=True, max_num_of_committees=MAX_NUM_OF_COMMITTEES)
        d2 = datetime.datetime.now()
        # print("   ", d2-d1)
        elections_results_map["n"].append(row.n)
        elections_results_map["m"].append(row.m)
        elections_results_map["k"].append(k)
        elections_results_map["model"].append(row.model)
        elections_results_map["params"].append(row.params)
        elections_results_map["inner_id"].append(row.inner_id)
        elections_results_map["rule_id"].append(rule_id)
        elections_results_map["unique"].append(1 if len(committees) == 1 else 0)
        elections_results_map["time"].append(d2-d1)


# In[43]:


elections_results_df = pd.DataFrame(elections_results_map)
elections_results_df


# In[49]:


elections_results_df[:-49]


# In[51]:


elections_results_df.to_csv(f"{dir_csv}/{model_id}_ns_{ns_name_to_save}_m_{m}_k_{k}_reps_{reps}_results.csv")


# In[65]:


unique_params = [dict(y) for y in set(tuple(x.items()) for x in list(elections_results_df["params"]))]
unique_params


# In[ ]:


print(f"\n\nInvestigating k={k} and m={m}")

stats_map = {"m": [], "k": [], "model": [], "params": [], "rule_id": [], "total": [], "unique_total": []}
for params in unique_params:
    print("-" * 100, params, sep="\n")
    pictures = []
    for rule_id in rules:
        df = elections_results_df.loc[(elections_results_df["params"] == params) & (elections_results_df["rule_id"] == rule_id)]
        # print(df)
        cnt = collections.defaultdict(lambda: 0)
        for row in df.itertuples():
            cnt[row.n] += row.unique
        stats_map["m"].append(row.m)
        stats_map["k"].append(row.k)
        stats_map["model"].append(row.model)
        stats_map["params"].append(row.params)
        stats_map["rule_id"].append(rule_id)
        stats_map["total"].append(reps)
        stats_map["unique_total"].append(dict(cnt))

        pic = plt.plot(list(cnt.keys()), [v / reps for v in cnt.values()], label=rule_id)
        pictures.append(pic)
    plt.title(f"Resampling {params}")
    plt.xlabel("n - number of voters")
    plt.ylabel("% of unique winning committees")
    plt.legend()
    # plt.legend(pictures, rules)
    plt.savefig(f"{dir_png}/{model_id}_ns_{ns_name_to_save}_m_{m}_k_{k}_reps_{reps}_params_{' '.join([str(i) for i in params.items() if i[0] != 'space'])}_aggregated.png")
    plt.show()
    plt.close()
    # plt.show()

pd.DataFrame(stats_map).to_csv(f"{dir_csv}/{model_id}_ns_{ns_name_to_save}_m_{m}_k_{k}_reps_{reps}_stats.csv")

# # In[ ]:
#
#
# print(f"Investigating k={k} and m={m}")
#
# for params in unique_params:
#     print("-" * 100)
#     for rule_id in rules:
#         df = elections_results_df.loc[(elections_results_df["params"] == params) & (elections_results_df["rule_id"] == rule_id)]
#         # print(df)
#         cnt = collections.defaultdict(lambda: 0)
#         for row in df.itertuples():
#             cnt[row.n] += row.unique
#         plt.title(f"Rule {rule_id}   params {params}")
#         plt.xlabel("n - number of voters")
#         plt.ylabel("% of unique winning committees")
#         plt.scatter(cnt.keys(), [v / reps for v in cnt.values()])
#         # plt.savefig(f"{model_id}_ns_{ns}_m_{m}_k_{k}_reps_{reps}_params_{' '.join([str(i) for i in params.items()])}_rule_{rule_id}.png")
#         # plt.show()
#
#
# # In[ ]:
#
#



print(f"Finished {__file__}")