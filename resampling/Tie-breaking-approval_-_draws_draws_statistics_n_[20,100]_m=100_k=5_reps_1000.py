# In[1]:


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
m = 100
k = 5
reps = 1000
MAX_NUM_OF_COMMITTEES = 2


# In[4]:


models = [{"model_id": 'resampling', "params_family": {"p": [0.5*k/m, k/m, 2*k/m], "phi": [0.25, 0.5, 0.75, 1.0]}}]
# models = [{"model_id": 'resampling', "params_family": {"p": [0.5*k/m, k/m], "phi": [0.25, 0.5, 0.75]}}]


# In[5]:


# rules = ['av', 'sav', 'cc', 'seqcc', 'pav', 'seqpav', 'slav', 'seqslav', 'geom2', 'seqphragmen', 'rule-x', 'rule-x-phase-1']
rules = ['av', 'sav', 'cc', 'pav', 'seqpav', 'slav', 'seqslav', 'seqphragmen', 'rule-x-phase-1', 'rule-x']


# In[ ]:





# In[ ]:





# In[6]:


elections_dict = {"n": [], "m": [], "model": [], "params": [], "inner_id": [], "election": []}


# In[7]:


for model in models:
    model_id = model["model_id"]
    params_family = model["params_family"]
    params_tuples = itertools.product(*list(params_family.values()))
    for params_tuple in params_tuples:
        params = {k: v for k, v in zip(list(params_family.keys()), params_tuple)}
        for n in ns:
            for i in range(reps):
                e = mapel.elections.models_.generate_approval_votes(model_id='resampling', num_voters=n, num_candidates=m, params=params)
                elections_dict["n"].append(n)
                elections_dict["m"].append(m)
                elections_dict["model"].append(model_id)
                elections_dict["params"].append(params)
                elections_dict["inner_id"].append(i)
                elections_dict["election"].append(e)


# In[ ]:


elections_df = pd.DataFrame(elections_dict)


# In[ ]:


elections_df


# In[26]:


dir_csv = "../results_server/resampling/csv_resampling"
dir_png = "../results_server/resampling/png_resampling"
os.makedirs(dir_csv, exist_ok=True)
os.makedirs(dir_png, exist_ok=True)
ns_name_to_save = [ns[0], ns[-1], ns[1]-ns[0] if len(ns) > 1 else ns[0]]
elections_df.to_csv(f"{dir_csv}/mapel_with_random_resampling_ns_{ns_name_to_save}_m_{m}_k_{k}_reps_{reps}.csv")


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


elections_results_df.to_csv(f"{dir_csv}/mapel_with_random_resampling_ns_{ns_name_to_save}_m_{m}_k_{k}_reps_{reps}_results.csv")


# In[65]:


unique_params = [dict(y) for y in set(tuple(x.items()) for x in list(elections_results_df["params"]))]
unique_params


# In[ ]:


print(f"\n\nInvestigating k={k} and m={m}")

stats_map = {"m": [], "k": [], "model": [], "params": [], "rule_id": [], "total": [], "unique_total": []}
for params in unique_params:
    if params["p"] > 0.6:  # TODO - get rid of it
        continue
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
    plt.savefig(f"{dir_png}/mapel_with_random_resampling_ns_{ns_name_to_save}_m_{m}_k_{k}_reps_{reps}_params_{' '.join([str(i) for i in params.items()])}_aggregated.png")
    plt.show()
    plt.close()
    # plt.show()

pd.DataFrame(stats_map).to_csv(f"{dir_csv}/mapel_with_random_resampling_ns_{ns_name_to_save}_m_{m}_k_{k}_reps_{reps}_stats.csv")

# # In[ ]:
#
#
# print(f"Investigating k={k} and m={m}")
#
# for params in unique_params:
#     if params["p"] > 0.6:  # TODO - get rid of it
#         continue
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
#         # plt.savefig(f"mapel_with_random_resampling_ns_{ns}_m_{m}_k_{k}_reps_{reps}_params_{' '.join([str(i) for i in params.items()])}_rule_{rule_id}.png")
#         # plt.show()
#
#
# # In[ ]:
#
#



print(f"Finished {__file__}")