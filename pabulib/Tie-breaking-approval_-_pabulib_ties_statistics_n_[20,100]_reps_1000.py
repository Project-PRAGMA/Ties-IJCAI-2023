#!/usr/bin/env python
# coding: utf-8

# In[41]:


print("Hi! We will conduct some experiments using real-life data Pabulib. Let's begin with some imports!")


# In[42]:


import abcvoting
import abcvoting.preferences
import abcvoting.abcrules

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import collections

import os
import sys
import datetime


# In[2]:


SEED = 47
random.seed(SEED)
np.random.seed(SEED)


# In[3]:


print(f"Script invoked with {sys.argv}")


# In[4]:


IS_PYTHON_SCRIPT = False
try:
    IS_PYTHON_SCRIPT = __file__[-3:] == '.py'
except:
    pass
print(f"IS_PYTHON_SCRIPT {IS_PYTHON_SCRIPT}")


# In[5]:


# You may change them
if len(sys.argv) == 1 or not IS_PYTHON_SCRIPT:
    m = 10
    k = 5
    number_of_most_approved_candidates_to_consider = None
elif len(sys.argv) == 3:
    m = int(sys.argv[1])
    k = int(sys.argv[2])
    number_of_most_approved_candidates_to_consider = None
elif len(sys.argv) == 4:
    m = int(sys.argv[1])
    k = int(sys.argv[2])
    number_of_most_approved_candidates_to_consider = int(sys.argv[3])
else:
    raise Exception("Incorrect number of arguments: expected 0 arguments or 2 arguments 'm k' or 3 arguments 'm k number_of_most_approved_candidates_to_consider'")
    sys.exit(1)


# In[6]:


ns = list(range(20,100+1,1))
reps = 1000

# Do not change
models = ['pabulib']
MAX_NUM_OF_COMMITTEES = 2


# In[7]:


print(f"Calculation started for arguments m={m}, k={k}, ns={ns}, reps={reps}, number_of_most_approved_candidates_to_consider={number_of_most_approved_candidates_to_consider}")


# In[8]:





# In[ ]:





# In[9]:


# rules = ['av', 'sav', 'cc', 'pav', 'seqpav', 'slav', 'seqslav', 'seqphragmen', 'rule-x', 'rule-x-phase-1']
rules = ['av', 'sav', 'cc', 'pav', 'seqpav', 'seqphragmen', 'rule-x', 'rule-x-phase-1']


# In[10]:


os.path.abspath(".")


# In[11]:


# INPUT_DIR = '..\\..\\20230110234132_pabulib'
INPUT_DIR = "20230114230136_pabulib"


# In[12]:


dir_csv = "../results_server/pabulib/csv_pabulib"
dir_png = "../results_server/pabulib/png_pabulib"
os.makedirs(dir_csv, exist_ok=True)
os.makedirs(dir_png, exist_ok=True)
ns_name_to_save = [ns[0], ns[-1], ns[1]-ns[0] if len(ns) > 1 else ns[0]]


# In[ ]:





# In[13]:


def parse_election_from_pb_file(input_file):
    with open(input_file, encoding='utf8') as f:
        lines = f.readlines()
        id_cnt = 0
        id_to_ind_map = {}
        votes = []
        for line in lines[lines.index("VOTES\n")+2:]:
            parts = line.split(";")
            original_projects = parts[-1].split("\n")[0].split(",")
            vote = []
            for project in original_projects:
                if project not in id_to_ind_map:
                    id_to_ind_map[project] = id_cnt
                    id_cnt += 1
                id = id_to_ind_map[project]
                vote.append(id)
            votes.append(list(set(vote)))   # to avoid duplicates as in e.g. poland_warszawa_2022_wola
        return votes, id_to_ind_map


# In[ ]:





# In[14]:


pb_elections_dict = {"n": [], "m": [], "model": [], "params": [], "inner_id": [], "election": []}


# In[15]:


for file in os.listdir(INPUT_DIR):
    if file[-3:] != ".pb":
        continue
    votes, id_to_ind_map = parse_election_from_pb_file(os.path.join(INPUT_DIR, file))
    tmp_n = len(votes)
    tmp_m = len(id_to_ind_map)
    i = 1
    pb_elections_dict["n"].append(tmp_n)
    pb_elections_dict["m"].append(tmp_m)
    pb_elections_dict["model"].append(file[:-3])
    pb_elections_dict["params"].append({"cands_mapping": id_to_ind_map})
    pb_elections_dict["inner_id"].append(i)
    pb_elections_dict["election"].append(votes)


# In[16]:


pb_elections_df = pd.DataFrame(pb_elections_dict)


# In[17]:


del pb_elections_dict


# In[18]:


pb_elections_df


# In[ ]:





# In[19]:


# el_rec = pb_elections_df.loc[311]
# p = abcvoting.preferences.Profile(num_cand=el_rec.m)
# p.add_voters(el_rec.election)
# # p.num_cand, p.aslist()


# In[20]:


# for tmp_k in [1,2]:
#     if tmp_k > p.num_cand:
#         continue
#     for rule_id in rules:
#         d1 = datetime.datetime.now()
#         if rule_id != 'rule-x-phase-1':
#             committees = abcvoting.abcrules.compute(rule_id=rule_id, profile=p, committeesize=tmp_k,
#                                                           resolute=False, max_num_of_committees=MAX_NUM_OF_COMMITTEES)
#         else:
#             committees = abcvoting.abcrules.compute_rule_x(profile=p, committeesize=tmp_k, resolute=False,
#                                                            skip_phragmen_phase=True, max_num_of_committees=MAX_NUM_OF_COMMITTEES)
#         print(f"{tmp_k}   {rule_id}  |   {committees}")


# In[ ]:





# In[21]:


pb_elections_df.to_csv(f"{dir_csv}/pabulib_input_elections_ns_{ns_name_to_save}_m_{m}_k_{k}_reps_{reps}_best_cands_num_{number_of_most_approved_candidates_to_consider}.csv")


# In[ ]:





# In[22]:


# Assumption: all candidates in big_elections are from range [0,m-1] for some m and m-1 is present in at least one vote
def sample_election_from_a_given_bigger_election(big_election: list, dest_n: int, dest_m: int,
                                                 number_of_most_approved_candidates_to_consider: int = None,
                                                 reindex_to_0_m_range: bool = True) -> list:
    cnt = collections.Counter()
    for vote in big_election:
        for c in vote:
            cnt[c] += 1
    max_m = max([c for c in cnt])
    if number_of_most_approved_candidates_to_consider is None or number_of_most_approved_candidates_to_consider > max_m:
        number_of_most_approved_candidates_to_consider = max_m
    cands_sorted_by_approvals = [c for c, a in cnt.most_common(number_of_most_approved_candidates_to_consider)]
    selected_cands_indices = np.random.choice(min(number_of_most_approved_candidates_to_consider, len(cands_sorted_by_approvals)), 
                                              size=min(dest_m, number_of_most_approved_candidates_to_consider, len(cands_sorted_by_approvals)), 
                                              replace=False)
#     print(cands_sorted_by_approvals, selected_cands_indices)
    selected_cands = [cands_sorted_by_approvals[i] 
                      for i in selected_cands_indices]
    selected_cands_set = set(selected_cands)
    possible_votes = [list(set(vote) & selected_cands_set) for vote in big_election if (set(vote) & selected_cands_set)]
    votes_indices = np.random.choice(len(possible_votes), size=dest_n, replace=True)
    possible_votes = [possible_votes[i].copy() for i in votes_indices]
    if reindex_to_0_m_range:
        selected_cand_to_ind = {c: i for i, c in enumerate(selected_cands)}
        possible_votes = [[selected_cand_to_ind[c] for c in vote] for vote in possible_votes]
    return possible_votes


# In[23]:


elections_dict = {"n": [], "m": [], "model": [], "params": [], "inner_id": [], "election": [], "p": []}

ps = []

for model_id in models:
    for n in ns:
        print(f"{model_id}, {n}")
        for i in range(reps):
            for _ in range(2*len(pb_elections_df)):
                ind = np.random.randint(len(pb_elections_df))
                row = pb_elections_df.loc[ind]
                if len(row.params['cands_mapping']) >= m:
                    break
            e = sample_election_from_a_given_bigger_election(row.election, dest_n=n, dest_m=m, 
                                                            number_of_most_approved_candidates_to_consider=number_of_most_approved_candidates_to_consider,
                                                            reindex_to_0_m_range=True)
            p = sum([len(v) for v in e]) / (len(e) * m)
            ps.append(p)
            elections_dict["n"].append(n)
            elections_dict["m"].append(m)
            elections_dict["model"].append(model_id)
            elections_dict["params"].append({"orig_pb_el": row.model})
            elections_dict["inner_id"].append(i)
            elections_dict["election"].append(e)
            elections_dict["p"].append(p)


# In[24]:


print(f"Average number of approved candidates: {sum(ps) / len(ps)}")

del pb_elections_df


# In[25]:


elections_df = pd.DataFrame(elections_dict)
elections_df


# In[26]:


del elections_dict


# In[27]:


elections_df['election'][0]


# In[28]:


elections_df.to_csv(f"{dir_csv}/elections_ns_{ns_name_to_save}_m_{m}_k_{k}_reps_{reps}_best_cands_num_{number_of_most_approved_candidates_to_consider}.csv")


# In[ ]:





# In[ ]:





# In[29]:


elections_results_map = {"n": [], "m": [], "k": [], "model": [], "params": [], "inner_id": [], "rule_id": [], "unique": [], "time": []}


# In[30]:


for row in elections_df.itertuples():
    profile = abcvoting.preferences.Profile(num_cand=m) 
    profile.add_voters(row.election)
    print(f"{row.Index} election:    proceeding {row.inner_id}-th: {row.model} {row.params}")
    for rule_id in rules:
        d1 = datetime.datetime.now()
        if rule_id != 'rule-x-phase-1':
            committees = abcvoting.abcrules.compute(rule_id=rule_id, profile=profile, committeesize=k,
                                                          resolute=False, max_num_of_committees=MAX_NUM_OF_COMMITTEES)
        else:
            committees = abcvoting.abcrules.compute_rule_x(profile=profile, committeesize=k, resolute=False,
                                                           skip_phragmen_phase=True, max_num_of_committees=MAX_NUM_OF_COMMITTEES)
        d2 = datetime.datetime.now()
        elections_results_map["n"].append(row.n)
        elections_results_map["m"].append(row.m)
        elections_results_map["k"].append(k)
        elections_results_map["model"].append(row.model)
        elections_results_map["params"].append(row.params)
        elections_results_map["inner_id"].append(row.inner_id)
        elections_results_map["rule_id"].append(rule_id)
        elections_results_map["unique"].append(1 if len(committees) == 1 else 0)
        elections_results_map["time"].append(d2-d1)


# In[31]:


del elections_df


# In[32]:


elections_results_df = pd.DataFrame(elections_results_map)
elections_results_df


# In[33]:


del elections_results_map


# In[34]:


# ties_elections_df = elections_results_df[elections_results_df['unique'] == 0]
# ties_elections_df


# In[35]:


elections_results_df.to_csv(f"{dir_csv}/pabulib_ns_{ns_name_to_save}_m_{m}_k_{k}_reps_{reps}_best_cands_num_{number_of_most_approved_candidates_to_consider}_results.csv")


# In[ ]:





# In[36]:


RULE_TO_LEGEND_NAME_DICT = {
    'av': "AV",
    'sav': "SAV",
    'cc': "CCAV",
    'seqcc': "GreedyCC",
    'pav': "PAV",
    'seqpav': "GreedyPAV",
    'slav': "SLAV",
    'seqslav': "GreedySLAV",
    'geom2': "Geom2",
    'seqphragmen': "Phragmen",
    'rule-x': "MEqS",
    'rule-x-phase-1': "MEqS-Phase-I",
}


# In[37]:


def get_rule_name(rule_id):
    return RULE_TO_LEGEND_NAME_DICT[rule_id] if rule_id in RULE_TO_LEGEND_NAME_DICT else rule_id


# In[38]:


print(f"\n\nInvestigating k={k} and m={m}")

stats_map = {"m": [], "k": [], "model": [], "params": [], "rule_id": [], "total": [], "unique_total": []}

for model_id in models:
    print("-" * 100, model_id, sep="\n")
    pictures = []
    for rule_id in rules:
        df = elections_results_df.loc[(elections_results_df["model"] == model_id) & (elections_results_df["rule_id"] == rule_id)]
        cnt = collections.defaultdict(lambda: 0)
        for row in df.itertuples():
            cnt[row.n] += row.unique
        stats_map["m"].append(row.m)
        stats_map["k"].append(row.k)
        stats_map["model"].append(row.model)
        stats_map["params"].append({})
        stats_map["rule_id"].append(rule_id)
        stats_map["total"].append(reps)
        stats_map["unique_total"].append(dict(cnt))

        pic = plt.plot(list(cnt.keys()), [v / reps for v in cnt.values()], label=rule_id)
        pictures.append(pic)
    plt.title(f"{model_id}")
    plt.xlabel("n - number of voters")
    plt.ylabel("% of unique winning committees")
    plt.legend()
    # plt.legend(pictures, rules)
    plt.savefig(f"{dir_png}/{model_id}_ns_{ns_name_to_save}_m_{m}_k_{k}_reps_{reps}_best_cands_num_{number_of_most_approved_candidates_to_consider}_aggregated.png")
    plt.show()
    plt.close()
    # plt.show()


# In[39]:


pd.DataFrame(stats_map).to_csv(f"{dir_csv}/{model_id}_ns_{ns_name_to_save}_m_{m}_k_{k}_reps_{reps}_best_cands_num_{number_of_most_approved_candidates_to_consider}_stats.csv")


# In[ ]:





# In[40]:


print(f"Finished for models={models} ns={ns_name_to_save} m={m} k={k} reps={reps} best_cands_num={number_of_most_approved_candidates_to_consider}")


# In[ ]:




