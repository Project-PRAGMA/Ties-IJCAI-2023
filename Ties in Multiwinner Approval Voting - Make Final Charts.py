#!/usr/bin/env python
# coding: utf-8

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

import ast


# In[62]:


MODELS = ['resampling', 'disjoint', 'euclidean', 'euclidean_cr', 'pabulib']


# In[3]:


INPUT_BASE_DIR = "results_server"


# In[4]:


# OUTPUT_BASE_DIR = None
OUTPUT_BASE_DIR = "results_local"


# In[5]:


output_png_dir = OUTPUT_BASE_DIR + "/png"


# In[6]:


def format_axen_dir(skip_xlabel=False, skip_ylabel=False):
    x_part = 'xlabel' if not skip_xlabel else ''
    y_part = 'ylabel' if not skip_ylabel else ''
    if not x_part and not y_part:
        return "no_label"
    if not x_part or not y_part:
        return x_part + y_part
    else:
        return x_part + "_" + y_part


# In[75]:


if output_png_dir is not None:
    for model in MODELS:
        for skip_xlabel, skip_ylabel in [(True, True), (True, False), (False, True), (False, False)]:
            os.makedirs(os.path.join(output_png_dir, 'unique', model, format_axen_dir(skip_xlabel, skip_ylabel)), exist_ok=True)
            os.makedirs(os.path.join(output_png_dir, 'tie', model, format_axen_dir(skip_xlabel, skip_ylabel)), exist_ok=True)


# In[79]:


# RULES = ['av', 'sav', 'cc', 'seqcc', 'pav', 'seqpav', 'slav', 'seqslav', 'geom2', 'seqphragmen', 'rule-x']
# RULES = ['av', 'sav', 'cc', 'pav', 'seqpav', 'slav', 'seqslav', 'geom2', 'seqphragmen', 'rule-x']
# RULES = ['av', 'sav', 'cc', 'pav', 'seqpav', 'slav', 'seqslav', 'seqphragmen'] #, 'rule-x']
# RULES = ['av', 'sav', 'cc', 'pav', 'seqpav', 'seqphragmen', 'rule-x', 'rule-x-phase-1']
RULES = ['sav', 'pav', 'seqpav',  'rule-x', 'rule-x-phase-1', 'seqphragmen', 'av', 'cc']


# In[9]:


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
    'rule-x-phase-1': "MEqS-Phase-I"
}


# In[10]:


reps = 1000


# #### Util

#  

# In[18]:


def get_dir(model_name="resampling", data_type="csv", base_dir=INPUT_BASE_DIR, create_if_absent=False):
    dir = os.path.join(base_dir, model_name, data_type+"_"+model_name)
    if not os.path.exists(dir):
        os.makedirs(dir, exist_ok=True)
    return dir


# In[12]:


def get_unique_values(l: list):
    ans = []
    for x in l:
        if x not in ans:
            ans.append(x)
    return list(ans)


# In[13]:


def get_rule_name(rule_id):
    return RULE_TO_LEGEND_NAME_DICT[rule_id] if rule_id in RULE_TO_LEGEND_NAME_DICT else rule_id


# In[14]:


def format_params(params: dict, accuracy_len=3):
    result_dict = {}
    for k, v in params.items():
        if type(v) != float:
            result_dict[k] = v
        else: 
            result_dict[k] = round(v, accuracy_len)
    return result_dict


# In[15]:


def load_dataframe(dir, file_name):
    df = pd.read_csv(os.path.join(dir, file_name))
    if "unique_total" in df.columns:
        df.unique_total = df.unique_total.apply(ast.literal_eval)
    if "params" in df.columns:
        df.params = df.params.apply(ast.literal_eval)
    return df


# In[106]:


def generate_charts(elections_results_df, models_list=None, params_list=None, rules=RULES, 
                    toggle_unique=True, skip_xlabel=False, skip_ylabel=False, skip_title=False,
                    styles_list = ['o-', 'x--', 'P-.', 'd:', 's-', '^--', '1-.', '*:'], figsize=(6.4, 5.0),
                    suffix=""
                   ):
    m = get_unique_values(elections_results_df["m"])[0]
    k = get_unique_values(elections_results_df["k"])[0]
    if models_list is None:
        models_list = get_unique_values(elections_results_df["model"])
    if params_list is None:
        params_list = get_unique_values(elections_results_df["params"])
    for model in models_list:
        for params in params_list:
            print("-" * 100)
            print(model, params)
            pictures = []
            plt.figure(figsize=figsize)
            for rule_id, style in zip(rules, styles_list):
                df = elections_results_df.loc[(elections_results_df["model"] == model) & (elections_results_df["params"] == params) & (elections_results_df["rule_id"] == rule_id)]
                if df.empty:
                    continue
                # print(df)
                cnt = df.iloc[0]["unique_total"]
                if toggle_unique:
                    pic = plt.plot(list(cnt.keys()), [v / reps for v in cnt.values()], style, markevery=10, label=get_rule_name(rule_id))
                else:
                    pic = plt.plot(list(cnt.keys()), [(reps - v) / reps for v in cnt.values()], style, markevery=10, label=get_rule_name(rule_id))
                pictures.append(pic)
            if not skip_title:
                plt.title(f"{model} {format_params(params)}", fontsize=17)
            if not skip_xlabel:
                plt.xlabel("n - number of voters", fontdict={'size': 15})
            if not skip_ylabel:
                if toggle_unique:
                    plt.ylabel("Ratio of unique winning committees", fontdict={'size': 15}) 
                else:
                    plt.ylabel("Ratio of ties in winning committees", fontdict={'size': 15})
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            plt.legend(fontsize=13)
            # plt.savefig(f"{output_png_dir}/mapel_with_random_{model_id}_ns_{ns_name_to_save}_m_{m}_k_{k}_reps_{reps}_params_{' '.join([str(i) for i in params.items()])}_aggregated.png")
            model_id = model
            print("Saving chart of model " + model_id + " to " + f"{output_png_dir}/{'unique' if toggle_unique else 'tie'}/{model_id}/{format_axen_dir(skip_xlabel, skip_ylabel)}/m_{m}_k_{k}_reps_{reps}_params_{' '.join([str(i) for i in params.items()])}{suffix}.png")
            if not os.path.exists(f"{output_png_dir}/{'unique' if toggle_unique else 'tie'}/{model_id}/{format_axen_dir(skip_xlabel, skip_ylabel)}/"):
                os.makedirs(f"{output_png_dir}/{'unique' if toggle_unique else 'tie'}/{model_id}/{format_axen_dir(skip_xlabel, skip_ylabel)}/", exist_ok=True)
            plt.savefig(f"{output_png_dir}/{'unique' if toggle_unique else 'tie'}/{model_id}/{format_axen_dir(skip_xlabel, skip_ylabel)}/m_{m}_k_{k}_reps_{reps}_params_{' '.join([str(i) for i in params.items()])}{suffix}.png")
            plt.show()
            plt.close()


# #### Code

# In[63]:


resampling_dir = get_dir("resampling")
disjoint_dir = get_dir("disjoint")
euclidean_dir = get_dir("euclidean")
euclidean_cr_dir = get_dir("euclidean_cr")
pabulib_dir = get_dir("pabulib")

input_dirs = [resampling_dir, disjoint_dir, euclidean_dir, euclidean_cr_dir, pabulib_dir]


# In[ ]:





# In[ ]:





# In[ ]:





# In[20]:


os.listdir(resampling_dir)


# In[ ]:





# In[25]:


for file in os.listdir(resampling_dir):
        if file.endswith(f"reps_{reps}_stats.csv"):
            example_election_df = load_dataframe(resampling_dir, file)
            break
example_election_df


# In[19]:


example_election_df["unique_total"]


# In[ ]:





# In[ ]:





# In[26]:


generate_charts(example_election_df)


# In[27]:


generate_charts(example_election_df, toggle_unique=False)


# In[ ]:





# In[ ]:





# In[28]:


file


# In[32]:




# # In[85]:


for input_dir in input_dirs:
    model = input_dir.split('\\')[1]
#     print(model, input_dir, input_dir.split('\\'))
    os.makedirs(os.path.join(OUTPUT_BASE_DIR, input_dir), exist_ok=True)
    for file in os.listdir(input_dir):
        if not file.endswith(f"_stats.csv"):
            continue
        election_stats_df = load_dataframe(input_dir, file)
        if 'rule-x' in election_stats_df.rule_id and 'rule-x-phase-1' in election_stats_df.rule_id:
            election_stats_df.to_csv(os.path.join(OUTPUT_BASE_DIR, input_dir, file))
            continue
        try:
            mes_stats_df = load_dataframe(os.path.join(INPUT_BASE_DIR, 'only_mes', 'new', model), file)
#             print("   ", mes_stats_df[:5])
            election_stats_df = election_stats_df[election_stats_df.rule_id != 'rule-x']
            election_stats_df = election_stats_df[election_stats_df.rule_id != 'rule-x-phase-1']
            election_stats_df = pd.concat([election_stats_df, mes_stats_df])
            print("   ", election_stats_df[:5], election_stats_df[-5:])
#             print(election_stats_df, mes_stats_df, df, sep="\n")
        except Exception as exc:
            print(exc)
        finally:
            print(os.path.join(OUTPUT_BASE_DIR, INPUT_BASE_DIR, input_dir, file))
            election_stats_df.to_csv(os.path.join(OUTPUT_BASE_DIR, input_dir, file))




# In[95]:


election_stats_df.rule_id.to_list()


# In[97]:


[r not in election_stats_df.rule_id.to_list() for r in RULES]


# In[100]:


input_dirs


# In[104]:


for input_dir in input_dirs:
    for file in os.listdir(input_dir):
        if not file.endswith(f"_stats.csv"):
            continue
        election_stats_df = load_dataframe(input_dir, file)        
        print(input_dir, file, [r not in election_stats_df.rule_id.to_list() for r in RULES])
        if any([r not in election_stats_df.rule_id.to_list() for r in RULES]):
            print("   FAIL")
            continue
        for skip_xlabel, skip_ylabel in [(False, False)]:  # [(True, True), (True, False), (False, True), (False, False)]:
            generate_charts(election_stats_df, skip_xlabel=skip_xlabel, skip_ylabel=skip_ylabel, skip_title=True)
            generate_charts(election_stats_df, toggle_unique=False, skip_xlabel=skip_xlabel, skip_ylabel=skip_ylabel, skip_title=True)


# In[107]:


# for input_dir in ['results_local/results_server\pabulib_voters_with_replacement/csv_pabulib']:
#     for file in os.listdir(input_dir):
#         if not file.endswith(f"_stats.csv"):
#             continue
#         election_stats_df = load_dataframe(input_dir, file)        
#         print(input_dir, file, [r not in election_stats_df.rule_id.to_list() for r in RULES])
#         best_cands_num_suffix = file.split("reps_1000_")[1].split("_stats")[0]
#         if any([r not in election_stats_df.rule_id.to_list() for r in RULES]):
#             print(   "FAIL")
#             continue
#         for skip_xlabel, skip_ylabel in [(False, False)]:  # [(True, True), (True, False), (False, True), (False, False)]:
#             generate_charts(election_stats_df, skip_xlabel=skip_xlabel, skip_ylabel=skip_ylabel, skip_title=True, suffix=best_cands_num_suffix)
#             generate_charts(election_stats_df, toggle_unique=False, skip_xlabel=skip_xlabel, skip_ylabel=skip_ylabel, skip_title=True, suffix=best_cands_num_suffix)


# In[108]:


# for input_dir in [pabulib_dir]:
#     for file in os.listdir(input_dir):
#         if not file.endswith(f"_stats.csv"):
#             continue
#         election_stats_df = load_dataframe(input_dir, file)        
#         print(input_dir, file, [r not in election_stats_df.rule_id.to_list() for r in RULES])
#         best_cands_num_suffix = file.split("reps_1000_")[1].split("_stats")[0]
#         if any([r not in election_stats_df.rule_id.to_list() for r in RULES]):
#             print(   "FAIL")
#             continue
#         for skip_xlabel, skip_ylabel in [(False, False)]:  # [(True, True), (True, False), (False, True), (False, False)]:
#             generate_charts(election_stats_df, skip_xlabel=skip_xlabel, skip_ylabel=skip_ylabel, skip_title=True, suffix=best_cands_num_suffix)
#             generate_charts(election_stats_df, toggle_unique=False, skip_xlabel=skip_xlabel, skip_ylabel=skip_ylabel, skip_title=True, suffix=best_cands_num_suffix)


# In[ ]:





# In[ ]:





# In[ ]:







