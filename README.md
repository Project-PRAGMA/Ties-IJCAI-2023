# IJCAI23 - Paper 4658 - Ties in Multiwinner Approval Voting

## Experiments:

We conducted several experiments to analyze ties frequency in multiwinner approval voting.

Multiwinner approval rules:
- all experiments:
  - AV (Approval Voting)
  - SAV (Satisfaction Approval Voting)
  - CCAV (Chamberlin-Courant Approval Voting)
  - PAV (Proportional Approval Voting)
  - GreedyPAV
  - Phragmen
  - MEqS (Method of Equal Shares, previously known as Rule-X)
  - First Phase of MEqS
- some experiments (excluded from more computationally expensive instances and omitted in the paper):
  - GreedyCCAV
  - SLAV
  - GreedySLAV

Several approval election models:
- Resampling
- Disjoint
- Euclidean_1D (Euclidean 1D with constant radius of candidates)
- Euclidean_CR (Euclidean 1D with changing radius of candidates)

We also introduced a model named pabulib, described in the paper in more details. 
Briefly, in this model we choose randomly a participatory budgeting instance containing sufficiently many projects,
then we select best_cands_num candidates with the highest av score (in the paper equal to m by default).
Further, we filter out votes not containing any of the remaining project
and uniformly select with replacement exactly m of them. 
In this experiment, we operate only on approval pb instances from pabulib library from Warsaw,
ignoring their costs.


The models were configured to test three situations depending on 
the average number of approved candidates, that is, ```p```:
  - circa half of the committee size (p = k/2)
  - circa the committee size (p = k)
  - circa twice the committee size (p = 2*k) (omitted for some larger instances)


If possible, we tried different levels of randomness in resampling and disjoint models. 
For euclidean models, we chose the radius to make the average number of approved candidates 
as close as possible to the above average numbers of candidates.


## How to run experiments:

At the beginning, please make sure you have the libraries from ```requirements.txt``` file installed.
You can do it via ```pip3 install requirements.txt```.
We recommend creating own virtual environment (conda or venv) and using ```Python 3.10.4```.

For each of the five models mentioned above, there is a directory with scripts for this model 
(that is, ```resampling, disjoint, euclidean, euclidean_cr, pabulib```). 
In each directory, there are multiple python files, each of the format: 
```Tie-breaking-approval_-_{text_with_model}_statistics_{_n_{n_range}_m={m}_k={k}_reps_{reps}.py```
To run such a file, go to file's directory and type ```python3 {file_name}```
(e.gz., ```python3 Tie-breaking-approval_-_draws_statistics_disjoint_n_[20,100]_m=10_k=5_reps_1000.py```)
For Pabulib the naming convention is slightly different (due to different parameters), 
but one can also see how we invoke them in ```run_all.sh``` script.
The script creates a directory results_server and stores results there.
Each script is initialized with the same seed, that is, ```47```. For this reason, the experiments are repeatable.
Each script generates ```{reps}``` elections for each ```n``` in ```{n_range}``` with a given values of ```m``` and ```k```.  

Finally, you generate charts via ```Ties in Multiwinner Approval Voting - Make Final Charts.py``` runnable in the same way as above.
It uses result files from ```results_server``` and stores them in ```results_local```).
Specifically, for reach rule R it is assumed that the path to stats files will be of format ```results_server/{R}/csv_{R}/```,
where stats files are files with aggregated statistics (that is, aggregates by combinations of parameters).
To do it, just type ```python3 Ties in Multiwinner Approval Voting - Make Final Charts.py``` and press Enter.


### Additional comments

In order to make our experiments reproducible, we've set randomness seed to a constant value, 
that is, 47 (both for numpy and random packages).
However, it still may happen that the results would be slightly different on your machine.
It may happen if any of your libraries is of different version, or, even in case of equal versions,
when randomness on your machine is implemented in another way than on our server.
Nevertheless, if you run these scripts twice, you should obtain the same results.

We would also like to emphasize that our scripts may require a significant amount of time and a lot of memory. 
Given m and k, we generate 1000 elections for each n in range [20,100]
and for each set of parameters. Then, for each rule, we need to compute two final committees
(or determine that there is exactly one). As one can imagine, it may require a significant amount of memory and time.
On our server, the most time-consuming were running for at least a few of days, further,
the most memory-consuming were using more that 60GB.