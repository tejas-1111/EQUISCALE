import numpy as np
import os
from get_data import get_adult, get_german, get_compas, get_bank, get_default
from make_experiment import run_experiment
import pickle
import os

from pathlib import Path
from metrics import combine_results
import json


def save_obj(obj, name):
    with open("results/" + name + ".pkl", "wb+") as f:
        pickle.dump(obj, f, 0)


def load_obj(name):
    with open("results/" + name + ".pkl", "rb") as f:
        return pickle.load(f)


RUN_ADULT = True
RUN_COMPAS = True
RUN_GERMAN = True
RUN_BANK = True
RUN_DEFAULT = True

METHODS = ["LR"]


setup = {
    "proc_train": 0.6,
    "proc_unlab": 0.2,
    "cv": 5,
    "num_c": 30,
    "num_gamma": 30,
    "verbose": 0,
    "n_jobs": 4,
}

check_run = {"Adult": RUN_ADULT, "Compas": RUN_COMPAS, "German": RUN_GERMAN, "Bank": RUN_BANK, "Default": RUN_DEFAULT}
datasets = {"Compas": get_compas, "German": get_german, "Adult": get_adult, "Bank": get_bank,"Default": get_default}
seeds = np.arange(10)

# setting alphas to a number means that all groups have the same reject rate
alphas_grid = np.linspace(0.4, 0.8, 21)

total = len(seeds)

if not os.path.exists("results"):
    os.makedirs("results")
if not os.path.exists("results/models"):
    os.makedirs("results/models")
for method in METHODS:
    print("[{}] is running".format(method))
    for data in datasets.keys():
        if not check_run[data]:
            print("[{}] skipped".format(data))
        else:
            X, y = datasets[data]()
            sensitives = np.unique(X[:, -1])
            for alphas in alphas_grid:
                results = []
                print("[Classification rate] {}".format(alphas))
                for i, seed in enumerate(seeds):
                    print("[{}]: {}/{}".format(data, i + 1, total))
                    result_seed = run_experiment(
                        X=X,
                        y=y,
                        alphas=alphas,
                        seed=seed,
                        method=method,
                        data_name=data,
                        **setup,
                    )
                    results.append(result_seed)

                avg, std = combine_results(results, True)
                pt = Path(f"../results/{data.lower()}/dpabst/ind")
                os.makedirs(pt, exist_ok=True)
                pt = Path(f"../results/{data.lower()}/dpabst/ind/{alphas:f}.json")
                if isinstance(avg, str) or isinstance(std, str):
                    print("run_experiments error [1]")
                    exit()
                results = {}
                for key in avg.keys():
                    results[key] = [avg[key].tolist(), std[key].tolist()]
                with open(
                    pt,
                    "w",
                ) as f:
                    f.write(json.dumps(results, indent=4))
