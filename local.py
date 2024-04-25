from time import sleep
import shlex
import subprocess

MAX_JOBS = 8
USER = (str(subprocess.run("whoami", stdout=subprocess.PIPE).stdout))[2:-3]
print(USER)

# Possible start/end values
# 0.00625, 0.0125, 0.01875, 0.025, 0.03125, 0.0375, 0.04375, 0.05, 0.05625, 0.0625, 0.06875, 0.075, 0.08125, 0.0875, 0.09375, 0.1
# 0.10625, 0.1125, 0.11875, 0.125, 0.13125, 0.1375, 0.14375, 0.15, 0.15625, 0.1625, 0.16875, 0.175, 0.18125, 0.1875, 0.19375, 0.2
# 0.20625, 0.2125, 0.21875, 0.225, 0.23125, 0.2375, 0.24375, 0.25, 0.25625, 0.2625, 0.26875, 0.275, 0.28125, 0.2875, 0.29375, 0.3
# 0.30625, 0.3125, 0.31875, 0.325, 0.33125, 0.3375, 0.34375, 0.35, 0.35625, 0.3625, 0.36875, 0.375, 0.38125, 0.3875, 0.39375, 0.4
# 0.40625, 0.4125, 0.41875, 0.425, 0.43125, 0.4375, 0.44375, 0.45, 0.45625, 0.4625, 0.46875, 0.475, 0.48125, 0.4875, 0.49375

# Values used to search
# 0.00625, 0.03125, 0.05625, 0.08125, 0.10625, 0.13125, 0.15625, 0.18125, 0.20625, 0.23125, 0.25625, 0.28125, 0.30625, 0.33125, 0.35625, 0.38125, 0.40625, 0.43125, 0.45625, 0.48125

settings = [
    {
        "model": "KP1",
        "dataset": "German",
        "fairness_condition": "None",
        "cost0": [0.20625, 0.2125, 0.21875, 0.225, 0.23125, 0.2375, 0.24375, 0.30625, 0.3125, 0.31875, 0.325, 0.33125, 0.3375, 0.34375, 0.35],
        "cost1": [0.20625, 0.2125, 0.21875, 0.225, 0.23125, 0.2375, 0.24375, 0.30625, 0.3125, 0.31875, 0.325, 0.33125, 0.3375, 0.34375, 0.35],
        "lr1": 1e-3,
        "lr2": 1e-3,
        "gamma": 0.7,
    },
    {
        "model": "KP1",
        "dataset": "German",
        "fairness_condition": "Ind",
        "cost0": [0.20625, 0.2125, 0.21875, 0.225, 0.23125, 0.2375, 0.24375, 0.25625, 0.2625, 0.26875, 0.275, 0.28125, 0.2875, 0.29375, 0.3],
        "cost1": [0.20625, 0.2125, 0.21875, 0.225, 0.23125, 0.2375, 0.24375, 0.25625, 0.2625, 0.26875, 0.275, 0.28125, 0.2875, 0.29375, 0.3],
        "lr1": 1e-3,
        "lr2": 3e-3,
        "gamma": 0.7,
    },
    {
        "model": "KP1",
        "dataset": "German",
        "fairness_condition": "Sep",
        "cost0": [0.20625, 0.2125, 0.21875, 0.225, 0.23125, 0.2375, 0.24375],
        "cost1": [0.20625, 0.2125, 0.21875, 0.225, 0.23125, 0.2375, 0.24375],
        "lr1": 1e-3,
        "lr2": 1e-3,
        "gamma": 0.7,
    },
    # ----------------------------------------
    # {
    #     "model": "RISAN",
    #     "dataset": "Compas",
    #     "fairness_condition": "None",
    #     "cost0": [0.2, 0.25, 0.3],
    #     "cost1": [0.2, 0.25, 0.3],
    #     "lr1": 1e-2,
    #     "lr2": 1e-3,
    #     "gamma": 1,
    # },
    # {
    #     "model": "RISAN",
    #     "dataset": "Compas",
    #     "fairness_condition": "Ind",
    #     "cost0": [0.2, 0.25, 0.3],
    #     "cost1": [0.2, 0.25, 0.3],
    #     "lr1": 1e-2,
    #     "lr2": 1e-3,
    #     "gamma": 1,
    # },
    # {
    #     "model": "RISAN",
    #     "dataset": "Compas",
    #     "fairness_condition": "Sep",
    #     "cost0": [0.2, 0.25, 0.3],
    #     "cost1": [0.2, 0.25, 0.3],
    #     "lr1": 1e-2,
    #     "lr2": 1e-3,
    #     "gamma": 1,
    # },
    # {
    #     "model": "RISAN",
    #     "dataset": "Compas",
    #     "fairness_condition": "Mixed",
    #     "cost0": [0.2, 0.25, 0.3],
    #     "cost1": [0.2, 0.25, 0.3],
    #     "lr1": 1e-2,
    #     "lr2": 1e-3,
    #     "gamma": 1,
    # },
    # {
    #     "model": "KP1",
    #     "dataset": "Compas",
    #     "fairness_condition": "None",
    #     "cost0": [0.2, 0.25, 0.3],
    #     "cost1": [0.2, 0.25, 0.3],
    #     "lr1": 1e-3,
    #     "lr2": 1e-3,
    #     "gamma": 0.7,
    # },
    # {
    #     "model": "KP1",
    #     "dataset": "Compas",
    #     "fairness_condition": "Ind",
    #     "cost0": [0.2, 0.25, 0.3],
    #     "cost1": [0.2, 0.25, 0.3],
    #     "lr1": 1e-3,
    #     "lr2": 1e-3,
    #     "gamma": 0.7,
    # },
    # {
    #     "model": "KP1",
    #     "dataset": "Compas",
    #     "fairness_condition": "Sep",
    #     "cost0": [0.2, 0.25, 0.3],
    #     "cost1": [0.2, 0.25, 0.3],
    #     "lr1": 1e-3,
    #     "lr2": 1e-3,
    #     "gamma": 0.7,
    # },
    # {
    #     "model": "KP1",
    #     "dataset": "Compas",
    #     "fairness_condition": "Mixed",
    #     "cost0": [0.2, 0.25, 0.3],
    #     "cost1": [0.2, 0.25, 0.3],
    #     "lr1": 1e-3,
    #     "lr2": 1e-3,
    #     "gamma": 0.7,
    # },
]


def alive(x):
    return x.poll() is None


processes = []
for s in settings:
    for c0, c1 in zip(s["cost0"], s["cost1"]):
        command = f"python -Wi main.py --dataset {s['dataset']} --model {s['model']} --fairness_condition {s['fairness_condition']} --cost_0 {c0} --cost_1 {c1} --lr1 {s['lr1']} --lr2 {s['lr2']} --gamma {s['gamma']} --tqdm No"
        cmd = shlex.split(command)
        while len(processes) >= MAX_JOBS:
            processes = list(filter(alive, processes))
            sleep(1)
        p = subprocess.Popen(
            cmd,
            stdout=open(
                f"{s['dataset']}_{s['model']}_{s['fairness_condition']}_{c0}_{c1}.txt",
                "w",
            ),
        )
        processes.append(p)
        sleep(1)

while len(processes) != 0:
    processes = list(filter(alive, processes))
    sleep(1)
print("Finished")
