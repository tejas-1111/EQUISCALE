from time import sleep

import argparse
import json
import shlex
import subprocess

MAX_PROCESSES = 24

parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, required=True)
args = parser.parse_args()

with open(f"run_configs/{args.name}.json", "r") as f:
    data = json.load(f)


def is_alive(p: subprocess.Popen):
    if p.poll() is None:
        return True


active_processes = []
idx = 1
total = 0
for config in data["configs"]:
    total += len(config["costs"])
for config in data["configs"]:
    for cost in config["costs"]:
        fc = config["fairness_conditions"]
        dataset = config["dataset"]
        model = config["model"]
        gamma = config["gamma"]
        lr1 = config["lr_1"]
        lr2 = config["lr_2"]
        if fc == "":
            command = f"""python -Wi main.py --dataset {dataset} --model {model}
            --cost {cost} --gamma {gamma} --lr1 {lr1} --lr2 {lr2}
            """
        else:
            fairness_conditions = " ".join(fc.split("-"))
            command = f"""python -Wi main.py --dataset {dataset} --model {model}
            --cost {cost} --gamma {gamma} --lr1 {lr1} --lr2 {lr2} --fairness_conditions {fairness_conditions}
            """
        command = shlex.split(command)
        while len(active_processes) >= MAX_PROCESSES:
            sleep(1)
            active_processes = list(filter(is_alive, active_processes))
        sleep(1)
        print(f"{idx}/{total} started: {command}")
        idx += 1
        p = subprocess.Popen(command)
        active_processes.append(p)

while (len(active_processes)) > 0:
    sleep(1)
    active_processes = list(filter(is_alive, active_processes))
    print(f"{len(active_processes)} trainings left")
