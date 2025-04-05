from time import sleep

import argparse
import json
import shlex
import subprocess

MAX_PROCESSES = 16

parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, required=True)
args = parser.parse_args()

with open(f"run_configs/{args.name}.json", "r") as f:
    config = json.load(f)

def is_alive(p: subprocess.Popen):
    if p.poll() is None:
        return True

active_processes = []
idx = 1
total = 1
for k in config.keys():
    total *= len(config[k])

for dataset in config["dataset"]:
    for model in config["model"]:
        for gamma in config["gamma"]:
            for lr1 in config["lr1"]:
                for lr2 in config["lr2"]:
                    for cost in config["cost"]:
                        for fc in config["fairness_conditions"]:
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
                            while len(active_processes) > MAX_PROCESSES:
                                sleep(5)
                                active_processes = list(filter(is_alive, active_processes))
                            sleep(5)
                            print(f"{idx}/{total} started: {command}")
                            idx += 1
                            p = subprocess.Popen(command)
                            active_processes.append(p)

while(len(active_processes)) > 0:
    sleep(5)
    active_processes = list(filter(is_alive, active_processes))
    print(f"{len(active_processes)} trainings left")

