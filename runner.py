import argparse
import json
import shlex
import subprocess

from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, required=True)
args = parser.parse_args()

with open(f"run_configs/{args.name}.json", "r") as f:
    config = json.load(f)

for c in tqdm(config["cost"], desc="Models progress", leave=False):
    if len(config["fairness_conditions"]) != 0:
        command = f"""python -Wi main.py --dataset {config["dataset"]} --model {config["model"]}
            --cost {c} --gamma {config["gamma"]} --lr1 {config["lr1"]} --lr2 {config["lr2"]} --tqdm
            --fairness_conditions {" ".join(config["fairness_conditions"])}"""
    else:
        command = f"""python -Wi main.py --dataset {config["dataset"]} --model {config["model"]}
            --cost {c} --gamma {config["gamma"]} --lr1 {config["lr1"]} --lr2 {config["lr2"]} --tqdm
            """
    command = shlex.split(command)
    subprocess.run(command)
