#!/bin/bash
#SBATCH -A research
#SBATCH --qos=medium
#SBATCH -n 10
#SBATCH --gres=gpu:1
#SBATCH --time=4-00:00:00
#SBATCH --mail-user=tejas.chaudhari
#SBATCH --mail-type=END
#SBATCH --exclude=gnode[001-012,015-017,019-023,025-035,037-045,068]

python -Wi main.py --dataset "$1" --model "$2" --fairness_condition "$3"  --cost "$4" --tqdm No
