from time import sleep
import shlex
import subprocess

MAX_JOBS = 6
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

# Entry in setting should be of the form:
# {
#         "model": "RISAN",
#         "dataset": "Compas",
#         "fairness_condition": "Sep",
#         "costs": [0.10625, 0.13125, 0.15625, 0.18125, 0.20625, 0.23125, 0.25625, 0.28125, 0.30625, 0.33125, 0.35625, 0.38125, 0.40625],
# },

settings = [
    {
        "model": "RISAN",
        "dataset": "Adult",
        "fairness_condition": "None",
        "costs": [0.187500, 0.197813, 0.208125, 0.218438, 0.228750, 0.239063, 0.249375, 0.259687, 0.270000, 0.280312, 0.290625, 0.300937, 0.311250, 0.321562, 0.331875, 0.342187, 0.352500, 0.362812, 0.373125, 0.383437, 0.393750],
    },
    {
        "model": "RISAN",
        "dataset": "Adult",
        "fairness_condition": "Ind",
        "costs": [0.218750, 0.225938, 0.233125, 0.240312, 0.247500, 0.254688, 0.261875, 0.269062, 0.276250, 0.283438, 0.290625, 0.297812, 0.305000, 0.312188, 0.319375, 0.326562, 0.333750, 0.340938, 0.348125, 0.355312, 0.362500],
    },
    {
        "model": "RISAN",
        "dataset": "Adult",
        "fairness_condition": "Sep",
        "costs": [0.225000, 0.234375, 0.243750, 0.253125, 0.262500, 0.271875, 0.281250, 0.290625, 0.300000, 0.309375, 0.318750, 0.328125, 0.337500, 0.346875, 0.356250, 0.365625, 0.375000, 0.384375, 0.393750, 0.403125, 0.412500],
    },
]


def alive(x):
    return x.poll() is None


processes = []
for s in settings:
    for c in s["costs"]:
        command = f"python -Wi main.py --dataset {s['dataset']} --model {s['model']} --fairness_condition {s['fairness_condition']} --cost {c} --tqdm No"
        cmd = shlex.split(command)
        while len(processes) >= MAX_JOBS:
            processes = list(filter(alive, processes))
            sleep(5)
        # print(cmd)
        p = subprocess.Popen(
            cmd,
            stdout=open(
                f"{s['dataset']}_{s['model']}_{s['fairness_condition']}_{c}.txt", "w"
            ),
        )
        processes.append(p)
        sleep(5)

while len(processes) != 0:
    processes = list(filter(alive, processes))
    sleep(5)
print("Finished")
