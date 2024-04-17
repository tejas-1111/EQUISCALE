from time import sleep
import shlex
import subprocess

MAX_JOBS = 8
USER = (str(subprocess.run("whoami", stdout=subprocess.PIPE).stdout))[2:-3]
print(USER)

settings = [
    {
        "model": "RISAN",
        "dataset": "Compas",
        "fairness_condition": "None",
        "costs": [
        ],
    },
]

for s in settings:
    for c in s["costs"]:
        command = f'sbatch run.sh {s["dataset"]} {s["model"]} {s["fairness_condition"]} {c}'
        cmd = shlex.split(command)

        while True:
            queue_cmd = shlex.split(f"squeue -u {USER}")
            result = subprocess.run(queue_cmd, stdout=subprocess.PIPE)
            output = str(result.stdout).split("\\n")
            if len(output) <= MAX_JOBS + 1:
                break
            else:
                sleep(30)

        print(cmd)
        subprocess.Popen(cmd)
        sleep(15)