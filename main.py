from argparse import ArgumentParser
from copy import deepcopy
import json
import math
import os
import random

from tqdm import tqdm
import numpy.typing as npt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import torch.utils.tensorboard

import dataset
import metrics
import models

parser = ArgumentParser()
parser.add_argument("--model", type=str, choices=["RISAN", "KP1"], required=True)
parser.add_argument(
    "--fairness_condition",
    type=str,
    choices=["None", "Ind", "Sep", "Mixed"],
    required=True,
)
parser.add_argument("--cost", type=float, required=True)
parser.add_argument(
    "--dataset",
    type=str,
    choices=["Adult", "Bank", "Compas", "Default", "German"],
    required=True,
)
parser.add_argument("--tqdm", type=str, choices=["Yes", "No"], required=True)

args = parser.parse_args()
MODEL: str = (args.model).lower()
FAIRNESS_CONDITION: str = (args.fairness_condition).lower()
EPOCHS: int = 10000
DATASET: str = (args.dataset).lower()
if DATASET == "german" or DATASET == "compas":
    BATCH_SIZE: int = 256
else:
    BATCH_SIZE: int = 2048
COST: float = float(args.cost)
DISABLE_TQDM = False if args.tqdm == "Yes" else True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WRITER = torch.utils.tensorboard.SummaryWriter(  # type: ignore
    f"runs/{DATASET}/{MODEL}/{FAIRNESS_CONDITION}/{COST}"
)

if DATASET == "adult":
    if MODEL == "risan":
        GAMMA = 8
        LR1 = 1e-3
        LR2 = 1e-3
    elif MODEL == "kp1":
        GAMMA = 0.7
        LR1 = 1e-3
        LR2 = 1e-3
elif DATASET == "bank":
    if MODEL == "risan":
        GAMMA = 1
        LR1 = 1e-3
        LR2 = 1e-3
    elif MODEL == "kp1":
        GAMMA = 0.7
        LR1 = 1e-3
        LR2 = 1e-3
elif DATASET == "compas":
    if MODEL == "risan":
        GAMMA = 1
        LR1 = 1e-2
        LR2 = 1e-3
    elif MODEL == "kp1":
        GAMMA = 0.7
        LR1 = 1e-3
        LR2 = 1e-3
elif DATASET == "default":
    if MODEL == "risan":
        GAMMA = 6.8
        LR1 = 1e-3
        LR2 = 1e-3
    elif MODEL == "kp1":
        GAMMA = 0.7
        LR1 = 1e-3
        LR2 = 1e-3
elif DATASET == "german":
    if MODEL == "risan":
        GAMMA = 1.175
        LR1 = 1e-3
        LR2 = 1e-3
    elif MODEL == "kp1":
        GAMMA = 0.7
        LR1 = 1e-3
        LR2 = 1e-3


def train_epoch(
    run_num: int,
    fold_num: int,
    epoch_num: int,
    train_dataloader: dataset.MultiEpochsDataLoader,
    model: models.RISAN | models.KP1,
    model_loss_fn: (
        metrics.DoubleSigmoidLoss | metrics.GenralizedCrossEntropy | nn.NLLLoss
    ),
    model_optimizer: torch.optim.Adam,
    lambdas: torch.Tensor | None,
    fairness_loss_fn: (
        metrics.DemographicParity | metrics.EqualizedOdds | metrics.MixedDPandEO | None
    ),
    lambdas_optimizer: torch.optim.Adam | None,
):
    """
    Trains one epoch of the model.

    Args:
        run_num:
          Current run number
        fold_num:
          Current fold number
        epoch_num:
          Current epoch number
        train_dataloader:
          Dataloader containing train samples.
        model:
          Inital model or the model after previous epoch
        model_loss_fn:
          The loss function to be used for training the model.
        model_optimimzer:
          The optimizer for the model.
        lambdas:
          None if fairness is not required, else a tensor of shape
          (3) for demographic parity, (6) for equalized odds, or (3) for
          mixed constraints.
        fairness_loss_fn:
          None if fairness is not required, else either demographic parity or
          equalized odds.
        lambdas optimizer:
          None if fairness is not required, else a maximizing optimizer for the
          lambas.
    """
    model.train()
    model_loss = torch.zeros(1, device=DEVICE)
    fairness_loss = torch.zeros(1, device=DEVICE)
    lambdas_loss = torch.zeros(1, device=DEVICE)

    for batch in tqdm(
        train_dataloader, desc="Training network", leave=False, disable=DISABLE_TQDM
    ):
        x, y, z = batch
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        z = z.to(DEVICE)

        probs = model.probs(x)
        if isinstance(model, models.RISAN) and isinstance(
            model_loss_fn, metrics.DoubleSigmoidLoss
        ):
            f, rho = model(x)
            model_loss_val: torch.Tensor = model_loss_fn(f, rho, y)
        elif isinstance(model, models.KP1) and isinstance(
            model_loss_fn, metrics.GenralizedCrossEntropy
        ):
            out = model(x)
            model_loss_val: torch.Tensor = model_loss_fn(out, y) + (
                1 - COST
            ) * model_loss_fn(out, torch.full_like(y, 2))
        else:
            print("train_epoch() error: [1]")
            exit()

        fairness_loss_val: torch.Tensor = (
            torch.zeros_like(model_loss_val, device=DEVICE)
            if lambdas is None or fairness_loss_fn is None
            else (lambdas * fairness_loss_fn(r=probs, true=y, sens=z)).sum()
        )

        loss = model_loss_val + fairness_loss_val
        model_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        model_optimizer.step()
        model_loss += model_loss_val.detach()
        fairness_loss += fairness_loss_val.detach()

    if (
        lambdas is not None
        and fairness_loss_fn is not None
        and lambdas_optimizer is not None
    ):
        for batch in tqdm(
            train_dataloader, desc="Updating lambdas", leave=False, disable=DISABLE_TQDM
        ):
            x, y, z = batch
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            z = z.to(DEVICE)

            probs = model.probs(x)
            loss: torch.Tensor = (
                lambdas * fairness_loss_fn(r=probs, true=y, sens=z)
            ).sum()
            lambdas_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            lambdas_optimizer.step()
            with torch.no_grad():
                lambdas.clamp_(min=0)
            lambdas_loss += loss.detach()

    model_loss /= len(train_dataloader)
    fairness_loss /= len(train_dataloader)
    lambdas_loss /= len(train_dataloader)

    WRITER.add_scalar(f"model loss/{run_num}/{fold_num}/train", model_loss, epoch_num)
    WRITER.add_scalar(
        f"fairness loss/{run_num}/{fold_num}/train", fairness_loss, epoch_num
    )
    WRITER.add_scalar(
        f"total loss/{run_num}/{fold_num}/train", model_loss + fairness_loss, epoch_num
    )
    WRITER.add_scalar(
        f"lambdas loss/{run_num}/{fold_num}/train", lambdas_loss, epoch_num
    )


def dev_epoch(
    run_num: int,
    fold_num: int,
    epoch_num: int,
    test_dataloader: dataset.MultiEpochsDataLoader,
    model: models.RISAN | models.KP1,
    model_loss_fn: (
        metrics.DoubleSigmoidLoss | metrics.GenralizedCrossEntropy | nn.NLLLoss
    ),
    lambdas: torch.Tensor | None,
    fairness_loss_fn: (
        metrics.DemographicParity | metrics.EqualizedOdds | metrics.MixedDPandEO | None
    ),
) -> torch.Tensor:
    """
    Runs one dev epoch for the model.

    Args:
        test_dataloader:
          Dataloader containing test samples.
        model:
          Inital model or the model after previous epoch
        model_loss_fn:
          The loss function used for training the model.
        lambdas:
          None if fairness is not required, else a tensor of shape (3) for demographic
          demographic parity, (6) for equalized odds or (3) for mixed constraints.
        fairness_loss_fn:
          None if fairness is not required, else either demographic parity,
          equalized odds, or mixed constraints.
        dev_loss:
          Contains value of the epoch loss after the epoch is finished
    """
    model.eval()
    model_loss = torch.zeros(1, device=DEVICE)
    fairness_loss = torch.zeros(1, device=DEVICE)

    preds = torch.empty(0, device=DEVICE)
    true = torch.empty(0, dtype=torch.long, device=DEVICE)
    sens = torch.empty(0, dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        for batch in tqdm(
            test_dataloader, desc="Dev loop", leave=False, disable=DISABLE_TQDM
        ):
            x, y, z = batch
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            z = z.to(DEVICE)
            true = torch.concat((true, y))
            sens = torch.concat((sens, z))

            probs = model.probs(x)
            if isinstance(model, models.RISAN) and isinstance(
                model_loss_fn, metrics.DoubleSigmoidLoss
            ):
                f, rho = model(x)
                model_loss_val: torch.Tensor = model_loss_fn(f, rho, y)
                preds = torch.concat((preds, model.infer(x)))
            elif isinstance(model, models.KP1) and isinstance(
                model_loss_fn, metrics.GenralizedCrossEntropy
            ):
                out = model(x)
                model_loss_val: torch.Tensor = model_loss_fn(out, y) + (
                    1 - COST
                ) * model_loss_fn(out, torch.full_like(y, 2))
                preds = torch.concat((preds, model.infer(x)))
            else:
                print("dev_epoch() error: [1]")
                exit()

            fairness_loss_val: torch.Tensor = (
                torch.zeros_like(model_loss_val, device=DEVICE)
                if lambdas is None or fairness_loss_fn is None
                else (lambdas * fairness_loss_fn(r=probs, true=y, sens=z)).sum()
            )
            model_loss += model_loss_val.detach()
            fairness_loss += fairness_loss_val.detach()

        model_loss /= len(test_dataloader)
        fairness_loss /= len(test_dataloader)

        results = (
            metrics.coverage(preds)
            | metrics.accuracy(preds, true)
            | metrics.independence_metrics(preds, sens)
            | metrics.separation_metrics(preds, true, sens)
            | metrics.l0d1_loss(preds, true, COST)
        )

        WRITER.add_scalar(f"cov/{run_num}/{fold_num}/dev", results["cov"], epoch_num)
        WRITER.add_scalar(f"acc/{run_num}/{fold_num}/dev", results["acc"], epoch_num)
        WRITER.add_scalar(f"l0d1/{run_num}/{fold_num}/dev", results["l0d1"], epoch_num)
        WRITER.add_scalar(f"model loss/{run_num}/{fold_num}/dev", model_loss, epoch_num)
        WRITER.add_scalar(
            f"fairness loss/{run_num}/{fold_num}/dev", fairness_loss, epoch_num
        )
        WRITER.add_scalar(
            f"total loss/{run_num}/{fold_num}/dev",
            model_loss + fairness_loss,
            epoch_num,
        )

        for metric in ["neg", "pos", "abs", "tnr", "fpr", "nar", "fnr", "tpr", "par"]:
            WRITER.add_scalar(
                f"{metric}/{run_num}/{fold_num}/dev/group 0",
                results[f"{metric}_0"],
                epoch_num,
            )
            WRITER.add_scalar(
                f"{metric}/{run_num}/{fold_num}/dev/group 1",
                results[f"{metric}_1"],
                epoch_num,
            )
            WRITER.add_scalar(
                f"{metric}/{run_num}/{fold_num}/dev/delta",
                torch.abs(results[f"{metric}_0"] - results[f"{metric}_1"]),
                epoch_num,
            )
        if lambdas is not None:
            for idx in range(lambdas.shape[0]):
                WRITER.add_scalar(
                    f"lambda_{idx}/{run_num}/{fold_num}/dev", lambdas[idx], epoch_num
                )

        return model_loss + fairness_loss


def test(
    test_dataloader: dataset.MultiEpochsDataLoader,
    model: models.RISAN | models.KP1,
) -> dict[str, torch.Tensor]:
    """
    Tests the model and calculates its metrics.

    Args:
        test_dataloader:
          Dataloader containing test samples.
        model:
          Model to be tested

    Returns:
        Returns a dictionary containing the model's performance metrics.
    """
    if DISABLE_TQDM:
        print("Started testing", flush=True)

    preds = torch.empty(0, device=DEVICE)
    true = torch.empty(0, dtype=torch.long, device=DEVICE)
    sens = torch.empty(0, dtype=torch.long, device=DEVICE)
    model.eval()

    with torch.no_grad():
        for batch in tqdm(
            test_dataloader, desc="Testing", leave=False, disable=DISABLE_TQDM
        ):
            x, y, z = batch
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            z = z.to(DEVICE)

            preds = torch.concat((preds, model.infer(x)))
            true = torch.concat((true, y))
            sens = torch.concat((sens, z))

        results = (
            metrics.coverage(preds)
            | metrics.accuracy(preds, true)
            | metrics.independence_metrics(preds, sens)
            | metrics.separation_metrics(preds, true, sens)
            | metrics.l0d1_loss(preds, true, COST)
        )

        return results


def train_one_fold(
    run_num: int,
    fold_num: int,
    train_data: npt.NDArray[np.float64],
    test_data: npt.NDArray[np.float64],
) -> dict[str, torch.Tensor]:
    """
    Trains one fold of the 5-fold cross-validation.

    Args:
        run_num:
          Current run number
        fold_num:
          Current fold number
        train_data:
          Numeric train data such that the second last column is the sensitive attribute
          with values 0 or 1 and the last column is the label with values 0 or 1.
        test_data:
          Numeric test data such that the second last column is the sensitive attribute
          with values 0 or 1 and the last column is the label with values 0 or 1.

    Returns:
        Returns a dictionary containing the model's performance metrics.
    """
    if DISABLE_TQDM:
        print(f"Fold number {fold_num} started", flush=True)

    train_x = train_data[:, :-1].astype(np.float32)
    train_y = train_data[:, -1].astype(np.int64)
    train_z = train_data[:, -2].astype(np.int64)
    test_x = test_data[:, :-1].astype(np.float32)
    test_y = test_data[:, -1].astype(np.int64)
    test_z = test_data[:, -2].astype(np.int64)

    train_x, test_x = dataset.data_processing(train_x, test_x)
    train_dataset = dataset.CustomDataset(train_x, train_y, train_z)
    test_dataset = dataset.CustomDataset(test_x, test_y, test_z)
    train_dataloader = dataset.MultiEpochsDataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        persistent_workers=True,
        pin_memory=True,
    )
    test_dataloader = dataset.MultiEpochsDataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=2,
        persistent_workers=True,
        pin_memory=True,
    )

    if MODEL == "risan":
        model = models.RISAN().to(DEVICE)
        model_loss_fn = metrics.DoubleSigmoidLoss(COST, GAMMA)
    elif MODEL == "kp1":
        model = models.KP1().to(DEVICE)
        model_loss_fn = metrics.GenralizedCrossEntropy(GAMMA)

    with torch.no_grad():
        model.eval()
        model(train_dataset[0][0].view(1, -1).to(DEVICE))
        model.train()

    model_optimizer = torch.optim.Adam(model.parameters(), fused=True, lr=LR1)

    if FAIRNESS_CONDITION == "ind":
        fairness_loss_fn = metrics.DemographicParity()
        lambdas = torch.nn.Parameter(torch.zeros(3, device=DEVICE))
        lambdas_optimizer = torch.optim.Adam([lambdas], fused=True, maximize=True, lr=LR2)
    elif FAIRNESS_CONDITION == "sep":
        fairness_loss_fn = metrics.EqualizedOdds()
        lambdas = torch.nn.Parameter(torch.zeros(6, device=DEVICE))
        lambdas_optimizer = torch.optim.Adam([lambdas], fused=True, maximize=True, lr=LR2)
    elif FAIRNESS_CONDITION == "mixed":
        fairness_loss_fn = metrics.MixedDPandEO()
        lambdas = torch.nn.Parameter(torch.zeros(3, device=DEVICE))
        lambdas_optimizer = torch.optim.Adam([lambdas], fused=True, maximize=True, lr=LR2)
    else:
        fairness_loss_fn = None
        lambdas = None
        lambdas_optimizer = None

    best_model = deepcopy(model)
    best_loss = math.inf
    last_improvement = 0

    for epoch_num in tqdm(
        range(EPOCHS), desc="Epochs", leave=False, disable=DISABLE_TQDM
    ):
        train_epoch(
            run_num,
            fold_num,
            epoch_num,
            train_dataloader,
            model,
            model_loss_fn,
            model_optimizer,
            lambdas,
            fairness_loss_fn,
            lambdas_optimizer,
        )

        dev_loss = dev_epoch(
            run_num,
            fold_num,
            epoch_num,
            test_dataloader,
            model,
            model_loss_fn,
            lambdas,
            fairness_loss_fn,
        )

        if dev_loss.item() < best_loss:
            best_loss = dev_loss.item()
            best_model = deepcopy(model)
            last_improvement = 0
        else:
            last_improvement += 1
            if last_improvement == 10:
                break

    best_model.eval()
    results = test(test_dataloader, best_model)
    os.makedirs(f"models/{DATASET}/{MODEL}/{FAIRNESS_CONDITION}/", exist_ok=True)
    torch.save(
        best_model.state_dict(),
        f"models/{DATASET}/{MODEL}/{FAIRNESS_CONDITION}/{COST:f}_{run_num}_{fold_num}.pt",
    )

    os.makedirs(
        f"outputs/{DATASET}/{MODEL}/{FAIRNESS_CONDITION}/{COST:f}/", exist_ok=True
    )
    temp = {}
    for key in results.keys():
        temp[key] = results[key].tolist()
    with open(
        f"outputs/{DATASET}/{MODEL}/{FAIRNESS_CONDITION}/{COST:f}/{run_num}_{fold_num}.json",
        "w",
    ) as f:
        f.write(json.dumps(temp, indent=4))

    return results


def train_one_run(
    run_num: int,
    train_data: npt.NDArray[np.float64],
    test_data: npt.NDArray[np.float64] | None,
) -> dict[str, torch.Tensor]:
    """
    Completes one run of the five runs of the model. Uses 5-fold cross-validation
    if test data is not provided.

    Args:
        run_num:
          Current run number
        train_data:
          Numeric training data from the dataset.
        test_data:
          None if no separate test data is provided, else the provided test data.

    Returns:
        Returns a dictionary contaning the average of all the model's performance across
        the various runs.
    """

    if DISABLE_TQDM:
        print(f"Run number {run_num} started", flush=True)

    returns = []
    if test_data is None:
        data = train_data.copy()
        data = np.array_split(data, 5)
        for fold_num in tqdm(
            range(5), leave=False, desc="Fold number", disable=DISABLE_TQDM
        ):
            fold_train_data = np.concatenate(data[:fold_num] + data[fold_num + 1 :])
            returns.append(
                train_one_fold(run_num, fold_num, fold_train_data, data[fold_num])
            )
    else:
        returns.append(train_one_fold(run_num, 0, train_data, test_data))

    results_avgs = metrics.combine_results(returns, False)
    if isinstance(results_avgs, tuple):
        print("train_one_run() error: [1]")
        exit()

    return results_avgs


def main():
    """
    Loads data, start model training, and stores results.
    """
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    print(
        f"{MODEL}: DATASET={DATASET}, FAIR_COND = {FAIRNESS_CONDITION}, COST={COST}, DEVICE={DEVICE}",
        flush=True,
    )

    if DATASET == "adult":
        train_data = pd.read_csv("data/adult_train.csv", header=None).to_numpy()
        test_data = pd.read_csv("data/adult_test.csv", header=None).to_numpy()
    elif DATASET == "bank":
        train_data = pd.read_csv("data/bank.csv", header=None).to_numpy()
        test_data = None
    elif DATASET == "compas":
        train_data = pd.read_csv("data/compas.csv", header=None).to_numpy()
        test_data = None
    elif DATASET == "default":
        train_data = pd.read_csv("data/default.csv", header=None).to_numpy()
        test_data = None
    elif DATASET == "german":
        train_data = pd.read_csv("data/german.csv", header=None).to_numpy()
        test_data = None
    else:
        print("main() error: [1]")
        exit()

    returns = []
    for run_num in tqdm(range(5), desc="Runs", leave=False, disable=DISABLE_TQDM):
        returns.append(train_one_run(run_num, train_data, test_data))
    WRITER.flush()

    results_avgs, results_stds = metrics.combine_results(returns, True)
    if isinstance(results_avgs, str) or isinstance(results_stds, str):
        print("main() error: [2]")
        exit()

    results = {}
    for key in results_avgs.keys():
        results[key] = [results_avgs[key].tolist(), results_stds[key].tolist()]

    os.makedirs(
        f"results/{DATASET}/{MODEL}/{FAIRNESS_CONDITION}",
        exist_ok=True,
    )
    with open(
        f"results/{DATASET}/{MODEL}/{FAIRNESS_CONDITION}/{COST:f}.json", "w"
    ) as f:
        f.write(json.dumps(results, indent=4))


if __name__ == "__main__":
    main()
