from copy import deepcopy
from typing import Any
import argparse
import json
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

import dataset
import metrics
import models
import utils


def get_data(
    dataset: str,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Returns training, validation, and testing data for a single run.

    Fetches and generates the train-val-test split from the data.

    Args:
        dataset:
          Name of the dataset

    Returns:
        Tuple of (train data, val data, test data)
    """

    data = pd.read_csv(f"data/{dataset}.csv", header=None).to_numpy()
    np.random.shuffle(data)
    train_data = data[: int(data.shape[0] * 0.64)]
    val_data = data[int(data.shape[0] * 0.64) : int(data.shape[0] * 0.8)]
    test_data = data[int(data.shape[0] * 0.8) :]

    return train_data, val_data, test_data


def test_model(
    test_dataloader: torch.utils.data.DataLoader, model: models.RISAN | models.KP1, args
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

    preds = torch.empty(0, device=args.device)
    true = torch.empty(0, dtype=torch.long, device=args.device)
    sens = torch.empty(0, dtype=torch.long, device=args.device)
    model.eval()

    with torch.no_grad():
        for batch in tqdm(
            test_dataloader, desc="Testing", leave=False, disable=not args.tqdm
        ):
            x, y, z = batch
            x = x.to(args.device)
            y = y.to(args.device)
            z = z.to(args.device)

            preds = torch.concat((preds, model.infer(x)))
            true = torch.concat((true, y))
            sens = torch.concat((sens, z))

        return (
            metrics.coverage(preds)
            | metrics.accuracy(preds, true)
            | metrics.independence_metrics(preds, sens)
            | metrics.separation_metrics(preds, true, sens)
        )


def train_model(
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: torch.utils.data.DataLoader,
    args,
) -> models.RISAN | models.KP1:
    """
    Trains a model and returns it.

    Args:
        train_dataloader:
          Dataloader containing train samples
        val_dataloader:
          Dataloader containing validation samples.

    Returns:
      A trained model
    """
    if args.model == "risan":
        model = models.RISAN().to(args.device)
        loss_fn = metrics.DoubleSigmoidLoss(args.cost, args.gamma)
    else:
        model = models.KP1().to(args.device)
        loss_fn = metrics.GeneralizedCrossEntropy(args.cost, args.gamma)

    with torch.no_grad():
        model.eval()
        model(train_dataloader.dataset[0][0].view(1, -1).to(args.device))
        model.train()
    model_optimizer = torch.optim.Adam(model.parameters(), fused=True, lr=args.lr1)
    lambdas = (
        torch.nn.Parameter(
            torch.zeros(len(args.fairness_conditions), device=args.device)
        )
        if args.fairness_conditions != "none"
        else None
    )
    lambdas_optimizer = (
        torch.optim.Adam([lambdas], fused=True, maximize=True, lr=args.lr2)
        if lambdas is not None
        else None
    )
    lambdas_fn = (
        metrics.FairnessConstraints(args.fairness_conditions)
        if args.fairness_conditions is not None
        else None
    )

    best_model = deepcopy(model)
    best_loss = float("inf")
    last_improvement = 0

    for epoch_num in tqdm(
        range(args.epochs), desc="Epochs", leave=False, disable=not args.tqdm
    ):
        model.train()
        num_batches = 0
        for batch in tqdm(
            train_dataloader, desc="Training", leave=False, disable=not args.tqdm
        ):
            num_batches += 1
            x, y, z = batch
            x = x.to(args.device)
            y = y.to(args.device)
            z = z.to(args.device)

            probs = model.probs(x)
            if isinstance(model, models.RISAN) and isinstance(
                loss_fn, metrics.DoubleSigmoidLoss
            ):
                f, rho = model(x)
                model_loss: torch.Tensor = loss_fn(f, rho, y)
            elif isinstance(model, models.KP1) and isinstance(
                loss_fn, metrics.GeneralizedCrossEntropy
            ):
                out = model(x)
                model_loss: torch.Tensor = loss_fn(out, y)
            else:
                print("Train model error: [1]")
                exit()

            if lambdas is not None and lambdas_fn is not None:
                fairness_loss: torch.Tensor = (
                    lambdas * lambdas_fn(r=probs, true=y, sens=z)
                ).sum()
            else:
                fairness_loss = torch.zeros_like(model_loss)

            model_optimizer.zero_grad()
            (model_loss + fairness_loss).backward()
            model_optimizer.step()

        model.eval()
        if (
            lambdas is not None
            and lambdas_fn is not None
            and lambdas_optimizer is not None
        ):
            for batch in tqdm(
                train_dataloader,
                desc="Updating lambdas",
                leave=False,
                disable=not args.tqdm,
            ):
                x, y, z = batch
                x = x.to(args.device)
                y = y.to(args.device)
                z = z.to(args.device)
                probs = model.probs(x)
                fairness_loss: torch.Tensor = (
                    lambdas * lambdas_fn(r=probs, true=y, sens=z)
                ).sum()
                lambdas_optimizer.zero_grad()
                fairness_loss.backward()
                lambdas_optimizer.step()
                with torch.no_grad():
                    lambdas.clamp_(min=0)

        obj = torch.zeros(1, device=args.device, requires_grad=False)
        num_batches = 0
        with torch.no_grad():
            for batch in tqdm(
                val_dataloader, desc="Validating", leave=False, disable=not args.tqdm
            ):
                num_batches += 1
                x, y, z = batch
                x = x.to(args.device)
                y = y.to(args.device)
                z = z.to(args.device)

                probs = model.probs(x)
                if isinstance(model, models.RISAN) and isinstance(
                    loss_fn, metrics.DoubleSigmoidLoss
                ):
                    f, rho = model(x)
                    model_loss: torch.Tensor = loss_fn(f, rho, y)
                elif isinstance(model, models.KP1) and isinstance(
                    loss_fn, metrics.GeneralizedCrossEntropy
                ):
                    out = model(x)
                    model_loss: torch.Tensor = loss_fn(out, y)
                else:
                    print("Train model error: [1]")
                    exit()

                if lambdas is not None and lambdas_fn is not None:
                    fairness_loss: torch.Tensor = (
                        lambdas * lambdas_fn(r=probs, true=y, sens=z)
                    ).sum()
                else:
                    fairness_loss = torch.zeros_like(model_loss)

                obj += (model_loss + fairness_loss).detach()

        obj /= num_batches
        if obj.item() < 0.9999 * best_loss:
            best_loss = obj.item()
            best_model = deepcopy(model)
            last_improvement = 0
        else:
            last_improvement += 1
            if last_improvement == 10:
                break

    return best_model


def run(args) -> dict[str, torch.Tensor]:
    """
    Returns the results of a single run as a dictionary.
    """
    train_data, val_data, test_data = get_data(args.dataset)
    train_x = train_data[:, :-1].astype(np.float32)
    train_y = train_data[:, -1].astype(np.int64)
    train_z = train_data[:, -2].astype(np.int64)
    val_x = val_data[:, :-1].astype(np.float32)
    val_y = val_data[:, -1].astype(np.int64)
    val_z = val_data[:, -2].astype(np.int64)
    test_x = test_data[:, :-1].astype(np.float32)
    test_y = test_data[:, -1].astype(np.int64)
    test_z = test_data[:, -2].astype(np.int64)

    train_x, val_x, test_x = dataset.data_processing(train_x, val_x, test_x)
    train_dataset = dataset.CustomDataset(train_x, train_y, train_z)
    val_dataset = dataset.CustomDataset(val_x, val_y, val_z)
    test_dataset = dataset.CustomDataset(test_x, test_y, test_z)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        persistent_workers=True,
        pin_memory=True,
        prefetch_factor=2,
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        persistent_workers=True,
        pin_memory=True,
        prefetch_factor=2,
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        persistent_workers=False,
        pin_memory=True,
        prefetch_factor=2,
    )

    model = train_model(train_dataloader, val_dataloader, args)
    run_results = test_model(test_dataloader, model, args)
    return run_results


def main() -> None:
    """
    Side effects:
    1) Creates and writes to f"outputs/{args.dataset}_{args.fairness_conditions}_{args.model}_{args.cost}_{args.gamma}_{args.lr1}_{args.lr2}"
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=["risan", "kp1"], required=True)
    parser.add_argument("--cost", type=float, required=True)
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["adult", "bank", "compas", "default", "german"],
        required=True,
    )
    parser.add_argument("--gamma", type=float, required=True)
    parser.add_argument("--lr1", type=float, default=1e-3)
    parser.add_argument("--lr2", type=float, default=1e-3)
    parser.add_argument("--tqdm", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--fairness_conditions",
        nargs="*",
        choices=["pos", "neg", "abs", "tpr", "fpr", "tnr", "fnr", "par", "nar"],
    )

    args = parser.parse_args()
    if args.fairness_conditions is not None:
        args.fairness_conditions.sort()
    else:
        args.fairness_conditions = "none"

    os.makedirs("outputs/", exist_ok=True)
    args.epochs = 10000
    args.batch_size = 2048
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    returns: list[dict[str, torch.Tensor]] = []
    for run_num in tqdm(range(25), desc="Runs", leave=False, disable=not args.tqdm):
        returns.append(run(args))
    combined_results = utils.combine_results(returns)
    if isinstance(args.fairness_conditions, list):
        args.fairness_conditions = "-".join(args.fairness_conditions)
    with open(
        f"outputs/{args.dataset}_{args.fairness_conditions}_{args.model}_{args.cost:f}_{args.gamma:f}_{args.lr1:f}_{args.lr2:f}.json",
        "w",
    ) as f:
        f.write(json.dumps(combined_results, indent=4))


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    torch.random.manual_seed(42)
    main()
