"""Contains loss functions and performance metrics."""

from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleSigmoidLoss(nn.Module):
    """
    Double sigmoid loss for RISAN.

    Attributes:
        cost:
          Cost of abstention/rejection
        gamma:
          Hyperparameter controlling the slope of the loss.
    """

    def __init__(self, cost: float, gamma: float) -> None:
        """
        Initialize double sigmoid loss for a given cost.

        Args:
            cost:
              Cost of abstention/rejection. Should be between (0, 0.5) exclusive.
            gamma:
              Hyperparameter controlling the slope of the loss. Should be > 0.
        """
        super().__init__()
        self.cost = cost
        self.gamma = gamma

    def forward(
        self, f_x: torch.Tensor, rho_x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """
        For more details, refer to the RISAN paper.

        Args:
            f_x:
              Tensor with shape (batch_size, 1).
            rho_x:
              Tensor with shape (batch_size, 1). All elements of rho_x should be
              positive.
            y:
              (int) Tensor with shape (batch_size). All elements of y should be
              0/1, since they will be converted to -1/1 here.

        Returns:
            A 0-dimension tensor denoting the average of the double sigmoid loss
            over all the samples in the batch
        """

        y = (2 * y - 1).float().view(-1, 1)
        return torch.mean(
            2 * self.cost * torch.sigmoid(-self.gamma * (y * f_x - rho_x))
            + 2 * (1 - self.cost) * torch.sigmoid(-self.gamma * (y * f_x + rho_x))
        )


class GeneralizedCrossEntropy(nn.Module):
    """
    Generalized cross entropy for training the model accoring to the paper
    "Generalizing Consistent Multi-Class Classification with Rejection
    to be Compatible with Arbitrary Losses"

    Arguments:
        cost:
          Cost of abstention/rejection
        gamma:
          Number denoting the trade-off between cross entropy and mean absolute
          error. Refer to the paper for more information.
    """

    def __init__(self, cost: float, gamma: float) -> None:
        """
        Initializes GCE.

        Args:
            cost:
              Cost of abstention/rejection
            gamma:
              Number denoting the trade-off between cross entropy and mean absolute
              error. Should belong to (0, 1], with 0 being exclusive and 1 inclusive.
        """
        super().__init__()
        self.gamma = gamma
        self.cost = cost

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:
              Tensor of shape (batch_size, *) where the last dimension is the liklihood
              for abstention.
            y:
              (int) Tensor of shape (batchsize) where class k+1 denotes abstention for
              a k-class classification with abstention option setting.

        Returns:
            A 0-dimension tensor denoting the average of the GCE over all the
            samples in the batch
        """
        x = F.softmax(x, dim=1)

        t = torch.gather(x, 1, torch.unsqueeze(y, 1))
        l1 = (1 - (t**self.gamma)) / self.gamma

        t = torch.gather(x, 1, torch.unsqueeze(torch.full_like(y, 2), 1))
        l2 = (1 - (t**self.gamma)) / self.gamma

        return torch.mean(l1 + (1 - self.cost) * l2)


class FairnessConstraints(nn.Module):
    """
    Class which implements fairness constraints as subclasses
    """

    class EQ_Pos(nn.Module):
        """
        Equal positive rate
        """

        def __init__(self):
            super().__init__()

        def forward(self, **kwargs) -> torch.Tensor:
            r = kwargs["r"]
            sens = kwargs["sens"].float()
            group_0_total = (sens == 0).sum() + 1e-18
            group_1_total = (sens == 1).sum() + 1e-18

            return torch.abs(
                (r[:, 1] @ (1 - sens) / group_0_total)
                - (r[:, 1] @ sens / group_1_total)
            )

    class EQ_Neg(nn.Module):
        """
        Equal negative rate
        """

        def __init__(self):
            super().__init__()

        def forward(self, **kwargs) -> torch.Tensor:
            r = kwargs["r"]
            sens = kwargs["sens"].float()
            group_0_total = (sens == 0).sum() + 1e-18
            group_1_total = (sens == 1).sum() + 1e-18

            return torch.abs(
                (r[:, 0] @ (1 - sens) / group_0_total)
                - (r[:, 0] @ sens / group_1_total)
            )

    class EQ_Abs(nn.Module):
        """
        Equal abstention rate
        """

        def __init__(self):
            super().__init__()

        def forward(self, **kwargs) -> torch.Tensor:
            r = kwargs["r"]
            sens = kwargs["sens"].float()
            group_0_total = (sens == 0).sum() + 1e-18
            group_1_total = (sens == 1).sum() + 1e-18

            return torch.abs(
                (r[:, 2] @ (1 - sens) / group_0_total)
                - (r[:, 2] @ sens / group_1_total)
            )

    class EQ_TPR(nn.Module):
        """
        Equal true positive rate
        """

        def __init__(self):
            super().__init__()

        def forward(self, **kwargs) -> torch.Tensor:
            r = kwargs["r"]
            true = kwargs["true"]
            sens = kwargs["sens"].float()
            group_0_y_1 = ((sens == 0) * (true == 1)).sum() + 1e-18
            group_1_y_1 = ((sens == 1) * (true == 1)).sum() + 1e-18

            return torch.abs(
                (r[:, 1].T @ ((1 - sens) * (true))) / group_0_y_1
                - (r[:, 1].T @ (sens * (true))) / group_1_y_1
            )

    class EQ_FNR(nn.Module):
        """
        Equal false negative rate
        """

        def __init__(self):
            super().__init__()

        def forward(self, **kwargs) -> torch.Tensor:
            r = kwargs["r"]
            true = kwargs["true"]
            sens = kwargs["sens"].float()
            group_0_y_1 = ((sens == 0) * (true == 1)).sum() + 1e-18
            group_1_y_1 = ((sens == 1) * (true == 1)).sum() + 1e-18

            return torch.abs(
                (r[:, 0].T @ ((1 - sens) * (true))) / group_0_y_1
                - (r[:, 0].T @ (sens * (true))) / group_1_y_1
            )

    class EQ_PAR(nn.Module):
        """
        Equal positive abstention rate
        """

        def __init__(self):
            super().__init__()

        def forward(self, **kwargs) -> torch.Tensor:
            r = kwargs["r"]
            true = kwargs["true"]
            sens = kwargs["sens"].float()
            group_0_y_1 = ((sens == 0) * (true == 1)).sum() + 1e-18
            group_1_y_1 = ((sens == 1) * (true == 1)).sum() + 1e-18

            return torch.abs(
                (r[:, 2].T @ ((1 - sens) * (true))) / group_0_y_1
                - (r[:, 2].T @ (sens * (true))) / group_1_y_1
            )

    class EQ_FPR(nn.Module):
        """
        Equal false positive rate
        """

        def __init__(self):
            super().__init__()

        def forward(self, **kwargs) -> torch.Tensor:
            r = kwargs["r"]
            true = kwargs["true"]
            sens = kwargs["sens"].float()
            group_0_y_0 = ((sens == 0) * (true == 0)).sum() + 1e-18
            group_1_y_0 = ((sens == 1) * (true == 0)).sum() + 1e-18

            return torch.abs(
                (r[:, 1].T @ ((1 - sens) * (1 - true))) / group_0_y_0
                - (r[:, 1].T @ (sens * (1 - true))) / group_1_y_0
            )

    class EQ_TNR(nn.Module):
        """
        Equal true negative rate
        """

        def __init__(self):
            super().__init__()

        def forward(self, **kwargs) -> torch.Tensor:
            r = kwargs["r"]
            true = kwargs["true"]
            sens = kwargs["sens"].float()
            group_0_y_0 = ((sens == 0) * (true == 0)).sum() + 1e-18
            group_1_y_0 = ((sens == 1) * (true == 0)).sum() + 1e-18

            return torch.abs(
                (r[:, 0].T @ ((1 - sens) * (1 - true))) / group_0_y_0
                - (r[:, 0].T @ (sens * (1 - true))) / group_1_y_0
            )

    class EQ_NAR(nn.Module):
        """
        Equal negative abstention rate
        """

        def __init__(self):
            super().__init__()

        def forward(self, **kwargs) -> torch.Tensor:
            r = kwargs["r"]
            true = kwargs["true"]
            sens = kwargs["sens"].float()
            group_0_y_0 = ((sens == 0) * (true == 0)).sum() + 1e-18
            group_1_y_0 = ((sens == 1) * (true == 0)).sum() + 1e-18

            return torch.abs(
                (r[:, 2].T @ ((1 - sens) * (1 - true))) / group_0_y_0
                - (r[:, 2].T @ (sens * (1 - true))) / group_1_y_0
            )

    def __init__(self, conditions: list[str]) -> None:
        super().__init__()
        self.names = deepcopy(conditions)
        self.conditions = []
        if "pos" in conditions:
            self.conditions.append(self.EQ_Pos())
        if "neg" in conditions:
            self.conditions.append(self.EQ_Neg())
        if "abs" in conditions:
            self.conditions.append(self.EQ_Abs())
        if "tpr" in conditions:
            self.conditions.append(self.EQ_TPR())
        if "fnr" in conditions:
            self.conditions.append(self.EQ_FNR())
        if "par" in conditions:
            self.conditions.append(self.EQ_PAR())
        if "fpr" in conditions:
            self.conditions.append(self.EQ_FPR())
        if "tnr" in conditions:
            self.conditions.append(self.EQ_TNR())
        if "nar" in conditions:
            self.conditions.append(self.EQ_NAR())

    def forward(self, r, true, sens) -> torch.Tensor:
        return torch.stack(
            [condition(r=r, true=true, sens=sens) for condition in self.conditions], dim=0
        )


def coverage(pred: torch.Tensor) -> dict[str, torch.Tensor]:
    """
    Calculates the coverage of a model.

    Args:
        pred:
          A (int) tensor of shape (num_samples) containing the predictions of the model.
          Prediction -1 denotes abstention.

    Returns:
        A dictionary with key "cov" containing a tensor with the coverage of the model.
    """
    return {"cov": 100 * (pred != -1).sum() / pred.shape[0]}


def accuracy(pred: torch.Tensor, true: torch.Tensor) -> dict[str, torch.Tensor]:
    """
    Calculates the accuracy of a model. The accuracy is only calculated on unabstained
    samples.

    Args:
        pred:
          A (int) tensor of shape (num_samples) containing the predictions of the model.
          Prediction -1 denotes abstention.
        true:
          A (int) tensor of shape (num_samples) containing the labels of the samples.
    Returns:
        A dictionary with key "acc" containing a tensor with the accuracy of the model.
    """
    return {
        "acc": 100
        * ((pred != -1) * (pred == true)).sum()
        / ((pred != -1).sum() + 1e-18)
    }


def independence_metrics(
    pred: torch.Tensor, sens: torch.Tensor
) -> dict[str, torch.Tensor]:
    """
    Calculates the prediction rates for independence metrics.

    Args:
        pred:
          A (int) tensor of shape (num_samples) containing the predictions of the model.
          Prediction -1 denotes abstention.
        sens:
          A (int) tensor of shape (num_samples) containing the sensitive attribute of the
          samples.

    Returns:
        A dictionary with 6 keys of the form {a}_{b}, each with a tensor denoting a
        particular rate. Possible {a} values are abs, pos, and neg denoting
        abstention, positive and negative rates respectivley. Possible {b} values are
        0 or 1, denoting the value of the sensitive attribute.
    """
    group_0_total = (sens == 0).sum() + 1e-18
    group_1_total = (sens == 1).sum() + 1e-18

    return {
        "pos_0": 100 * ((sens == 0) * (pred == 1)).sum() / group_0_total,
        "pos_1": 100 * ((sens == 1) * (pred == 1)).sum() / group_1_total,
        "neg_0": 100 * ((sens == 0) * (pred == 0)).sum() / group_0_total,
        "neg_1": 100 * ((sens == 1) * (pred == 0)).sum() / group_1_total,
        "abs_0": 100 * ((sens == 0) * (pred == -1)).sum() / group_0_total,
        "abs_1": 100 * ((sens == 1) * (pred == -1)).sum() / group_1_total,
    }


def separation_metrics(
    pred: torch.Tensor, true: torch.Tensor, sens: torch.Tensor
) -> dict[str, torch.Tensor]:
    """
    Calculates the error rates for separation metrics.

    Args:
        pred:
          A tensor of shape (num_samples) containing the predictions of the model.
          Prediction -1 denotes abstention.
        true:
          A (int) tensor of shape (num_samples) containing the labels of the samples.
        sens:
          A (int) tensor of shape (num_samples) containing the sensitive attribute of the
          samples.

    Returns:
        A dictionary with 12 keys of the form {a}_{b}, each with a tensor denoting a
        particular rate. Possible {a} values are nar, tnr, fpr, par, fnr, tpr denoting
        negative abstention, true negative, false positive, positive abstention,
        false negative, and true positive rates respectively. Possible {b} values are
        0 or 1, denoting the value of the sensitive attribute.
    """
    group_0_y_0 = ((sens == 0) * (true == 0)).sum() + 1e-18
    group_0_y_1 = ((sens == 0) * (true == 1)).sum() + 1e-18
    group_1_y_0 = ((sens == 1) * (true == 0)).sum() + 1e-18
    group_1_y_1 = ((sens == 1) * (true == 1)).sum() + 1e-18

    return {
        "tpr_0": 100 * ((sens == 0) * (pred == 1) * (true == 1)).sum() / group_0_y_1,
        "tpr_1": 100 * ((sens == 1) * (pred == 1) * (true == 1)).sum() / group_1_y_1,
        "fnr_0": 100 * ((sens == 0) * (pred == 0) * (true == 1)).sum() / group_0_y_1,
        "fnr_1": 100 * ((sens == 1) * (pred == 0) * (true == 1)).sum() / group_1_y_1,
        "par_0": 100 * ((sens == 0) * (pred == -1) * (true == 1)).sum() / group_0_y_1,
        "par_1": 100 * ((sens == 1) * (pred == -1) * (true == 1)).sum() / group_1_y_1,
        "fpr_0": 100 * ((sens == 0) * (pred == 1) * (true == 0)).sum() / group_0_y_0,
        "fpr_1": 100 * ((sens == 1) * (pred == 1) * (true == 0)).sum() / group_1_y_0,
        "tnr_0": 100 * ((sens == 0) * (pred == 0) * (true == 0)).sum() / group_0_y_0,
        "tnr_1": 100 * ((sens == 1) * (pred == 0) * (true == 0)).sum() / group_1_y_0,
        "nar_0": 100 * ((sens == 0) * (pred == -1) * (true == 0)).sum() / group_0_y_0,
        "nar_1": 100 * ((sens == 1) * (pred == -1) * (true == 0)).sum() / group_1_y_0,
    }


def l0d1_loss(
    pred: torch.Tensor,
    true: torch.Tensor,
    cost: float,
) -> dict[str, torch.Tensor]:
    """
    Calculates the l0d1 loss for a model.

    Args:
        pred:
          A tensor of shape (num_samples) containing the predictions of the model.
          Prediction -1 denotes abstention.
        true:
          A (int) tensor of shape (num_samples) containing the labels of the samples.
        cost:
          Cost of abstention.

    Returns:
        A dictionary with key "l0d1" containing a tensor denoting the l0d1 loss
        of the model
    """
    return {
        "l0d1": (((pred != true) * (pred != -1)).sum() + cost * (pred == -1).sum())
        / pred.shape[0]
    }


def combine_results(
    results: list[dict[str, torch.Tensor]]
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """
    Combines a list of multiple results by calculating the average and standard deviation.

    Args:
        results:
          A list of results to combine by averaging.

    Returns:
        Returns a tuple (d0, d1) containing two dictionaries, where d0
        contains the averages and d1 contains the standard deviation.
    """
    avgs = {}
    std = {}
    for key in results[0].keys():
        avgs[key] = torch.mean(
            torch.stack([result[key] for result in results], dim=0), dim=0
        )
        std[key] = torch.std(
            torch.stack([result[key] for result in results], dim=0), dim=0
        )

    return avgs, std
