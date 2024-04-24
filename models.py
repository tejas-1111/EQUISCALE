import torch
import torch.nn as nn
import torch.nn.functional as F


class RISAN(nn.Module):
    """
    RISAN (from the paper RISAN: Robust instance specific deep abstention network)
    with input independent rho.
    """

    def __init__(self) -> None:
        super().__init__()

        self.fc_1 = nn.LazyLinear(32)
        self.b_1 = nn.LazyBatchNorm1d()
        self.d_1 = nn.Dropout1d()
        self.f = nn.LazyLinear(1)
        self.rho = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.f(self.d_1(F.relu(self.b_1(self.fc_1(x))))), F.softplus(self.rho)

    def infer(self, x: torch.Tensor) -> torch.Tensor:
        """
        Does the inference for a batch in the binary classification with
        abstention option setting.

        Args:
            x:
              A tensor of shape (batch_size, num_features) denoting a batch input
              to the model.

        Returns:
            A single (int) tensor of shape (batch_size) denoting the predictions
            for the batch. Prediction of -1 denotes abstention.
        """
        f, rho = self.__call__(x)
        f, rho = f.flatten(), rho.flatten()
        return (f > rho).long() - (torch.abs(f) <= rho).long()

    def probs(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns the classwise prediction probabilities for a sample.

        Args:
            x:
              A tensor of shape (batch_size, num_features) denoting a batch input
              to the model.

        Returns:
            A single tensor of shape (batch_size, 3) denoting the classwise prediction
            probabilities for the batch. The values at [:, 2] denote the probability
            of abstention.
        """
        f, rho = self.__call__(x)
        f, rho = f.flatten(), rho.flatten()
        r = F.softmax(
            torch.stack([-f - rho, f - rho, rho - torch.abs(f)], dim=1), dim=1
        )
        return r


class KP1(nn.Module):
    """
    Model trained using the calibrated loss function using generalized cross entropy
    from the paper "Generalizing Consistent Multi-Class Classification with Rejection
    to be Compatible with Arbitrary Losses".
    """

    def __init__(self):
        super().__init__()

        self.fc_1 = nn.LazyLinear(32)
        self.b_1 = nn.LazyBatchNorm1d()
        self.d_1 = nn.Dropout1d()
        self.out = nn.LazyLinear(3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.d_1(F.relu(self.b_1(self.fc_1(x)))))

    def infer(self, x: torch.Tensor) -> torch.Tensor:
        """
        Does the inference for a batch in the binary classification with
        abstention option setting.

        Args:
            x:
              A tensor of shape (batch_size, num_features) denoting a batch input
              to the model.

        Returns:
            A single (int) tensor of shape (batch_size) denoting the predictions
            for the batch. Prediction of -1 denotes abstention.
        """
        preds = torch.argmax(self.__call__(x), dim=1).flatten()
        preds = torch.where(preds != 2, preds, -1)
        return preds

    def probs(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns the classwise prediction probabilities for a sample.

        Args:
            x:
              A tensor of shape (batch_size, num_features) denoting a batch input
              to the model.

        Returns:
            A single tensor of shape (batch_size, 3) denoting the classwise prediction
            probabilities for the batch. The values at [:, 2] denote the probability
            of abstention.
        """
        return F.softmax(self.__call__(x), dim=1)