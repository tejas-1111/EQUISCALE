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

        self.fc_1 = nn.LazyLinear(64)
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
    
    def tsne(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns learnt vectors for TSNE plots.

        Args:
            x:
              A tensor of shape (batch_size, num_features) denoting a batch input 
              to the model.

        Returns:
            A single tensor of shape (batch_size, 64) denoting the learn features for 
            samples.
        """

        return self.fc_1(x)


class KP1(nn.Module):
    """
    Model trained using the calibrated loss function using generalized cross entropy
    from the paper "Generalizing Consistent Multi-Class Classification with Rejection
    to be Compatible with Arbitrary Losses".
    """

    def __init__(self):
        super().__init__()

        self.fc_1 = nn.LazyLinear(64)
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
    
    def tsne(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns learnt vectors for TSNE plots.

        Args:
            x:
              A tensor of shape (batch_size, num_features) denoting a batch input 
              to the model.

        Returns:
            A single tensor of shape (batch_size, 64) denoting the learn features for 
            samples.
        """

        return self.fc_1(x)


class FNNC(nn.Module):
    """
    Model from the paper "FNNC: Achieving Fairness through Neural Networks". Note,
    we use the constraints proposed in our paper and keep the probability of abstention
    0 to achieve the same results.
    """

    def __init__(self) -> None:
        super().__init__()

        self.fc_1 = nn.LazyLinear(64)
        self.b_1 = nn.LazyBatchNorm1d()
        self.d_1 = nn.Dropout1d()
        self.fc_2 = nn.LazyLinear(1)

        self.rho: None | float = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.tanh(self.fc_2(self.d_1(F.relu(self.b_1(self.fc_1(x))))))

    def infer_bin(self, x: torch.Tensor) -> torch.Tensor:
        """
        Does the inference for a batfch in binary classification setting.

        Args:
            x:
              A tensor of shape (batch_size, num_features) denoting a batch input
              to the model.

        Returns:
            A single (int) tensor of shape (batch_size) denoting the predictions
            for the batch.
        """
        return ((self.__call__(x).flatten()) > 0).long()

    def infer(self, x: torch.Tensor):
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
        output = self.__call__(x).flatten()
        return (output > self.rho).long() - (torch.abs(output) <= self.rho).long()

    def probs(self, x: torch.Tensor):
        """
        Returns the classwise prediction probabilities for a sample.

        For FNNC, when doing binary classification (which is when fairness is trained),
        the probability of abstention is always 0.

        Args:
            x:
              A tensor of shape (batch_size, num_features) denoting a batch input
              to the model.

        Returns:
            A single tensor of shape (batch_size, 3) denoting the classwise prediction
            probabilities for the batch. The values at [:, 2] denote the probability
            of abstention.
        """
        output = (self.__call__(x).flatten() + 1) / 2
        return torch.stack((1 - output, output, torch.zeros_like(output)), dim=1)

    def tsne(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns learnt vectors for TSNE plots.

        Args:
            x:
              A tensor of shape (batch_size, num_features) denoting a batch input 
              to the model.

        Returns:
            A single tensor of shape (batch_size, 64) denoting the learn features for 
            samples.
        """

        return self.fc_1(x)