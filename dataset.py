"""Contains dataset, preprocessing, and dataloader classes"""

import numpy as np
import numpy.typing as npt
import torch
import torch.utils.data


class CustomDataset(torch.utils.data.Dataset):
    """
    Implementation for dataset.

    Attributes:
        x:
          Feature vectors of all the samples. NDArray[np.float32] of shape (num_samples, num_features)
        y:
          Labels of all the samples. NDArray[np.int64] of shape (num_samples)
        z:
          Sensitive attributes of all the samples. NDArray[np.int64] of shape (num_samples)
    """

    def __init__(
        self, x: npt.NDArray[np.float32], y: npt.NDArray[np.int64], z: npt.NDArray[np.int64]
    ) -> None:
        """
        Initialize a dataset with feature vectors, labels and sensitive attributes.

        Args:
            x:
              Feature vectors of all the samples. Should be a NDArray[np.float32]
              of shape (num_samples, num_features).
            y:
              Labels of all the samples. Should be either 0 or 1 for all the samples.
              Should be a NDArray[np.int64] of shape (num_samples).
            z:
              Sensitive attribute of all the samples. Should be either 0 or 1 for
              all the samples. Should be a NDArray[np.int64] of shape (num_samples).
        """
        self.x = x
        self.y = y
        self.z = z

    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.

        Returns:
            Single integer denoting the number of samples in the dataset.
        """
        return self.x.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns the feature vector, label and sensitive attribute of a sample.

        Args:
            index:
              Should be an interger between 0 and num_samples-1 inclusive.

        Returns:
            Return a tuple of 3 tensors (t1, t2, t3) where t1 is a tensor of shape
            (num_samples) denoting the feature vector, t2 is a 0-dimension (int) tensor
            denoting the label, and t3 is a 0-dimension (int) tensor denoting the sensitive
            attribute of the samples.
        """
        return (
            torch.tensor(self.x[index], dtype=torch.float),
            torch.tensor(self.y[index], dtype=torch.long),
            torch.tensor(self.z[index], dtype=torch.long),
        )


def data_processing(
    train_x: npt.NDArray[np.float32], test_x: npt.NDArray[np.float32]
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """
    Normalize the data.

    Normalizes the training data to the range [-1, 1] inclusive and then uses the
    training data features' maximum and minimum to normalize the test data as well.

    Args:
      train_x:
        Feature vectors from the training data. Should be a NDArray[np.float32]
        of shape (num_samples, num_features)
      test_x:
        Feature vectors from the test data. Should be a NDArray[np.float32]
        of shape (num_samples, num_features)

    Returns:
      Returns a tuple of two numpy arrays (a0, a1) where a0 is the normalized training
      data and a1 is the normalized test data, both with shape (num_samples, num_features)
    """
    mins = train_x.min(axis=0)
    maxs = train_x.max(axis=0)

    train_x = 2 * ((train_x - mins) / (maxs - mins + 1e-9)) - 1
    test_x = 2 * ((test_x - mins) / (maxs - mins + 1e-9)) - 1

    return train_x, test_x


# Taken from huggingface pytorch image models.
# https://github.com/huggingface/pytorch-image-models/blob/v0.9.16/timm/data/loader.py#L370C1-L402C42


class MultiEpochsDataLoader(torch.utils.data.DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        if self.batch_sampler is None:
            self.sampler = _RepeatSampler(self.sampler)
        else:
            self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return (
            len(self.sampler) # type: ignore
            if self.batch_sampler is None
            else len(self.batch_sampler.sampler) # type: ignore
        )

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """Sampler that repeats forever.

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)
