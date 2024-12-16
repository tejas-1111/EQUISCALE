"""Dataset and Preprocessing file"""

from typing import Literal, Dict, Tuple

import numpy as np
import numpy.typing as npt
import torch
import torch.utils.data


class CustomDataset(torch.utils.data.Dataset):
    """
    Implementation for dataset.

    Attributes:
        x:
          Feature vectors of all samples. Float. (num_samples, feature dimension)
        y:
          Labels of all samples. Int. (num_samples)
        z:
          Sensitive attribute of all samples. Int. (num_samples)
    """

    def __init__(
        self,
        x: npt.NDArray[np.float32],
        y: npt.NDArray[np.int64],
        z: npt.NDArray[np.int64],
        encodings: Tuple[Literal["One-Hot", "Embedding"], Dict[int, int]] | None,
        scaling: Literal["Standardize", "Normalize"] | None,
    ) -> None:
        super().__init__()
