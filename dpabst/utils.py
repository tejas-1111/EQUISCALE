"""Contains utility functions"""

import torch


def combine_results(
    results: list[dict[str, torch.Tensor]],
) -> dict[str, tuple[float, float]]:
    """
    Combine results of multiple runs for saving by calculating average and 
    standard deviation of the metrics.

    Args:
        results:
          Dictionaries containing results from multiple runs.

    Returns:
        A dictionary which combines the same keys with a 2-tuple, where the first
        item is the average and the second item is the standard deviation of the
        metrics. 
    """
    combined_results = {}
    for key in results[0].keys():
        readings = torch.stack([result[key] for result in results])
        combined_results[key] = (torch.mean(readings).item(), torch.std(readings).item())
    return combined_results
