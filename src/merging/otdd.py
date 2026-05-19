import logging
import os

import matplotlib
import torch

## Local Imports

if os.name == "posix" and "DISPLAY" not in os.environ:
    matplotlib.use("Agg")
    nodisplay = True
else:
    nodisplay = False


logger = logging.getLogger(__name__)

try:
    pass
except ImportError:
    logger.warning("ot.gpu not found - coupling computation will be in cpu")


def sinkhorn(cost_matrix, reg, num_iters=100, a=None, b=None, device="cpu"):
    """
    Sinkhorn algorithm to solve regularized optimal transport problem.
    Given a cost matrix, computes the optimal transport plan.

    Args:
        cost_matrix (torch.Tensor): a (n, m) matrix of pairwise costs.
        reg (float): the regularization parameter.
        num_iters (int): number of iterations.
        a (torch.Tensor, optional): source distribution. Defaults to uniform.
        b (torch.Tensor, optional): target distribution. Defaults to uniform.
        device (str, optional): device. Defaults to "cpu".

    Returns:
        torch.Tensor: the optimal transport plan.
        float: the optimal transport cost.
    """

    n, m = cost_matrix.shape
    if a is None:
        a = torch.ones(n, device=device) / n
    if b is None:
        b = torch.ones(m, device=device) / m

    K = torch.exp(-cost_matrix / reg)

    u = torch.ones(n, device=device) / n
    v = torch.ones(m, device=device) / m

    for _ in range(num_iters):
        u = a / (K @ v)
        v = b / (K.T @ u)

    # Transport plan
    P = u.reshape(-1, 1) * K * v.reshape(1, -1)

    # Transport cost
    cost = torch.sum(P * cost_matrix)

    logger.debug(f"Sinkhorn cost: {cost.item()}")
    logger.debug(f"Transport plan P: {P}")

    return P, cost.item()
