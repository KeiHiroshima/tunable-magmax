import sys
from logging import getLogger

# from venv import logger
import geomloss
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from src.merging.otdd import sinkhorn
from src.modeling import ImageEncoder

sys.path.append("../")
from .. import utils

"""
Dataset Distance Metrics Implementation
Implements MMD and Cosine Similarity based dataset distances
Compatible with OTDD interface from https://github.com/microsoft/otdd
"""


# NOTE: no import-time configuration here on purpose — see the same note in
# src/merging/task_vectors.py.
logger = getLogger(__name__)


class FeatureCost:
    """
    Feature cost computation compatible with OTDD's FeatureCost.
    Extracts features using provided embeddings.
    """

    def __init__(
        self,
        src_embedding: nn.Module | None = None,
        tgt_embedding: nn.Module | None = None,
        src_dim: tuple[int, ...] | None = None,
        tgt_dim: tuple[int, ...] | None = None,
        p: int = 2,
        device: str = "cpu",
    ):
        """
        Args:
            src_embedding: Embedding network for source dataset
            tgt_embedding: Embedding network for target dataset
            src_dim: Input dimension for source data
            tgt_dim: Input dimension for target data
            p: Norm order (not used in MMD/Cosine, kept for compatibility)
            device: Device to run computations on
        """
        self.src_embedding = src_embedding
        self.tgt_embedding = tgt_embedding
        self.src_dim = src_dim
        self.tgt_dim = tgt_dim
        self.p = p
        self.device = device

        if src_embedding is not None:
            self.src_embedding = src_embedding.to(device)
            self.src_embedding.eval()
        if tgt_embedding is not None:
            self.tgt_embedding = tgt_embedding.to(device)
            self.tgt_embedding.eval()

        # from otdd
        self.method = "precomputed_labeldist"
        self.symmetric_tasks, self.diagonal_cov = False, False

    def extract_features(
        self, loader: DataLoader, embedding: nn.Module
    ) -> torch.Tensor:
        """Extract features from a dataloader using the embedding network."""
        features_list = []

        with torch.no_grad():
            for batch in loader:
                if isinstance(batch, (list, tuple)):
                    data = batch[0]
                else:
                    data = batch

                data = data.to(self.device)
                feats = embedding.model.encode_image(data)
                features_list.append(feats.cpu())

        return torch.cat(features_list, dim=0)


class DatasetDistance:
    """
    Dataset distance computation using MMD or Cosine Similarity.
    Interface compatible with OTDD's DatasetDistance class.
    """

    def __init__(
        self,
        loader_src: Subset,
        loader_tgt: Subset,
        feature_cost: FeatureCost | None = None,
        method: str = "mmd",
        kernel_type: str = "rbf",
        kernel_bandwidth: float | None = None,
        device: str = "cpu",
        **kwargs,
    ):
        """
        Args:
            loader_src: DataLoader for source dataset
            loader_tgt: DataLoader for target dataset
            feature_cost: FeatureCost object for feature extraction
            method: Distance method ('mmd' or 'cosine')
            kernel_type: Kernel type for MMD ('rbf', 'linear', 'poly')
            kernel_bandwidth: Bandwidth parameter for RBF kernel (sigma)
            device: Device to run computations on
            **kwargs: Additional arguments for compatibility
        """
        self.loader_src = loader_src
        self.loader_tgt = loader_tgt
        self.feature_cost = feature_cost
        self.method = method.lower()
        self.kernel_type = kernel_type
        self.kernel_bandwidth = kernel_bandwidth
        self.device = device
        self.reg = kwargs.get("reg", 1.0)

        # Scaling factor for converting distance to similarity
        self.scaling_factor = 100.0

        if self.method not in ["mmd", "cosine", "ot"]:
            raise ValueError(
                f"Method must be 'mmd', 'cosine', or 'ot', got {self.method}"
            )

        if self.method == "mmd" and kernel_type not in ["rbf", "linear", "poly"]:
            raise ValueError(
                f"Kernel type must be 'rbf', 'linear', or 'poly', got {kernel_type}"
            )

    def _extract_all_features(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract features from both source and target datasets."""
        if self.feature_cost is not None:
            # Use provided feature extractor
            src_features = self.feature_cost.extract_features(
                self.loader_src, self.feature_cost.src_embedding
            )
            tgt_features = self.feature_cost.extract_features(
                self.loader_tgt, self.feature_cost.tgt_embedding
            )
        else:
            # Use raw data
            src_data = []
            tgt_data = []

            for batch in self.loader_src:
                if isinstance(batch, (list, tuple)):
                    data = batch[0]
                else:
                    data = batch
                src_data.append(data.reshape(data.shape[0], -1))

            for batch in self.loader_tgt:
                if isinstance(batch, (list, tuple)):
                    data = batch[0]
                else:
                    data = batch
                tgt_data.append(data.reshape(data.shape[0], -1))

            src_features = torch.cat(src_data, dim=0)
            tgt_features = torch.cat(tgt_data, dim=0)

        return src_features.to(self.device), tgt_features.to(self.device)

    def _compute_kernel_matrix(
        self, X: torch.Tensor, Y: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Compute kernel matrix K(X, Y).

        Args:
            X: Feature matrix of shape (n, d)
            Y: Feature matrix of shape (m, d). If None, compute K(X, X)

        Returns:
            Kernel matrix of shape (n, m) or (n, n)
        """
        if Y is None:
            Y = X

        if self.kernel_type == "linear":
            # Linear kernel: K(x, y) = <x, y>
            return torch.mm(X, Y.t())

        elif self.kernel_type == "rbf":
            # RBF kernel: K(x, y) = exp(-||x - y||^2 / (2 * sigma^2))
            # Compute pairwise squared distances
            X_norm = (X**2).sum(1).view(-1, 1)
            Y_norm = (Y**2).sum(1).view(1, -1)
            dists_sq = X_norm + Y_norm - 2.0 * torch.mm(X, Y.t())

            # Compute bandwidth if not provided
            if self.kernel_bandwidth is None:
                # Use median heuristic
                with torch.no_grad():
                    pdist = torch.pdist(X)
                    if len(pdist) > 0:
                        sigma = torch.median(pdist)
                    else:
                        sigma = 1.0
            else:
                sigma = self.kernel_bandwidth

            return torch.exp(-dists_sq / (2 * sigma**2))

        elif self.kernel_type == "poly":
            # Polynomial kernel: K(x, y) = (1 + <x, y>)^3
            return (1 + torch.mm(X, Y.t())) ** 3

        else:
            raise ValueError(f"Unknown kernel type: {self.kernel_type}")

    def _compute_mmd(
        self, X: torch.Tensor, Y: torch.Tensor, unbiased: bool = True
    ) -> float:
        """
        Compute Maximum Mean Discrepancy (MMD) between two samples.

        MMD^2 = E[k(x, x')] + E[k(y, y')] - 2*E[k(x, y)]

        Args:
            X: Source features of shape (n, d)
            Y: Target features of shape (m, d)
            unbiased: Whether to use unbiased estimator

        Returns:
            MMD distance (scalar)
        """
        n = X.shape[0]
        m = Y.shape[0]

        # Compute kernel matrices
        Kxx = self._compute_kernel_matrix(X, X)
        Kyy = self._compute_kernel_matrix(Y, Y)
        Kxy = self._compute_kernel_matrix(X, Y)

        if unbiased:
            # Unbiased estimator (exclude diagonal)
            # Term 1: 1/(n*(n-1)) * sum_{i!=j} k(x_i, x_j)
            term1 = (Kxx.sum() - Kxx.diagonal().sum()) / (n * (n - 1))

            # Term 2: 1/(m*(m-1)) * sum_{i!=j} k(y_i, y_j)
            term2 = (Kyy.sum() - Kyy.diagonal().sum()) / (m * (m - 1))

            # Term 3: 2/(n*m) * sum_{i,j} k(x_i, y_j)
            term3 = 2.0 * Kxy.sum() / (n * m)
        else:
            # Biased estimator (include diagonal)
            term1 = Kxx.mean()
            term2 = Kyy.mean()
            term3 = 2.0 * Kxy.mean()

        mmd_squared = term1 + term2 - term3

        # MMD can be negative due to numerical errors in unbiased estimator
        mmd = torch.sqrt(torch.clamp(mmd_squared, min=0.0))

        return mmd.item()

    def _compute_cosine_distance(self, X: torch.Tensor, Y: torch.Tensor) -> float:
        """
        Compute average cosine distance between dataset centroids.

        Distance = 1 - cosine_similarity(mean(X), mean(Y))

        Args:
            X: Source features of shape (n, d)
            Y: Target features of shape (m, d)

        Returns:
            Cosine distance (scalar in [0, 2])
        """
        # Compute centroids
        X_mean = X.mean(dim=0)
        Y_mean = Y.mean(dim=0)

        # Compute cosine similarity
        cos_sim = torch.nn.functional.cosine_similarity(
            X_mean.unsqueeze(0), Y_mean.unsqueeze(0)
        )

        # Convert to distance
        cos_dist = 1.0 - cos_sim.item()

        return cos_dist

    def _compute_otdd(
        self, X: torch.Tensor, Y: torch.Tensor
    ) -> tuple[float, torch.Tensor]:
        """
        Compute Optimal Transport Dataset Distance (OTDD).
        """
        # Compute pairwise cost matrix (squared Euclidean distance)
        X_norm = (X**2).sum(1).view(-1, 1)
        Y_norm = (Y**2).sum(1).view(1, -1)
        cost_matrix = X_norm + Y_norm - 2.0 * torch.mm(X, Y.t())
        cost_matrix = torch.clamp(cost_matrix, min=0.0)

        # scaling by max value
        cost_matrix_max = cost_matrix.max()
        cost_matrix = cost_matrix / (cost_matrix_max + 1e-8)

        P, cost = sinkhorn(cost_matrix, reg=self.reg, device=self.device)
        return cost, P

    def distance(
        self, maxsamples: int | None = None, return_coupling: bool = False
    ) -> float | tuple[float, torch.Tensor]:
        """
        Compute dataset distance.

        Args:
            maxsamples: Maximum number of samples to use (for compatibility)
            return_coupling: Whether to return coupling matrix (for compatibility)

        Returns:
            Distance value, or (distance, coupling) if return_coupling=True
        """
        # Extract features
        src_features, tgt_features = self._extract_all_features()

        # Subsample if requested
        if maxsamples is not None:
            n_src = src_features.shape[0]
            n_tgt = tgt_features.shape[0]

            if n_src > maxsamples:
                indices = torch.randperm(n_src)[:maxsamples]
                src_features = src_features[indices]

            if n_tgt > maxsamples:
                indices = torch.randperm(n_tgt)[:maxsamples]
                tgt_features = tgt_features[indices]

        # Compute distance based on method
        coupling = None
        if self.method == "mmd":
            dist = self._compute_mmd(src_features, tgt_features)
        elif self.method == "cosine":
            dist = self._compute_cosine_distance(src_features, tgt_features)
        elif self.method == "ot":
            dist, coupling = self._compute_otdd(src_features, tgt_features)

        else:
            raise ValueError(f"Unknown method: {self.method}")

        if return_coupling:
            if coupling is None:
                # For compatibility, return dummy coupling matrix
                n = src_features.shape[0]
                m = tgt_features.shape[0]
                coupling = torch.ones(n, m) / (n * m)
            return dist, coupling
        else:
            return dist


def compute_mmd_similarity(
    data_src: Subset,
    data_tgt: Subset,
    feature_cost: FeatureCost | None = None,
    kernel_type: str = "rbf",
    kernel_bandwidth: float | None = None,
    device: str = "cpu",
    maxsamples: int | None = None,
) -> float:
    """
    Compute MMD distance between two datasets.

    Args:
        loader_src: Subset for source dataset
        loader_tgt: Subset for target dataset
        feature_cost: FeatureCost object for feature extraction
        kernel_type: Kernel type ('rbf', 'linear', 'poly')
        kernel_bandwidth: Bandwidth for RBF kernel
        device: Device to run on
        maxsamples: Maximum samples to use

    Returns:
        MMD distance (scalar)
    """

    loader_src = DataLoader(data_src, batch_size=64, shuffle=False)
    loader_tgt = DataLoader(data_tgt, batch_size=64, shuffle=False)

    dist_obj = DatasetDistance(
        loader_src=loader_src,
        loader_tgt=loader_tgt,
        feature_cost=feature_cost,
        method="mmd",
        kernel_type=kernel_type,
        kernel_bandwidth=kernel_bandwidth,
        device=device,
    )

    distance = dist_obj.distance(maxsamples=maxsamples)

    conveting_method = "exponential"
    if conveting_method == "exponential":
        similarity = torch.exp(
            -torch.tensor(distance * dist_obj.scaling_factor) / 1.0
        ).item()
    else:
        similarity = 1.0 / (1.0 + distance)  # Convert distance to similarity

    logger.debug(f"MMD distance: {distance}, similarity: {similarity}")
    return similarity


def compute_cosine_similarity(
    data_src: Subset,
    data_tgt: Subset,
    feature_cost: FeatureCost | None = None,
    device: str = "cpu",
    maxsamples: int | None = None,
) -> float:
    """
    Compute cosine distance between dataset centroids.

    Args:
        data_src: Subset for source dataset
        data_tgt: Subset for target dataset
        feature_cost: FeatureCost object for feature extraction
        device: Device to run on
        maxsamples: Maximum samples to use

    Returns:
        Cosine distance (scalar in [0, 2])
    """
    loader_src = DataLoader(data_src, batch_size=64, shuffle=False)
    loader_tgt = DataLoader(data_tgt, batch_size=64, shuffle=False)

    dist_obj = DatasetDistance(
        loader_src=loader_src,
        loader_tgt=loader_tgt,
        feature_cost=feature_cost,
        method="cosine",
        device=device,
    )
    distance = dist_obj.distance(maxsamples=maxsamples)

    conveting_method = "exponential"
    if conveting_method == "exponential":
        similarity = torch.exp(
            -torch.tensor(distance * dist_obj.scaling_factor) / 2.0
        ).item()
    else:
        similarity = 1.0 / (1.0 + distance)  # Convert distance to similarity

    logger.debug(f"Cosine distance: {distance}, similarity: {similarity}")
    return similarity


def compute_otdd_similarity(
    data_src: Subset,
    data_tgt: Subset,
    feature_cost: ImageEncoder | None = None,
    reg: float = 0.1,
    device: str = "cpu",
    maxsamples: int | None = None,
) -> float:
    """
    Compute OTDD between two datasets.

    Args:
        data_src: Subset for source dataset
        data_tgt: Subset for target dataset
        feature_cost: FeatureCost object for feature extraction
        reg: Regularization parameter for Sinkhorn algorithm
        device: Device to run on
        maxsamples: Maximum samples to use

    Returns:
        OTDD (scalar)
    """
    loader_src = DataLoader(data_src, batch_size=64, shuffle=False)
    loader_tgt = DataLoader(data_tgt, batch_size=64, shuffle=False)
    dist_obj = DatasetDistance(
        loader_src=loader_src,
        loader_tgt=loader_tgt,
        feature_cost=feature_cost,
        method="ot",
        device=device,
        reg=reg,
    )

    distance = dist_obj.distance(maxsamples=maxsamples)

    conveting_method = "exponential"
    if conveting_method == "exponential":
        similarity = torch.exp(
            -torch.tensor(distance * dist_obj.scaling_factor) / 1.0
        ).item()
    else:
        similarity = 1.0 / (1.0 + distance)  # Convert distance to similarity

    logger.debug(f"OTDD distance: {distance}, similarity: {similarity}")
    return similarity


def count_labels(
    loader_tgt: Subset,
    task_class_dict: dict[list] = None,
    task_idx: int = None,
) -> float:
    """
    Compute label distance between two datasets as the absolute difference
    in label distributions.

    Args:
        loader_src: DataLoader for source dataset
        loader_tgt: DataLoader for target dataset
        task_class_dict: Optional dict mapping task indices to class lists

    Returns:
        Label distance (scalar)
    """

    class_in_task = task_class_dict[task_idx]

    labels_target_data = [onedata[1] for onedata in loader_tgt]

    num_data_in_task = 0
    num_data_in_task = sum(1 for label in labels_target_data if label in class_in_task)

    logger.debug(f"Number of data in task {task_idx}: {num_data_in_task}")
    return num_data_in_task


cost_routines = {
    1: (lambda x, y: geomloss.utils.distances(x, y)),
    2: (lambda x, y: geomloss.utils.squared_distances(x, y) / 2),
}

