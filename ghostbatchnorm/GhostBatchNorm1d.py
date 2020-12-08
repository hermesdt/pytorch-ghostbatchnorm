import torch
from torch import nn
import numpy as np

class GhostBatchNorm1d(nn.Module):
    def __init__(self, n_in, vbs, momentum=0.1, eps=1e-5, metric_dim=1):
        super().__init__()
        self.vbs = vbs
        self.eps = eps
        self.mm = momentum
        self.metric_dim = metric_dim

        gamma = nn.Parameter(torch.ones(n_in))
        self.register_parameter("gamma", gamma)

        beta = nn.Parameter(torch.zeros(n_in))
        self.register_parameter("beta", beta)

        self.register_buffer("running_mean", torch.zeros(n_in))
        self.register_buffer("running_std", torch.ones(n_in))


    def forward(self, X):
        num_ghost_batches = np.ceil(X.size(0)/self.vbs).astype(int)
        ghost_batches = X.view(num_ghost_batches, self.vbs, X.size(-1))

        ghost_mean = ghost_batches.mean(dim=self.metric_dim).unsqueeze(1)
        ghost_std = ghost_batches.std(dim=self.metric_dim).unsqueeze(1)

        normalized_ghost_batches = (ghost_batches - ghost_mean) / ghost_std
        normalized_batch = normalized_ghost_batches.view(X.size())

        self.running_mean = self._calculate_running_metric(self.running_mean, ghost_mean, num_ghost_batches)
        self.running_std = self._calculate_running_metric(self.running_std, ghost_std, num_ghost_batches)

        return self.gamma * normalized_batch + self.beta

    def _calculate_running_metric(self, running_metric, ghost_metric, num_ghost_batches):
        weighted_prev = ((1-self.mm)**num_ghost_batches) * running_metric

        exp_idxs = torch.arange(0, num_ghost_batches).flip(dims=(0,))
        weighted_new = (
            (self.mm * (1-self.mm)**exp_idxs)[..., None] * ghost_metric.squeeze(1)
        ).sum(dim=0)

        return weighted_prev + weighted_new
