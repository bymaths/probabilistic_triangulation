from pyro.distributions import TorchDistribution,constraints
from torch.distributions.multivariate_normal import _batch_mahalanobis, _standard_normal, _batch_mv

from pyro.distributions.util import broadcast_shape
import torch
import math



def cholesky_wrapper(mat, default_diag=None, force_cpu=True):
    device = mat.device
    if force_cpu:
        mat = mat.cpu()
    try:
        tril = torch.linalg.cholesky(mat, upper=False)
    except RuntimeError:
        n_dims = mat.size(-1)
        tril = []
        default_tril_single = torch.diag(mat.new_tensor(default_diag)) if default_diag is not None \
            else torch.eye(n_dims, dtype=mat.dtype, device=mat.device)
        for cov in mat.reshape(-1, n_dims, n_dims):
            try:
                tril.append(torch.cholesky(cov, upper=False))
            except RuntimeError:
                tril.append(default_tril_single)
        tril = torch.stack(tril, dim=0).reshape(mat.shape)
    return tril.to(device)


class AngularCentralGaussian(TorchDistribution):
    arg_constraints = {'scale_tril': constraints.lower_cholesky}
    has_rsample = True

    def __init__(self, scale_tril, validate_args=None, eps=1e-6):
        q = scale_tril.size(-1)
        assert q > 1
        assert scale_tril.shape[-2:] == (q, q)
        batch_shape = scale_tril.shape[:-2]
        event_shape = (q,)
        self.scale_tril = scale_tril.expand(batch_shape + (-1, -1))
        self._unbroadcasted_scale_tril = scale_tril
        self.q = q
        self.area = 2 * math.pi ** (0.5 * q) / math.gamma(0.5 * q)
        self.eps = eps
        super().__init__(batch_shape, event_shape, validate_args=validate_args)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        value = value.expand(
            broadcast_shape(value.shape[:-1], self._unbroadcasted_scale_tril.shape[:-2])
            + self.event_shape)
        M = _batch_mahalanobis(self._unbroadcasted_scale_tril, value)
        half_log_det = self._unbroadcasted_scale_tril.diagonal(dim1=-2, dim2=-1).log().sum(-1)
        return M.log() * (-self.q / 2) - half_log_det - math.log(self.area)

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        normal = _standard_normal(shape,
                                  dtype=self._unbroadcasted_scale_tril.dtype,
                                  device=self._unbroadcasted_scale_tril.device)
        gaussian_samples = _batch_mv(self._unbroadcasted_scale_tril, normal)
        gaussian_samples_norm = gaussian_samples.norm(dim=-1)
        samples = gaussian_samples / gaussian_samples_norm.unsqueeze(-1)
        samples[gaussian_samples_norm < self.eps] = samples.new_tensor(
            [1.] + [0. for _ in range(self.q - 1)])
        return samples