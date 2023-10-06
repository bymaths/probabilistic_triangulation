import torch
from distribution import AngularCentralGaussian, cholesky_wrapper
from pyro.distributions import MultivariateStudentT
from utils import homo_to_eulid, eulid_to_humo, matrix_to_quaternion,quaternion_to_matrix
import torch.nn.functional as F

def cal_mpjpe_batch(points3d, points2d, R, t):
    """
    Args:
        points3d : (...,J,3)
        points2d : (...,V,J,2)
        R : (...,V,3,3)
        t: (...,V,3,1)
    Returns:
        weights : (...)
    """
    return torch.exp(
        -(
            homo_to_eulid(
                (R[...,None,:,:] @ points3d[...,None,:,:,None] + t[...,None,:,:]
                ).squeeze(-1)
            ) - points2d 
        ).norm(dim=-1).mean((-1,-2))
    )


class ProbabilisticTriangulation():
    def __init__(self, n_batch, n_view):
        self.n_batch = n_batch
        self.n_view = n_view
        self.expect_quan = torch.zeros(self.n_batch,self.n_view-1,4)
        self.tril_R = torch.eye(4,4)[None,None].expand(self.n_batch, self.n_view-1,-1,-1)
        self.mu_t = torch.zeros(self.n_batch,self.n_view-1,3)
        self.tril_t = torch.eye(3,3)[None,None].expand(self.n_batch,self.n_view-1,-1,-1)
        #  conv_quan (B,V,4,4)
        self.distrR = AngularCentralGaussian(self.tril_R)
        #  mu_t (B,V,3) conv_t (B,V,3,3)
        self.distrT = MultivariateStudentT(loc=self.mu_t,scale_tril=self.tril_t,df=3)

    def sample(self, size : torch.Size()):
        self.quan = self.distrR(size)
        self.t = self.distrT(size)
        # print(self.quan.shape, self.t.shape)

        sample_R = torch.cat([torch.eye(3)[None,None,None].expand(size[0],self.n_batch,-1,-1,-1) ,quaternion_to_matrix(self.quan)], dim = -3)
        sample_t = torch.cat([torch.zeros(size[0],self.n_batch,1,3) ,self.t] , dim = -2).unsqueeze(-1)
        return sample_R, sample_t

    def update_paramater_init(self, R,t):
        """
        Args:
            R : (B,V+1,3,3) -> (B,V,3,3)
            t : (B,V+1,3,1) -> (B,V,3,1)
        Returns:
            sample_quan : (M,B,V,4)
            sample_t : (M,B,V,3)
            weights: (M,B)
            M = 16
        """
        self.sample((15,))
        sample_quan = torch.cat([ matrix_to_quaternion(R[:,1:])[None], self.quan ],dim=0)
        sample_t = torch.cat([ t[None,:,1:].squeeze(-1), self.t], dim=0)
        self.quan = sample_quan
        self.t = sample_t
        weights = torch.tensor([1]+[0.1 for i in range(15)])[...,None].expand(-1,self.n_batch)
        self.update_paramater_with_weights(weights)

    def update_paramater_with_weights(self, weights):
        """
        Args:
            self.quan : (M,B,V,4)
            self.t : (M,B,V,3)
            weights : (M,B)
        Returns:
            conv_quan : (B,V,4,4)
            mu_t : (B,V,3)
            conv_t : (B,V,3,3)
        """
        
        # (B,V,4,M) @ (B,V,M,4) -> (B,V,4,4)
        conv_quan = (
            self.quan.permute(1,2,3,0) @ (self.quan * weights[...,None,None]).permute(1,2,0,3)
        ) / weights.sum(0)[...,None,None]
        self.tril_quan = cholesky_wrapper(conv_quan)

        self.mu_t = self.t.mean(0)

        centered_t = self.t - self.mu_t[None]
        # (B,V,3,M) @ (B,V,M,3) -> (B,V,3,3)
        conv_t = (
            centered_t.permute(1,2,3,0) @ (centered_t * weights[...,None,None]).permute(1,2,0,3)
        ) / weights.sum(0)[...,None,None]
        self.tril_t = cholesky_wrapper(conv_t)

        self.expect_quan = (self.quan * weights[...,None,None]).sum(0) / weights.sum(0)[...,None,None]
        self.distrR = AngularCentralGaussian(self.tril_quan)
        self.distrT = MultivariateStudentT(loc=self.mu_t,scale_tril=self.tril_t,df = 3)

    def getRt(self):
        """
        Returns:
            R : (B,V,3,3)
            t : (B,V,3,1)
        """
        R = torch.cat( [ torch.eye(3)[None,None].expand(self.n_batch,1,3,3), quaternion_to_matrix(self.expect_quan) ] ,dim=-3)
        t = torch.cat([torch.zeros(self.n_batch,1,3) ,self.mu_t] , dim = -2).unsqueeze(-1)
        return R,t



