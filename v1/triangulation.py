import torch
from distribution import AngularCentralGaussian, cholesky_wrapper
from pyro.distributions import MultivariateStudentT
from utils import *
import torch.nn.functional as F

class ProbabilisticTriangulation():
    def __init__(self, cfg):
        """
        Members:
            expect_quan: Rotation (B,V,*)
            tril_R: (B,V-1,4,4)
            mu_t: Translation (B,V,*)
            tril_t: (B,V-1,3,3)
        """
        self.nB = cfg["nB"]
        self.nV = cfg["nV"]
        self.M = cfg["M"]
        self.isDistr = cfg["isDistr"]
        
        if self.isDistr:
            self.expect_quan = Rotation(torch.tensor([1.,0.,0.,0.]).repeat(self.nB,self.nV,1))
            self.mu_t = Translation(torch.zeros(self.nB,self.nV,3))

            self.tril_R = torch.eye(4,4).repeat(self.nB, self.nV-1, 1, 1)
            self.tril_t = torch.eye(3,3).repeat(self.nB, self.nV-1, 1, 1)
            #  conv_quan (B,V,4,4)
            self.distrR = AngularCentralGaussian(self.tril_R)
            #  mu_t (B,V,3) conv_t (B,V,3,3)
            self.distrT = MultivariateStudentT(loc=self.mu_t.distr_norm(),scale_tril=self.tril_t,df=3)

            self.bufferR, self.bufferT = None,None
        
        else:
            self.bufferR = Rotation(torch.randn(self.M//8,self.nB, self.nV-1, 4))
            self.bufferT = Translation(torch.randn(self.M//8,self.nB, self.nV-1, 3))

        self.lr = 1e-2

    def sample(self, nM):

        if self.isDistr:
            if self.bufferR is not None:
                nM -= self.bufferR.quan.shape[-4]
            rot = Rotation(self.distrR((nM,)))
            t = Translation(self.distrT((nM,)))
            if self.bufferR is not None:
                # print(rot.quan.shape, self.bufferR.quan.shape)
                rot.cat(self.bufferR)
                t.cat(self.bufferT)
            return rot,t

        else:
            buffer_nM = self.bufferR.quan.shape[-4]
            
            temp_buffer_quan = self.bufferR.quan[...,1:,:].repeat(nM//buffer_nM, 1, 1, 1)
            rot = Rotation(
                temp_buffer_quan + torch.randn_like(temp_buffer_quan) * self.lr
            )
            
            temp_buffer_vector = self.bufferT.vector[...,1:,:].repeat(nM//buffer_nM, 1, 1, 1)
            t = Translation(
                temp_buffer_vector + torch.randn_like(temp_buffer_vector) * self.lr
            )
            self.lr *= 0.1
            rot.random(lr = self.lr)
            t.random(lr = self.lr*10)
            return rot,t

        
    def update_paramater_init(self,points3d,points2d, rot,t):
        """
        Args:
            rot Tensor -> Rotation:  (B,V,3,3)
            t Tensor -> Translation: (B,V,3,1)
        Returns:
            sample_quan : (M,B,V,4)
            sample_t : (M,B,V,3)
            weights: (M,B)
        """

        self.lr = 1e-3
        self.bufferR = Rotation(rot.repeat(self.M//8,1,1,1,1))
        self.bufferT = Translation(t.repeat(self.M//8,1,1,1,1))
        self.bufferR.random(self.lr)
        self.bufferT.random(self.lr*10)
        rot, t = self.sample(self.M)

        # weights = torch.cat([
        #     torch.ones(self.M-1,self.nB) * (0.5/(self.M-1)),
        #     torch.ones(1,self.nB)*0.5,
        # ], dim = 0)
        weights = cal_mpjpe_batch(points3d,points2d, rot,t)
        self.update_paramater_with_weights(rot, t, weights)

    def update_paramater_with_weights(self,rot,t, weights):
        """
        Args:
            rot : (M,B,V,*)
            t : (M,B,V,*)
            weights : (M,B)
        Returns:
            conv_quan : (B,V,4,4)
            mu_t : (B,V,3)
            conv_t : (B,V,3,3)
        """


        topk_weight, indices = torch.topk(weights, self.M//8, dim=0)
        # indices (M/2, B) -> (M/2,B,V,*)
        indices = indices[...,None,None]
        half_quan = rot.quan.gather(0, indices.expand(-1,-1,self.nV,4))
        half_vector = t.vector.gather(0, indices.expand(-1,-1,self.nV,3))

        rot = Rotation(half_quan)
        t = Translation(half_vector)
        weights = topk_weight

        # (M,B,V,4) * (M,B,1,1) -> (B,V,4)
        self.expect_quan = Rotation(
            (rot.quan * weights[...,None,None]).sum(0) / weights.sum(0)[...,None,None]
        )
        # (M,B,V,3) * (M,B) -> (B,V,3)
        self.mu_t = Translation(
            (t.vector * weights[...,None,None]).sum(0) / weights.sum(0)[...,None,None]
        )

        if self.isDistr:
            # (B,V-1,4,M) @ (B,V-1,M,4) -> (B,V-1,4,4)
            conv_quan = (
                rot.distr_norm().permute(1,2,3,0) @ (rot.distr_norm() * weights[...,None,None]).permute(1,2,0,3)
            ) / weights.sum(0)[...,None,None,None]

            # u,s,vt = torch.linalg.svd(conv_quan)
            # s *= torch.tensor([1,0.1,0.01,0.001])[None,None]
            # conv_quan = u @ torch.diag_embed(s) @ vt

            self.tril_quan = cholesky_wrapper(conv_quan)

            # (M,B,V,3) - (1,B,V,3) -> (M,B,V,3) -> (M,B    ,V,3)
            centered_t = Translation(t.vector - self.mu_t.vector[None]).distr_norm()
            # (B,V-1,3,M) @ (B,V-1,M,3) -> (B,V-1,3,3)
            conv_t = (
                centered_t.permute(1,2,3,0) @ (centered_t * weights[...,None,None]).permute(1,2,0,3)
            ) / weights.sum(0)[...,None,None,None]

            # u,s,vt = torch.linalg.svd(conv_t)
            # s *= torch.tensor([1,0.1,0.01])[None,None]
            # conv_t = u @ torch.diag_embed(s) @ vt

            self.tril_t = cholesky_wrapper(conv_t)

            self.distrR = AngularCentralGaussian(self.tril_quan)
            self.distrT = MultivariateStudentT(loc=self.mu_t.distr_norm(),scale_tril=self.tril_t,df = 3)


        self.bufferR = Rotation(half_quan)
        self.bufferT = Translation(half_vector)
            

    def getbest_Rt(self):
        return Rotation(self.bufferR.quan[0]), Translation(self.bufferT.vector[0])

    def getbuffer_Rt(self):
        return self.bufferR, self.bufferT     


