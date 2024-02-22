import torch
from triangulation import ProbabilisticTriangulation
import cv2
import numpy as np
from utils import *

class CalibrationBatch():
    def __init__(self,cfg, points2d, confi2d):
        """
        points2d : (B,V,J,2)
        confi2d : (B,V,J)
        points3d : (B,J,3)
        confi3d : (B,J)
        R : (B,V,3,3)
        t : (B,V,3,1)
        isdistribution : bool
        """
        self.n_batch,self.n_view,self.n_joint = points2d.shape[:3]
        self.M = cfg["M"]
        self.points2d = points2d
        self.confi2d = confi2d
        self.points3d = torch.zeros((self.n_batch,self.n_joint,3))
        self.confi3d = torch.zeros((self.n_batch,self.n_joint))
        self.R = torch.zeros((self.n_batch,self.n_view,3,3))
        self.t = torch.zeros((self.n_batch,self.n_view,3,1))
        self.prob_tri = ProbabilisticTriangulation(cfg)
        self.buffer_weights = None


    def weighted_triangulation(self, points2d, confi2d, R ,t):
        """
        Args:
            points2d : (V',J,2)
            confi2d : (V',J)
            R : (V',3,3)
            t : (V',3,1)
        Returns:
            points3d : (J,3)
            confi3d : (J)
        """
        n_view_filter= points2d.shape[0]
        points3d = torch.zeros((self.n_joint, 3))
        confi3d = torch.zeros((self.n_joint))
        # print(points2d.shape,confi2d.shape,R.shape,t.shape)
        for j in range(self.n_joint):
            A = []
            for i in range(n_view_filter):
                if confi2d[i,j] > 0.5:
                    P = torch.cat([R[i],t[i]],dim=1)
                    P3T = P[2]
                    A.append(confi2d[i,j] * (points2d[i,j,0]*P3T - P[0]))
                    A.append(confi2d[i,j] * (points2d[i,j,1]*P3T - P[1]))
            A = torch.stack(A)
            # print(A.shape)
            if A.shape[0] >= 4:
                u, s, vh = torch.linalg.svd(A)
                error = s[-1]
                X = vh[len(s) - 1]
                points3d[j,:] = X[:3] / X[3]
                confi3d[j] = np.exp(-torch.abs(error))
            else:
                points3d[:,j] = torch.tensor([0.0,0.0,0.0])
                confi3d[j] = 0

        return points3d, confi3d

    def weighted_triangulation_sample(self, points2d, confi2d, R ,t):
        """
        Args:
            points2d : (B,V',J,2)
            confi2d : (B,V',J)
            R : ((M),B, V',3,3)
            t : ((M),B, V',3,1)
        Returns:
            sample_points3d : ((M),B,J,3)
            sample_confi3d : ((M),B,J)
        """
        if len(R.matrix.shape[:-2]) > len(points2d.shape[:-2]):
            nM = R.matrix.shape[0]
            sample_points3d = torch.zeros((nM,self.n_batch,self.n_joint,3))
            sample_confi3d = torch.zeros((nM,self.n_batch,self.n_joint))
            for i in range(nM):
                for j in range(self.n_batch):
                    sample_points3d[i,j], sample_confi3d[i,j] = self.weighted_triangulation(
                        points2d[j], confi2d[j], R.matrix[i,j], t.trans[i,j]
                    )
            return sample_points3d, sample_confi3d
            
        else:
            sample_points3d = torch.zeros((self.n_batch,self.n_joint,3))
            sample_confi3d = torch.zeros((self.n_batch,self.n_joint))

            for j in range(self.n_batch):
                sample_points3d[j], sample_confi3d[j] = self.weighted_triangulation(
                    points2d[j], confi2d[j], R.matrix[j], t.trans[j]
                )

            return sample_points3d, sample_confi3d

    def pnp(self,batch_id):
        self.R[batch_id,0] = torch.eye(3)
        self.t[batch_id,0] = torch.zeros((3,1))
        for i in range(1, self.n_view):
            mask = torch.logical_and(self.confi2d[batch_id,i]>0.8,self.confi3d[batch_id]>0.8)
            p2d = self.points2d[batch_id,i,mask].numpy()
            p3d = self.points3d[batch_id,mask].numpy()
            ret, rvec, tvec = cv2.solvePnP(p3d, p2d, np.eye(3), np.zeros(5))
            R, _ = cv2.Rodrigues(rvec)
            self.R[batch_id,i] = torch.tensor(R)
            self.t[batch_id,i] = torch.tensor(tvec)


    def eight_point(self):
        for batch_id in range(self.n_batch):
            mask = torch.logical_and(self.confi2d[batch_id,0]>0.5, self.confi2d[batch_id,1]>0.5)
            
            p0 = self.points2d[batch_id,0,mask].numpy()
            p1 = self.points2d[batch_id,1,mask].numpy()
            # p0,p1 (N,2)
            # print(p0.shape,p1.shape)
            E, mask = cv2.findEssentialMat(p0, p1, focal=1.0, pp=(0., 0.),
                                            method=cv2.RANSAC, prob=0.999, threshold=0.0003)
            p0_inliers = p0[mask.ravel() == 1]
            p1_inliers = p0[mask.ravel() == 1]
            point, R, t,mask  = cv2.recoverPose(E, p0_inliers, p1_inliers)
            self.R[batch_id,0],self.t[batch_id,0] = torch.eye(3), torch.zeros((3,1))
            self.R[batch_id,1],self.t[batch_id,1] = torch.tensor(R),torch.tensor(t)

            # print(self.R[batch_id,1],self.t[batch_id,1])

            self.points3d[batch_id], self.confi3d[batch_id] = self.weighted_triangulation(
                self.points2d[batch_id,:2],self.confi2d[batch_id,:2],self.R[batch_id,:2],self.t[batch_id,:2]
            )
            
            self.pnp(batch_id)
            
            # print(self.R[batch_id,0],self.t[batch_id,0])
            # print(self.mpjpe(2))
            # print(self.confi3d[batch_id])

            self.points3d[batch_id], self.confi3d[batch_id] = self.weighted_triangulation(
                self.points2d[batch_id],self.confi2d[batch_id],self.R[batch_id],self.t[batch_id]
            )
            # print(self.confi3d[batch_id])
            # print(self.mpjpe(self.n_view))

    def monte_carlo(self):
        self.eight_point()
        self.prob_tri.update_paramater_init(self.points3d, self.points2d ,self.R,self.t)

        weights_log = []
        for i in range(4):
            rot,t = self.prob_tri.sample(self.M)
            # print(self.points2d.shape, self.confi2d.shape, rot.quan.shape, t.vector.shape)
            sample_points3d, sample_confi3d = self.weighted_triangulation_sample(self.points2d, self.confi2d, rot, t)
            weights = cal_mpjpe_batch(sample_points3d, self.points2d,  rot, t)
            weights_log.append(weights)
            self.prob_tri.update_paramater_with_weights(rot,t,weights)

        rot,t = self.prob_tri.getbuffer_Rt()
        # print(self.points2d.shape, self.confi2d.shape, rot.quan.shape, t.vector.shape)
        points3d, confi3d = self.weighted_triangulation_sample(self.points2d, self.confi2d, rot, t)
        self.buffer_weights = cal_mpjpe_batch(points3d,self.points2d,rot,t)

        rot,t = self.prob_tri.getbest_Rt()
        self.R = rot.matrix
        self.t = t.trans
        return weights_log

    def mpjpe(self, n_view_filter):
        return (homo_to_eulid((self.R[...,:n_view_filter,None,:,:] @ self.points3d[...,None,:,:,None] + self.t[...,:n_view_filter,None,:,:]).squeeze(-1)) - self.points2d[:,:n_view_filter] ).mean()




