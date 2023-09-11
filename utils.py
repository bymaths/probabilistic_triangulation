import numpy as np
import torch 
import cv2

JOINT_LINKS = [(0,1),(0,2),(1,3),(2,4),(0,5),(0,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)]

def eulid_to_humo(points):
    """
        points: (...,N,M)
        return: (...,N,M+1)
    """
    if isinstance(points, np.ndarray):
        return np.concatenate([points, np.ones((*points.shape[:-1],1))], axis=-1)
    elif torch.is_tensor(points):
        return torch.cat([points, torch.ones((*points.shape[:-1],1),dtype=points.dtype,device=points.device)],dim=-1)
    else:
        raise TypeError("Works Only with numpy arrays and Pytorch tensors")
    
def homo_to_eulid(points):
    """
        points: (...,N,M+1)
        return: (...,N,M)
    """
    if isinstance(points, np.ndarray):
        return points[...,:-1] / points[...,-1,None]
    elif torch.is_tensor(points):
        return points[...,:-1] / points[...,-1,None]
    else:
        raise TypeError("Works Only with numpy arrays and Pytorch tensors")
    
def calIOU(b1,b2):
    """
    Input:
        b1,b2: [x1,y1,x2,y2]
    """
    s1 = (b1[2] - b1[0]) * (b1[3]-b1[1])
    s2 = (b2[2] - b2[0]) * (b2[3]-b2[1])
    a = max(0,min(b1[2],b2[2]) - max(b1[0],b2[0])) * max(0,min(b1[3],b2[3]) - max(b1[1],b2[1]))
    return a/(s1+s2-a)

class Camera():
    def __init__(self,n_view,n_joint):
        self.n_view = n_view
        self.n_joint = n_joint
        self.points2d = np.zeros(self.n_view,2,self.n_joint)
        self.confidence = np.zeros(self.n_joint)
        self.R = np.eyes(3)
        self.t = np.zeros((3,1))

    def update(self, R,t):
        self.R = R
        self.t = t

    def get_P(self):
        return np.concatenate([self.R,self.T],axis=1)

class Calibration():
    def __init__(self, n_view, n_joint):
        self.n_view = n_view
        self.n_joint = n_joint
        self.points3d = np.zeros(3,self.n_joint)
        self.confidence3d = np.zeros(self.n_joint)
        self.cameras = [Camera() for i in range(self.n_view)]

    def update(self, points2d, confidence):
        assert points2d.shape == self.points2d.shape
        assert confidence.shape == self.confidence.shape
        self.points2d = points2d
        self.confidence = confidence

    def get_vaild_index(self, confi):
        """
        Return:
            bool: (N)
        """
        return confi > 0.8

    def weighted_triangulation(self, filter_n_view):
        assert filter_n_view <= self.n_view 
        assert filter_n_view >= 2
        
        for j in range(self.n_joint):
            A = []
            for i in range(filter_n_view):
                if self.cameras[i].confidence[j] > 0.5:
                    P = self.cameras[i].get_P()
                    P3T = P[2]
                    A.append(self.cameras[i].confidence[j] * (self.cameras[i].points2d[0,j] * P3T - P[0]))
                    A.append(self.cameras[i].confidence[j]*(self.cameras[i].points2d[1,j] * P3T - P[1]))
            A = np.array(A)
            if A.shape[0] >=4:
                u, s, vh = np.linalg.svd(A)
                error = s[-1]
                X = vh[len(s) - 1]
                self.points3d[:,j] = X[:3] / X[3]
                self.confidence3d[j] = np.exp(-np.abs(error))
            else:
                self.points3d[:,j] = np.array([0.0,0.0,0.0])
                self.confidence3d[j] = 0

    def pnp(self):
        for camera in self.cameras:
            mask = self.get_vaild_index(camera.confidence)
            points2d = camera.points2d[:,mask]
            points3d = self.points3d[:,mask]
            ret, rvec, tvec = cv2.solvePnP(points3d,points2d, np.eye(3), np.zeros(5))
            R, _ = cv2.Rodrigues(rvec)
            camera.R = R
            camera.t = tvec



    def eight_point(self):
        mask = np.logical_and(self.get_vaild_index(self.cameras[0].confidence) , self.get_vaild_index(self.cameras[1].confidence) )

        points2d_0 = self.cameras[0].points2d[:,mask]
        points2d_1 = self.cameras[1].points2d[:,mask]
        

        E, mask = cv2.findEssentialMat(points2d_0, points2d_1, focal=1.0, pp=(0., 0.),
                                        method=cv2.RANSAC, prob=0.999, threshold=0.0003)
        points2d_0_inliers = points2d_0[mask.ravel() == 1]
        points2d_1_inliers = points2d_1[mask.ravel() == 1]
        
        point, R, t,mask  = cv2.recoverPose(E, points2d_0_inliers, points2d_1_inliers)

        self.cameras[0].R,self.cameras[0].T =  np.eye(3),np.zeros((3,1))
        self.cameras[1].R,self.cameras[1].T =  R,t
        self.weighted_triangulation(filter_n_view=2)
        self.pnp()
        self.weighted_triangulation(filter_n_view=self.n_view)


