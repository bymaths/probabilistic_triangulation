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

def eight_point(points2d):
    """
    points2d: (2,N)
    points3d: (3,N)
    """
    E, mask = cv2.findEssentialMat(points2d[0], points2d[1], focal=1.0, pp=(0., 0.),
                                       method=cv2.RANSAC, prob=0.999, threshold=0.0003)
    point2ds_0_inliers = points2d[0][mask.ravel() == 1]
    point2ds_1_inliers = points2d[1][mask.ravel() == 1]
    
    point, R, t,mask  = cv2.recoverPose(E, point2ds_0_inliers, point2ds_1_inliers)

        