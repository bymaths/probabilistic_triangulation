import numpy as np
import torch 
import cv2

JOINT_LINKS = [(0,1),(0,2),(1,3),(2,4),(0,5),(0,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)]

def cal_mpjpe(points3d, points2d, R, t):
    """
    Args:
        points3d : (...,J,3)
        points2d : (...,V,J,2)
        R : (...,V,3,3)
        t: (...,V,3,1)
    """
    return (homo_to_eulid((R[...,None,:,:] @ points3d[...,None,:,:,None] + t[...,None,:,:]).squeeze(-1)) - points2d ).mean()

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