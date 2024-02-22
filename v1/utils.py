import numpy as np
import torch 
import cv2
import torch.nn.functional as F

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


def cal_mpjpe_batch(points3d, points2d, rot, t):
    """
    Args:
        points3d : ((M),...,J,3)
        points2d : (...,V,J,2)
        rot : Rotation ((M),...,V,*)
        t: Translation ((M),...,V,*)
    Returns:
        weights : (...) or ((M),...)
    """
    if(len(rot.quan.shape[:-1]) > len(points2d.shape[:-2])):
        return torch.pow(torch.exp(
            -(
                homo_to_eulid(
                    (rot.matrix[...,None,:,:] @ points3d[...,None,:,:,None] + t.trans[...,None,:,:]
                    ).squeeze(-1)
                ) - points2d[None] 
            ).norm(dim=-1).mean((-1,-2))
        ), 4)
    else:
        return torch.exp(
            -(
                homo_to_eulid(
                    (rot.matrix[...,None,:,:] @ points3d[...,None,:,:,None] + t.trans[...,None,:,:]
                    ).squeeze(-1)
                ) - points2d 
            ).norm(dim=-1).mean((-1,-2))
        )

def eulid_to_homo(points):
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


def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))

def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret
    
def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)

    return quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))

class Rotation():
    def __init__(self, rot):
        """
            quan ((M),B,V,4)
            matrix ((M),B,V,3,3)
        """
        assert((rot.shape[-1] == 4) or (rot.shape[-1] == 3 and rot.shape[-2] == 3))

        if rot.shape[-1] == 4:
            rot = self.standard_quan(rot)
            self.quan = rot
            self.matrix = quaternion_to_matrix(self.quan)

        else:
            self.matrix = rot
            self.quan = matrix_to_quaternion(self.matrix)

    def cat(self,rot):
        assert(isinstance(rot, (Rotation, torch.Tensor)))
        if isinstance(rot, torch.Tensor):
            rot = Rotation(rot)
        assert(rot.quan.shape[-3:] == self.quan.shape[-3:])
        if len(self.quan.shape) > len(rot.quan.shape):
            self.quan = torch.cat([self.quan, rot.quan[None]], dim=-4)
            self.matrix = torch.cat([self.matrix, rot.matrix[None]], dim=-5)
        else:
            self.quan = torch.cat([self.quan, rot.quan], dim=-4)
            self.matrix = torch.cat([self.matrix, rot.matrix], dim=-5)

    def distr_norm(self):
        return self.quan[...,1:,:]

    def standard_quan(self,rot):
        rot = rot / torch.clamp_(rot.norm(dim = -1)[...,None],min=1e-4)
        size = rot.shape
        rot0 = torch.tensor([1,0,0,0]).repeat(*(size[:-2]),1,1)
        if F.l1_loss(rot0,rot[...,0:1,:]) < 1e-6:
            return rot
        else:
            return torch.cat([rot0, rot],dim=-2)

    def random(self, lr):
        assert(len(self.quan.shape) == 4)
        self.quan[1:,:,1:] += torch.randn_like(self.quan[1:,:,1:]) * lr
        self.quan = self.standard_quan(self.quan)
        self.matrix = quaternion_to_matrix(self.quan)

        


class Translation():
    def __init__(self, t):
        """
            vector ((M),B,V,3)
            trans ((M),B,V,3,1)
        """
        assert(t.shape[-1] == 3 or (t.shape[-1]==1 and t.shape[-2]==3))
        if t.shape[-1] == 3:
            t = self.standard_vector(t)
            self.vector = t
            self.trans = t.unsqueeze(-1)
        else:
            self.vector = t.squeeze(-1)
            self.trans = t
        # (M,B)
        t_norm = self.vector[...,1,:].norm(dim=-1)
        if not (t_norm == 0.0).any():
            self.vector[...,1:,:] /= t_norm[...,None,None]
            self.trans = self.vector.unsqueeze(-1)

    def cat(self,t):
        assert(isinstance(t, (Translation, torch.Tensor)))
        if isinstance(t ,torch.Tensor):
            t = Translation(t)
        assert(t.trans.shape[-3:] == self.trans.shape[-3:])
        if len(self.vector.shape) > len(t.vector.shape):
            self.vector = torch.cat([self.vector, t.vector[None]],dim=-4)
            self.trans = torch.cat([self.trans, t.trans[None]], dim=-5)
        else:
            self.vector = torch.cat([self.vector, t.vector],dim=-4)
            self.trans = torch.cat([self.trans, t.trans], dim=-5)

    def distr_norm(self):
        return self.vector[...,1:,:]

    def standard_vector(self,t):
        size = t.shape
        t0 = torch.tensor([0,0,0]).repeat(*(size[:-2]),1,1)
        if F.l1_loss(t0,t[...,0:1,:]) < 1e-6:
            return t / torch.clamp_(t[...,1,:].norm(dim = -1)[...,None,None],min=1e-4)
        else:
            return torch.cat([t0, t/torch.clamp_(t[...,0,:].norm(dim = -1)[...,None,None],min=1e-4)],dim=-2)

    def random(self, lr):
        assert(len(self.vector.shape) == 4)
        self.vector[1:,:,1:] += torch.randn_like(self.vector[1:,:,1:]) * lr
        self.vector = self.standard_vector(self.vector)
        self.trans = self.vector.unsqueeze(-1)