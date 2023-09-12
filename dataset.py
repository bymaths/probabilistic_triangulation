from torch.utils.data import Dataset, DataLoader
from utils import *


class H36M(Dataset):

    def __init__(self, label_path = '/Users/byjiang/Code/Data/H36M_pose.npy',istrain=True):
        super().__init__()
        label = np.load(label_path,allow_pickle=True).item()
        self.istrain = istrain
        
        self.cameras = label['subject_cameras']
        self.db = label['train'] if self.istrain else label['test']

    def __len__(self):
        return len(self.db)
    
    def __getitem__(self, index):
        """
        pose/3d : (17,3)
        R : (4,3,3)
        t : (4,3,1)
        K : (4,3,3)
        pose/2d : (4,17,2)
        confi : (4,17)
        B,V,J,D,1
        B,4,17,3,1
        """
        frame = self.db[index]
        camera = self.cameras[frame['subject_index']]
        R, t = camera['R'].copy(), camera['t'].copy()

        pose_3d = (R[0,None] @ frame['pose/3d'][...,None] + t[0,None]).squeeze(-1)
        R[1:] = R[1:] @ np.linalg.inv(R[0])[None]
        t[1:] = t[1:] - R[1:] @ t[0,None]
        R[0] = np.eye(3)
        t[0] = np.zeros((3,1))
        scale = np.linalg.norm(t[1])
        pose_3d = pose_3d / scale
        t  = t / scale

        # X_2d = R X_3d + t
        pose_2d = homo_to_eulid( (R[:,None] @ pose_3d[None,...,None] + t[:,None]).squeeze(-1))
        confi = np.ones(pose_2d.shape[:-1])
        return pose_3d, pose_2d, confi, R, t
        # return {
        #     'pose/3d': frame['pose/3d'],
        #     'pose/2d':  homo_to_eulid( (camera['R'][:,None] @ frame['pose/3d'][...,None] + camera['t'][:,None]).squeeze(-1)),
        #     'R': camera['R'],
        #     't': camera['t']
        #     # 'K': camera['K']
        # }


if __name__ == "__main__":
    h36m = H36M()
    h36mloader = DataLoader(h36m, batch_size = 1, shuffle = True)
    for step, (pose_3d, pose_2d, confi, R, t) in enumerate(h36mloader):
        print(pose_2d.shape,pose_3d.shape, confi.shape,R.shape,t.shape)
        break