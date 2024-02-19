from utils import eulid_to_homo
import numpy as np
import cv2

class Compose():
    def __init__(self, transforms):
        self.transforms = transforms
    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample
        
class Crop():
    def __init__(self, scaleRange,moveRange):
        self.scaleRange = scaleRange
        self.moveRange = moveRange

    def __call__(self,sample):
        H, W, C = sample['image'].shape
        scale = np.random.uniform(self.scaleRange[0],self.scaleRange[1])
        moveX = np.random.uniform(self.moveRange[0], self.moveRange[1])
        moveY = np.random.uniform(self.moveRange[0], self.moveRange[1])
        B = sample['box']
        width = B[2] - B[0]
        height = B[3] - B[1]
        cx = int((B[0] + B[2])/2 + moveX * width)
        cy = int((B[1] + B[3])/2 + moveY * height)

        side = int(max(width, height) * scale)
        A = np.asarray([
            cx - side//2,
            cy - side//2,
            cx + side - side//2,
            cy + side - side//2
        ])
        Aclip = np.clip(A, [0, 0, 0, 0], [W, H, W, H])
        img = np.zeros((side, side, C))
        img[(Aclip[1]-A[1]):(Aclip[3]-A[1]), (Aclip[0]-A[0]):(Aclip[2]-A[0])
            ] = sample['image'][Aclip[1]: Aclip[3], Aclip[0]: Aclip[2]]

        sample['image'] = img
        sample['camera'].update_after_crop(A)
        del sample['box']
        return sample

class Resize():
    def __init__(self, image_size, interpolation=cv2.INTER_NEAREST):
        self.image_size = (image_size, image_size) if isinstance(
            image_size, (int, float)) else image_size
        self.interpolation = interpolation

    def __call__(self, sample):
        sample['camera'].update_after_resize(sample['image'].shape[:2], self.image_size)
        sample['image'] = cv2.resize(
            sample['image'], self.image_size, interpolation=cv2.INTER_CUBIC)
        return sample
    
class NormSkeleton():
    def __init__(self, root_id=6):
        self.root_id = root_id
    
    def __call__(self, sample):
        x3d = eulid_to_homo(sample['x3d']) @ sample['camera'].extrinsics().T

        x2d = x3d @ sample['camera'].K.T
        x2d[:,:2] /= x2d[:,2:]

        sample['x3d'] = (x3d-x3d[self.root_id]).astype(np.float32)
        sample['x2d'] = x2d[:,:2].astype(np.float32)
        sample['K'] = sample['camera'].K
        sample['R'] = sample['camera'].R
        sample['t'] = sample['camera'].t
        del sample['camera']
        return sample

class NormImage():
    def __init__(self):
        pass

    def __call__(self,sample):
        sample['image'] = np.clip(sample['image']/255., 0.,1.).transpose(2,0,1).astype(np.float32)
        return sample


class RandomBrightness(object):
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta
    def __call__(self, sample):
        if np.random.randint(2):
            delta = np.random.uniform(-self.delta, self.delta)
            sample['image'] += delta
        return sample
    
class RandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."
    def __call__(self, sample):
        if np.random.randint(2):
            alpha = np.random.uniform(self.lower, self.upper)
            sample['image'] *= alpha
        return sample


class RandomSaturation(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."
    def __call__(self, sample):
        if np.random.randint(2):
            sample['image'][:, :, 1] *= np.random.uniform(self.lower, self.upper)
        return sample

class RandomHue(object):
    def __init__(self, delta=18.0):
        assert delta >= 0.0 and delta <= 360.0
        self.delta = delta
    def __call__(self, sample):
        if np.random.randint(2):
            sample['image'][:, :, 0] += np.random.uniform(-self.delta, self.delta)
            sample['image'][:, :, 0][sample['image'][:, :, 0] > 360.0] -= 360.0
            sample['image'][:, :, 0][sample['image'][:, :, 0] < 0.0] += 360.0
        return sample
    
class SwapChannels(object):
    def __init__(self, swaps):
        self.swaps = swaps
    def __call__(self, image):
        image = image[:, :, self.swaps]
        return image

class RandomLightingNoise(object):
    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))
    def __call__(self, sample):
        if np.random.randint(2):
            swap = self.perms[np.random.randint(len(self.perms))]
            sample['image'] = sample['image'][:,:,swap]
        return sample
    
class ConvertColor(object):
    def __init__(self, current='BGR', transform='HSV'):
        self.transform = transform
        self.current = current
    def __call__(self, sample):
        if self.current == 'BGR' and self.transform == 'HSV':
            sample['image'] = cv2.cvtColor(sample['image'] , cv2.COLOR_BGR2HSV)
        elif self.current == 'HSV' and self.transform == 'BGR':
            sample['image']  = cv2.cvtColor(sample['image'] , cv2.COLOR_HSV2BGR)
        else:
            raise NotImplementedError
        return sample

class PhotometricDistort():
    def __init__(self):
        self.distort = Compose([
            RandomBrightness(),
            RandomContrast(),
            ConvertColor(transform='HSV'),
            RandomSaturation(),
            RandomHue(),
            ConvertColor(current='HSV', transform='BGR'),
            RandomLightingNoise(),
        ])
    def __call__(self, sample):
        sample['image'] = np.clip(sample['image'],0,255).astype(np.float32)
        return self.distort(sample)
    

class GenHeatmap():
    def __init__(self, num_keypoints, image_size = 256, heatmap_size = 64):
        self.num_keypoints = num_keypoints
        self.image_size = image_size
        self.heatmap_size = heatmap_size
        

    def __call__(self,sample):
        hm = np.zeros((self.num_keypoints, self.heatmap_size, self.heatmap_size), dtype=np.float32)
        reg = np.zeros((self.num_keypoints, 2), dtype=np.float32)
        ind = np.zeros((self.num_keypoints), dtype=np.int64)
        mask = np.zeros((self.num_keypoints), dtype=np.uint8)

        for i,x2d in enumerate(sample['x2d']):
            ct = x2d * self.heatmap_size/self.image_size 
            ct_int = (ct + 0.5).astype(np.int32)
            if ct_int[0] < self.heatmap_size and ct_int[1] < self.heatmap_size and ct_int[0] >= 0 and ct_int[1] >= 0:
                radius = 2
                self.draw_gaussian(hm[i], ct, radius)
                ind[i] = ct_int[1] * self.heatmap_size + ct_int[0]
                reg[i] = ct - ct_int
                mask[i] = 1

        sample['hm'] = hm
        sample['reg'] = reg
        sample['mask'] = mask
        sample['ind'] = ind
        return sample

    def gaussian2D(self, shape, sigma=1):
        m, n = [(ss - 1.) / 2. for ss in shape]
        y, x = np.ogrid[-m:m+1, -n:n+1]

        h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        return h

    def draw_gaussian(self, heatmap, center, radius):
        diameter = 2 * radius + 1
        gaussian = self.gaussian2D((diameter, diameter), sigma=diameter / 6)

        x, y = int(center[0]), int(center[1])

        height, width = heatmap.shape[0:2]

        left, right = min(x, radius), min(width - x, radius + 1)
        top, bottom = min(y, radius), min(height - y, radius + 1)

        masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
        masked_gaussian = gaussian[radius - top:radius +
                                bottom, radius - left:radius + right]
        if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
            np.maximum(masked_heatmap, masked_gaussian, out=masked_heatmap)
        return heatmap
    




            
