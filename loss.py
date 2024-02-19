import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class WingLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.omega = 10
        self.sigma = 2
        self.c = self.omega - self.omega * np.log(1+self.omega/self.sigma)
        
    def forward(self, pred, target):
        """
        Args:
            pred : (B,N,3)
            target: (B,N,3)
        """
        l1, loss = self.calloss(pred, target)
        l2 = torch.sqrt(((pred - target) ** 2).sum(-1)).mean(-1)

        loss_stats = {
            'loss': loss.mean().cpu().detach(),
            'diff/l1' : l1.mean().cpu().detach(),
            'diff/l2': l2.mean().cpu().detach(),
        }

        return loss.mean(), loss_stats

    def calloss(self, x,t):
        diff = torch.abs(x - t).sum(-1)
        is_small = (diff < self.omega).float() 
        small_loss = self.omega * torch.log(1+diff/self.sigma)
        big_loss = diff - self.c
        loss = (small_loss * is_small + big_loss * (1-is_small)) * 0.1
        return diff, loss

class FocalLoss(nn.Module):
    """
    Args:
        pred (B, c, h, w)
        target (B, c, h, w)
    """

    def __init__(self):
        super(FocalLoss, self).__init__()
        self.epsilon = 1e-8

    def forward(self, pred, target):
        pred = pred.clamp(min=self.epsilon, max=1-self.epsilon)
        yeq1_index = target.ge(0.9).float()
        other_index = target.lt(0.9).float()

        yeq1_loss = (yeq1_index * torch.log(pred) * torch.pow(1-pred,2)).sum()
        other_loss = (other_index * torch.log(1 - pred) * torch.pow(pred, 2) * torch.pow(1 - target, 4)).sum()
        num_yeq1 = yeq1_index.float().sum()

        if num_yeq1 == 0:
            loss = - other_loss
        else:
            loss = - (yeq1_loss + other_loss) / num_yeq1

        return loss

def _tranpose_and_gather_feat(feat, ind):
    """
    Args:
        feat (B,C*2,H,W)
        ind (B,C)
    Returns:
        feat (B,C,2)
    """
    # (B,C*2,H,W) -> (B,C,2,H*W)
    feat = feat.view(feat.size(0), feat.size(1)//2, 2, -1)
    return torch.gather(feat, 3, ind.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 2, -1)).squeeze(3)

class RegL1Loss(nn.Module):
    """
    Args:
        output (B, dim, h, w)
        mask (B, max_obj)
        ind (B, max_obj)
        target (B, max_obj, dim)
    Temp:
        pred (B, max_obj, dim)
    """

    def __init__(self):
        super(RegL1Loss, self).__init__()

    def forward(self, output, mask, ind, target):
        pred = _tranpose_and_gather_feat(output, ind)

        mask = mask.unsqueeze(2).expand_as(pred).float()
        loss = F.l1_loss(pred * mask, target * mask, reduction='sum')
        loss = loss / (mask.sum() + 1e-4)
        return loss
    
class DetLoss(nn.Module):
    def __init__(self,cfg):
        super(DetLoss, self).__init__()
        self.cfg=cfg
        self.hm_crit = FocalLoss()
        self.reg_crit = RegL1Loss()

    def forward(self, output, batch):

        hm_loss = self.hm_crit(output['hm'], batch['hm'])
        reg_loss = self.reg_crit(output['reg'],batch['mask'],batch['ind'], batch['reg'])
        loss =  hm_loss + reg_loss

        B,C,H,W = output['hm'].shape
        max_val, max_idx = torch.max(output['hm'].view(B, C, -1), dim=2)
        # reg = _tranpose_and_gather_feat(output['reg'], max_idx)
        reg = torch.gather(output['reg'].view(B,C,2,H*W), 3, max_idx.unsqueeze(-1).unsqueeze(-1).expand(-1,-1,2,-1)).squeeze(3)

        x = (torch.stack([max_idx%W, max_idx//W],dim=-1) + reg) * self.cfg['image_size'] / self.cfg['heatmap_size']
        l1 = torch.abs(x - batch['x2d']).sum(-1).mean(-1)
        l2 = torch.sqrt(((x - batch['x2d']) ** 2).sum(-1)).mean(-1)        


        loss_stats = {
            'loss': loss.mean().cpu().detach(),
            'loss/hm' : hm_loss.mean().cpu().detach(),
            'loss/reg': reg_loss.mean().cpu().detach(),
            'diff/l1': l1.mean().cpu().detach(),
            'diff/l2': l2.mean().cpu().detach(),
        }
        output['pred_x2d'] = torch.cat([x,max_val[...,None]],dim=-1)
        return loss.mean(), loss_stats
    
class CocktailLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.hm_crit = FocalLoss()
        # self.reg_crit = RegL1Loss()

        self.omega = 25
        self.sigma = 5
        self.c = self.omega - self.omega * np.log(1+self.omega/self.sigma)


    def forward(self, output, batch):

        # 2d
        hm_loss, reg_loss = 0., 0.
        B,V = batch['hm'].shape[:2]
        # print(output['hm'].shape, batch['hm'].shape, output['reg'].shape, batch['mask'].shape, batch['ind'].shape, batch['reg'].shape)
        for i in range(V):
            hm_loss += self.hm_crit(output['hm'][:,i], batch['hm'][:,i]) / V
            # reg_loss += self.reg_crit(output['reg'][:,i],batch['mask'][:,i],batch['ind'][:,i], batch['reg'][:,i]) / V

        # 3d 
        pred = output['x3d']
        target = batch['x3d'][:,0]
        x3d_l1, x3d_loss = self.calloss(pred, target)
        x3d_l2 = torch.sqrt(((pred - target) ** 2).sum(-1)).mean(-1)

        x2d_l1 = torch.abs(output['x2d'] - batch['x2d']).sum(-1).mean(-1)
        x2d_l2 = torch.sqrt(((output['x2d'] - batch['x2d']) ** 2).sum(-1)).mean(-1)

        loss = (x3d_loss + hm_loss)*0.5

        loss_stats = {
            'loss': loss.mean().cpu().detach(),
            'loss/hm' : hm_loss.mean().cpu().detach(),
            # 'loss/reg': reg_loss.mean().cpu().detach(),
            'loss/x3d': x3d_loss.mean().cpu().detach(),
            'x2d/l1': x2d_l1.mean().cpu().detach(),
            'x2d/l2': x2d_l2.mean().cpu().detach(),
            'x3d/l1' : x3d_l1.mean().cpu().detach(),
            'x3d/l2': x3d_l2.mean().cpu().detach(),
        }

        return loss.mean(), loss_stats

    def calloss(self, x,t):
        diff = torch.abs(x - t).sum(-1)
        is_small = (diff < self.omega).float() 
        small_loss = self.omega * torch.log(1+diff/self.sigma)
        big_loss = diff - self.c
        loss = (small_loss * is_small + big_loss * (1-is_small)) * 0.1
        return diff, loss




class Net2d(nn.Module):
    def __init__(self, cfg, model):
        super().__init__()
        self.model = model.to(cfg['device'])
        self.loss = DetLoss(cfg).to(cfg['device'])

    def forward(self, batch):
        out_hm, out_reg, _ = self.model(batch['image'])
        output = {'hm':out_hm,'reg':out_reg}
        loss, loss_stats = self.loss(output, batch)
        return output, loss, loss_stats


class Net3d(nn.Module):
    def __init__(self, cfg, model):
        super().__init__()
        self.model = model.to(cfg['device'])
        self.loss = CocktailLoss().to(cfg['device'])

    def forward(self, batch):
        out_x3d, out_hm, out_reg, out_x2d = self.model(batch['image'],batch['K'])
        output = {'x3d':out_x3d,'hm':out_hm,'reg':out_reg,'x2d':out_x2d}
        loss, loss_stats = self.loss(output, batch)
        return output, loss, loss_stats
    
class NetPose(nn.Module):
    def __init__(self, cfg, model):
        super().__init__()
        self.model = model.to(cfg['device'])
        self.loss = WingLoss().to(cfg['device'])

    def forward(self, batch):
        out = self.model(batch['xdir'])
        loss, loss_stats = self.loss(out, batch['x3d'][:,0])
        return out, loss, loss_stats


class NetTri(nn.Module):
    def __init__(self, cfg, model):
        super().__init__()
        self.model = model.to(cfg['device'])
        self.loss = WingLoss().to(cfg['device'])

    def forward(self, batch):
        out = self.model(batch['xdir'], batch['xfeat'])
        loss, loss_stats = self.loss(out, batch['x3d'][:,0])
        return out, loss, loss_stats

# class Net(nn.Module):
#     def __init__(self, cfg, model):
#         super().__init__()
#         self.model = model.to(cfg['device'])
#         self.loss = WingLoss().to(cfg['device'])

#     def forward(self, batch):
#         lds_pred,_ = self.model(batch['image'])
#         lds_pred = lds_pred.squeeze()
#         loss, loss_stats = self.loss(lds_pred, batch['x2d'])
#         return lds_pred, loss, loss_stats


        