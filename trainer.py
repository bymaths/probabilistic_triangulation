import torch
from tqdm import tqdm
import tensorboardX
import time
import os

# EMA
class EMA():
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}
        

# AverageMeter
class AverageMeter():
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count > 0:
            self.avg = self.sum / self.count

# Logger
class Logger():
    def __init__(self, cfg, is_train):
        if not os.path.exists(cfg['save_dir']):
            os.makedirs(cfg['save_dir'])

        time_str = time.strftime('%Y-%m-%d-%H-%M')
        if is_train:
            save_path = os.path.join(cfg['save_dir'],'trainlogs')
        else:
            save_path = os.path.join(cfg['save_dir'],'testlogs')

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        log_dir = os.path.join(save_path, 'logs_{}'.format(time_str))    
        print(log_dir)
        self.writer = tensorboardX.SummaryWriter(log_dir = log_dir)

    def scalar_summay(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)


# Trainer
class Trainer():
    def __init__(self, cfg, net):
        self.cfg = cfg
        self.net = net
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=cfg['lr'])
        
        # for state in self.optimizer.state.values():
        #     for k, v in state.items():
        #         if isinstance(v, torch.Tensor):
        #             state[k] = v.to(device=cfg['device'], non_blocking=True)

        self.ema = EMA(self.net, 0.999)
        self.ema.register()
        
        self.train_logger = Logger(cfg, is_train=True)
        self.test_logger = Logger(cfg, is_train=False)
        
    def run(self, train_loader, test_loader):
        start_epoch = 0
        for epoch in range(start_epoch + 1, self.cfg['num_epoch']+1):
            # train
            log_dict = self.run_epoch(train_loader, is_train=True)
            for k, v in log_dict.items():
                print('train epoch=', epoch, k, v.avg)
                self.train_logger.scalar_summay(k, v.avg, epoch)
                
            # test
            log_dict = self.run_epoch(test_loader, is_train=False)
            for k, v in log_dict.items():
                print('test epoch=', epoch, k, v.avg)
                self.test_logger.scalar_summay(k, v.avg, epoch)

            if hasattr(self.net.model, 'module'):
                model_state_dict = self.net.model.module.state_dict()
            else:
                model_state_dict = self.net.model.state_dict()

            save_path = os.path.join(self.cfg['save_dir'], f'model_{epoch}.pth')
            
            torch.save(model_state_dict, save_path)
            

    def run_epoch(self, data_loader, is_train=True):
        if is_train:
            self.net.train()
            torch.set_grad_enabled(True)
            
        else:
            self.net.eval()
            torch.set_grad_enabled(False)
            
        avg_loss_stats = {key: AverageMeter() for key in ['time/data', 'time/infer']}

        t0 = time.time()
        for iter_id, batch in tqdm(enumerate(data_loader)):
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device = self.cfg['device'], non_blocking=True)
                
            t1 = time.time()
            avg_loss_stats['time/data'].update(t1-t0)
            t0 = time.time()

            output, loss, loss_stats = self.net(batch)

            t1 = time.time()
            avg_loss_stats['time/infer'].update(t1-t0)
            t0 = time.time()

            if is_train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.ema.update()

            if 'loss' not in avg_loss_stats:
                for key in loss_stats:
                    avg_loss_stats[key] = AverageMeter()
                    
            for key in loss_stats:
                avg_loss_stats[key].update(loss_stats[key].item(), data_loader.batch_size)
            
        return avg_loss_stats
        