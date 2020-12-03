import os
import math
import time
import datetime
from multiprocessing import Process
from multiprocessing import Queue

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import imageio
from torchvision.transforms import ToPILImage
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
import ssim
toImg=ToPILImage()
class timer():
    def __init__(self):
        self.acc = 0
        self.tic()

    def tic(self):
        self.t0 = time.time()

    def toc(self, restart=False):
        diff = time.time() - self.t0
        if restart: self.t0 = time.time()
        return diff

    def hold(self):
        self.acc += self.toc()

    def release(self):
        ret = self.acc
        self.acc = 0

        return ret

    def reset(self):
        self.acc = 0
def bg_target(queue):
    while True:
        if not queue.empty():
            filename, img = queue.get()
            if filename is None: break
            img.save(filename)
class checkpoint():
    def __init__(self, args):
        self.args = args
        self.ok = True
        self.log = torch.Tensor()
        self.ssim_log=torch.Tensor()
        now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')

        if not args.load:
            if not args.save:
                args.save = now
            self.dir = os.path.join('..', 'experiment', args.save)
        else:
            self.dir = os.path.join('..', 'experiment', args.load)
            if os.path.exists(self.dir):
                self.log = torch.load(self.get_path('psnr_log.pt'))
                self.ssim_log=torch.load(self.get_path('ssim_log.pt'))
                print('Continue from epoch {}...'.format(len(self.log)))
            else:
                args.load = ''

        if args.reset:
            os.system('rm -rf ' + self.dir)
            args.load = ''

        os.makedirs(self.dir, exist_ok=True)
        os.makedirs(self.get_path('model'), exist_ok=True)
        for d in args.data_test:
            os.makedirs(self.get_path('results-{}'.format(d)), exist_ok=True)

        open_type = 'a' if os.path.exists(self.get_path('log.txt'))else 'w'
        self.log_file = open(self.get_path('log.txt'), open_type)
        with open(self.get_path('config.txt'), open_type) as f:
            f.write(now + '\n\n')
            for arg in vars(args):
                f.write('{}: {}\n'.format(arg, getattr(args, arg)))
            f.write('\n')
        self.n_processes = 8
    def get_path(self, *subdir):
        return os.path.join(self.dir, *subdir)

    def save(self, trainer, epoch, is_best=False):
        trainer.model.save(self.get_path('model'), epoch, is_best=is_best)
        trainer.loss.save(self.dir)
        trainer.loss.plot_loss(self.dir, epoch)
        self.plot_psnr(epoch)
        self.plot_ssim(epoch)
        self.plot_psnr_ssim(epoch)
        trainer.optimizer.save(self.dir)
        torch.save(self.log, self.get_path('psnr_log.pt'))
        torch.save(self.ssim_log, self.get_path('ssim_log.pt'))

    def add_log(self, log):
        self.log = torch.cat([self.log, log])
        self.ssim_log = torch.cat([self.ssim_log, log])

    def write_log(self, log, refresh=False,verbose=True):
        if verbose:print(log)
        self.log_file.write(log + '\n')
        if refresh:
            self.log_file.close()
            self.log_file = open(self.get_path('log.txt'), 'a')

    def done(self):
        self.log_file.close()

    def plot_psnr(self, epoch):
        axis = np.linspace(1, epoch, epoch)
        for idx_data, d in enumerate(self.args.data_test):
            label = 'SR on {}'.format(d)
            fig = plt.figure()
            plt.title(label)
            for idx_scale, scale in enumerate(self.args.scale):
                plt.plot(
                    axis,
                    self.log[:, idx_data, idx_scale].numpy(),
                    label='Scale {}'.format(scale)
                )
            plt.legend()
            plt.xlabel('Epochs')
            plt.ylabel('PSNR')
            plt.grid(True)
            plt.savefig(self.get_path('test_PSNR_{}.pdf'.format(d)))
            plt.close(fig)
    def plot_ssim(self, epoch):
        axis = np.linspace(1, epoch, epoch)
        for idx_data, d in enumerate(self.args.data_test):
            label = 'SR on {}'.format(d)
            fig = plt.figure()
            plt.title(label)
            for idx_scale, scale in enumerate(self.args.scale):
                plt.plot(
                    axis,
                    self.ssim_log[:, idx_data, idx_scale].numpy(),
                    label='Scale {}'.format(scale)
                )
            plt.legend()
            plt.xlabel('Epochs')
            plt.ylabel('SSIM')
            plt.grid(True)
            plt.savefig(self.get_path('test_SSIM_{}.pdf'.format(d)))
            plt.close(fig)
    def plot_psnr_ssim(self,epoch):
        axis = np.linspace(1, epoch, epoch)
        for idx_data, d in enumerate(self.args.data_test):
            label = 'SR on {}'.format(d)
            fig = plt.figure()
            ax=fig.add_subplot(111)
            ax.set_title(label)
            ax1=ax.twinx()
            for idx_scale, scale in enumerate(self.args.scale):
                ax.plot(
                    axis,
                    self.log[:, idx_data, idx_scale].numpy(),
                    label='PSNR_Scale {}'.format(scale),
                    color="red"
                )
                ax1.plot(
                    axis,
                    self.ssim_log[:, idx_data, idx_scale].numpy(),
                    label='SSIM_Scale {}'.format(scale),
                    color="blue"
                )

            fig.legend(loc=1)
            ax.set_xlabel('Epochs')
            ax.set_ylabel('PSNR')
            ax1.set_ylabel('SSIM')
            ax.grid(True)
            ax1.grid(True)
            plt.savefig(self.get_path('test{}.pdf'.format(d)),dpi=200, bbox_inches='tight')
            plt.close(fig)


    def begin_background(self):
        self.queue = Queue()
        self.process = [
            Process(target=bg_target, args=(self.queue,)) \
            for _ in range(self.n_processes)
        ]
        
        for p in self.process: p.start()

    def end_background(self):
        for _ in range(self.n_processes): self.queue.put((None, None))
        while not self.queue.empty(): time.sleep(1)
        for p in self.process: p.join()

    def save_results(self, dataset, filename, save_list, scale):
        if self.args.save_results:
            filename = self.get_path(
                'results-{}'.format(dataset.dataset.name),
                '{}_x{}_'.format(filename, scale)
            )
            postfix = ('SR', 'LR', 'HR',"map1","map2","map3","map4","map5","map6")
            for v, p in zip(save_list, postfix):
                tensor_cpu = toImg(v[0].cpu().byte())
                self.queue.put(('{}{}.png'.format(filename, p), tensor_cpu))

def quantize(img, rgb_range):
    pixel_range = 255 / rgb_range
    return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)

def calc_psnr(sr, hr, scale, rgb_range, dataset=None):
    if hr.nelement() == 1: return 0

    diff = (sr - hr) / rgb_range
    # print(sr.shape,hr.shape,diff.sum())
    if dataset and dataset.dataset.benchmark:
        shave = scale
        if diff.size(1) > 1:
            gray_coeffs = [65.738, 129.057, 25.064]
            convert = diff.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
            diff = diff.mul(convert).sum(dim=1)
    else:
        shave = scale + 6

    valid = diff[..., shave:-shave, shave:-shave]
    mse = valid.pow(2).mean()

    return -10 * math.log10(mse)
def cac_ssim(img1,img2,scale,rgb_range,dataset=None):
    def rgb_to_ycbcr(image: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(image):
            raise TypeError("Input type is not a torch.Tensor. Got {}".format(
                type(image)))

        if len(image.shape) < 3 or image.shape[-3] != 3:
            raise ValueError("Input size must have a shape of (*, 3, H, W). Got {}"
                             .format(image.shape))

        r: torch.Tensor = image[..., 0, :, :]
        g: torch.Tensor = image[..., 1, :, :]
        b: torch.Tensor = image[..., 2, :, :]

        delta = .5
        y: torch.Tensor = .257 * r + .504 * g + .098 * b + 16
        cb: torch.Tensor = (-.148) * r - .291 * g + .439 * b + 128
        cr: torch.Tensor = (0.439) * r - .368 * g - .071 * b + 128
        return torch.stack((y, cb, cr), -3)
    if dataset and dataset.dataset.benchmark:
        shave = scale
        img1=rgb_to_ycbcr(img1)[:,0:1,:,:]
        img2=rgb_to_ycbcr(img2)[:,0:1,:,:]
    else:
        shave = scale + 6
    sr=img1[:,:,shave:-shave,shave:-shave]
    hr=img2[:,:,shave:-shave,shave:-shave]
    return ssim.ssim(sr,hr,11,reduction="mean",max_val=rgb_range).cpu()
def make_optimizer(args, target):
    '''
        make optimizer and scheduler together
    '''
    # optimizer
    trainable = filter(lambda x: x.requires_grad, target.parameters())
    kwargs_optimizer = {'lr': args.lr, 'weight_decay': args.weight_decay}

    if args.optimizer == 'SGD':
        optimizer_class = optim.SGD
        kwargs_optimizer['momentum'] = args.momentum
    elif args.optimizer == 'ADAM':
        optimizer_class = optim.Adam
        kwargs_optimizer['betas'] = args.betas
        kwargs_optimizer['eps'] = args.epsilon
    elif args.optimizer == 'RMSprop':
        optimizer_class = optim.RMSprop
        kwargs_optimizer['eps'] = args.epsilon

    # scheduler
    milestones = list(map(lambda x: int(x), args.decay.split('-')))
    kwargs_scheduler = {'milestones': milestones, 'gamma': args.gamma}
    scheduler_class = lrs.MultiStepLR

    class CustomOptimizer(optimizer_class):
        def __init__(self, *args, **kwargs):
            super(CustomOptimizer, self).__init__(*args, **kwargs)

        def _register_scheduler(self, scheduler_class, **kwargs):
            self.scheduler = scheduler_class(self, **kwargs)

        def save(self, save_dir):
            torch.save(self.state_dict(), self.get_dir(save_dir))

        def load(self, load_dir, epoch=1):
            self.load_state_dict(torch.load(self.get_dir(load_dir)))
            if epoch > 1:
                for _ in range(epoch): self.scheduler.step()

        def get_dir(self, dir_path):
            return os.path.join(dir_path, 'optimizer.pt')

        def schedule(self):
            self.scheduler.step()

        def get_lr(self):
            return self.scheduler.get_last_lr()[0]#get_lr()[0]

        def get_last_epoch(self):
            return self.scheduler.last_epoch
    
    optimizer = CustomOptimizer(trainable, **kwargs_optimizer)
    optimizer._register_scheduler(scheduler_class, **kwargs_scheduler)
    return optimizer

