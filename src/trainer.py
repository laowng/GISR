import os
import math
from decimal import Decimal

import utility

import torch
import torch.nn.utils as utils
from tqdm import tqdm

class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args
        self.scale = args.scale

        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss
        self.optimizer = utility.make_optimizer(args, self.model)

        if self.args.load != '':
            self.optimizer.load(self.ckp.dir, epoch=len(self.ckp.log))

        self.error_last = 1e8

    def train(self):
        self.loss.step()
        epoch = self.optimizer.get_last_epoch() + 1
        lr = self.optimizer.get_lr()

        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        self.loss.start_log()
        self.model.train()

        timer_data, timer_model = utility.timer(), utility.timer()
        # TEMP
        self.loader_train.dataset.set_scale(0)
        for batch, (lr, hr, _,) in enumerate(self.loader_train):
            lr, hr = self.prepare(lr, hr)
            timer_data.hold()
            timer_model.tic()

            self.optimizer.zero_grad()
            sr = self.model(lr, 0)
            loss = float(self.args.loss_rate[0])*self.loss(sr, hr)+float(self.args.loss_rate[1])*self.loss(lr,torch.nn.functional.interpolate(sr, size=(lr.size(-2),lr.size(-1)), mode="bicubic", align_corners=False))
            loss.backward()
            if self.args.gclip > 0:
                utils.clip_grad_value_(
                    self.model.parameters(),
                    self.args.gclip
                )
            self.optimizer.step()

            timer_model.hold()

            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))

            timer_data.tic()

        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]
        self.optimizer.schedule()

    def test(self):
        torch.set_grad_enabled(False)

        epoch = self.optimizer.get_last_epoch()
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(
            torch.zeros(1, len(self.loader_test), len(self.scale))
        )
        self.model.eval()

        timer_test = utility.timer()
        if self.args.save_results: self.ckp.begin_background()
        result= {}
        result_ssim= {}
        for idx_data, d in enumerate(self.loader_test):
            for idx_scale, scale in enumerate(self.scale):
                d.dataset.set_scale(idx_scale)
                result['[{} x{}]'.format(d.dataset.name,scale,)]={}
                result_ssim['[{} x{}]'.format(d.dataset.name,scale,)]={}
                for lr, hr, filename in tqdm(d, ncols=80):
                    lr, hr = self.prepare(lr, hr)
                    sr = self.model(lr, idx_scale)
                    sr = utility.quantize(sr, self.args.rgb_range)
                    save_list = [sr]
                    psnr=utility.calc_psnr(
                        sr, hr, scale, self.args.rgb_range, dataset=d
                    )
                    ssim=utility.cac_ssim(sr,hr,scale,self.args.rgb_range,dataset=d)
                    self.ckp.log[-1, idx_data, idx_scale] +=psnr
                    self.ckp.ssim_log[-1, idx_data, idx_scale]+=ssim

                    if self.args.save_gt:
                        save_list.extend([lr, hr])
                    if self.args.save_results:
                        result['[{} x{}]'.format(d.dataset.name, scale, )][str(filename)] = psnr
                        result_ssim['[{} x{}]'.format(d.dataset.name, scale, )][str(filename)] = ssim
                        self.ckp.save_results(d, filename[0], save_list, scale)

                self.ckp.log[-1, idx_data, idx_scale] /= len(d)
                self.ckp.ssim_log[-1, idx_data, idx_scale] /= len(d)
                best = self.ckp.log.max(0)
                ssim_best = self.ckp.ssim_log.max(0)
                self.ckp.write_log(
                    '[{} x{}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                        d.dataset.name,
                        scale,
                        self.ckp.log[-1, idx_data, idx_scale],
                        best[0][idx_data, idx_scale],
                        best[1][idx_data, idx_scale] + 1
                    )
                )
                self.ckp.write_log(
                    '[{} x{}]\tSSIM: {:.4f} (Best: {:.4f} @epoch {})'.format(
                        d.dataset.name,
                        scale,
                        self.ckp.ssim_log[-1, idx_data, idx_scale],
                        ssim_best[0][idx_data, idx_scale],
                        ssim_best[1][idx_data, idx_scale] + 1
                    )
                )

        if self.args.save_results:
            for data_name in result:
                self.ckp.write_log("\n")
                self.ckp.write_log("{}".format(data_name))
                for img_name in result[data_name]:
                    self.ckp.write_log("{}:\t{}".format(img_name,result[data_name][img_name]))
            for data_name in result:
                self.ckp.write_log("\n")
                self.ckp.write_log("{}".format(data_name))
                for img_name in result[data_name]:
                    self.ckp.write_log("{}:\t{}".format(img_name,result_ssim[data_name][img_name]))

        self.ckp.write_log('Forward: {:.2f}s\n'.format(timer_test.toc()))
        self.ckp.write_log('Saving...')

        if self.args.save_results:
            self.ckp.end_background()

        if not self.args.test_only:
            self.ckp.save(self, epoch, is_best=(best[1][0, 0] + 1 == epoch))

        self.ckp.write_log(
            'Total: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )

        torch.set_grad_enabled(True)

    def prepare(self, *args):
        device = torch.device('cpu' if self.args.cpu else 'cuda')
        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(a) for a in args]

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.optimizer.get_last_epoch() + 1
            return epoch >= self.args.epochs

