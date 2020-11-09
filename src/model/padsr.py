import math
import torch
import torch.nn as nn
from model import common

url = {
    'r16f64x2': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x2-1bc95232.pt',
    'r16f64x3': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x3-abf2a44e.pt',
    'r16f64x4': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x4-6b446fab.pt',
    'r32f256x2': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x2-0edfb8a3.pt',
    'r32f256x3': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x3-ea3ef2c6.pt',
    'r32f256x4': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x4-4f62e9ef.pt'
}

def make_model(args, parent=False):
    return EDSR(args)

class EDSR(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(EDSR, self).__init__()

        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        self.scale = args.scale[0]
        act = nn.ReLU(True)
        url_name = 'r{}f{}x{}'.format(n_resblocks, n_feats, self.scale)
        if url_name in url:
            self.url = url[url_name]
        else:
            self.url = None
        self.sub_mean = common.MeanShift(args.rgb_range)
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)

        # define head module
        m_head = [conv(args.n_colors, n_feats, kernel_size)]

        # define body module
        m_body = [
            common.ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=args.res_scale
            ) for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        m_tail = [
            # common.Upsampler(conv, self.scale, n_feats, act=False),
            conv(n_feats, args.n_colors, kernel_size)
        ]
        # self.head=Meta.GLCMconv1(args.n_colors, n_feats, kernel_size, padding=(kernel_size - 1) // 2,)
        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail_1=PadSR(n_feats)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res += x
        x=self.tail_1(res,self.scale)
        x = self.tail(x)
        x = self.add_mean(x)

        return x

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))


class PadSR(nn.Module):
    def __init__(self, inC,outC=-1):
        super(PadSR, self).__init__()
        self.inC = inC
        self.outC=outC if outC!=-1 else inC
        self.scale = -1
        self.inH = -1
        self.inW = -1
        self.outH = -1
        self.outW = -1
        self.pad_H = Wconv(1, inC, padding=1)
        self.pad_W = Wconv(1, inC,self.outC, padding=1) if outC!=-1 else self.pad_H
    def get_device(self):
        return torch.device("cuda" if next(self.parameters()).is_cuda else "cpu")

    def forward(self, input, scale):
        N, inC = input.size(0), input.size(-3)
        if inC != self.inC:  # 输入合法检测
            raise Exception("inC!=self.inC")
        if self.inH != input.size(-2) or self.inW != input.size(-1) or self.scale != scale:
            self.inH = input.size(-2)
            self.inW = input.size(-1)
            self.scale = round(float(scale), 1)
            self.scale_ceil = math.ceil(scale)
            self.scale_floor = math.floor(scale)
            self.alpha=3
        self.outH = math.ceil(self.scale * self.inH)
        self.outW = math.ceil(self.scale * self.inW)
        scale_t = ((torch.arange(1, self.scale_ceil+1,dtype=torch.float32) - math.ceil(self.scale_ceil / 2)) / self.scale_ceil).unsqueeze(-1)
        # scale_t_ceil = math.atan(self.scale_ceil/self.alpha)
        # pad_num=self.scale_ceil-2
        # index_shift=math.ceil(self.scale_ceil / 2)-1
        # scale_t=torch.tensor([0]+[0+pad*scale_t_ceil/(pad_num+1) for pad in range(1,pad_num+1)]+[scale_t_ceil])
        # scale_t=(scale_t-scale_t[index_shift]).unsqueeze(-1)
        input = torch.stack(self.pad_H(input, scale_t), dim=3).view(N, self.inC, self.inH * self.scale_ceil, self.inW)
        input = torch.stack(self.pad_W(input, scale_t), dim=-1).view(N, self.outC, -1)
        mask = self.mask_made().to(self.get_device())

        input = self.mask_select(input, mask).view(N, self.outC, self.outH, self.outW)
        return input

    def mask_made(self):
        inH, inW, scale = self.inH, self.inW, self.scale
        scale_floor = self.scale_floor
        scale = round((scale - scale_floor),1)
        if scale!=0:
            addH = math.ceil(inH * scale)
            addW = math.ceil(inW * scale)
            outH_index = (torch.arange(1, addH+1, 1).div(scale) - 1).floor().int().tolist()
            outW_index = (torch.arange(1, addW+1, 1).div(scale) - 1).floor().int().tolist()
            H_mask = torch.zeros(1, inH, 1, dtype=torch.uint8)
            W_mask = torch.zeros(1, 1, inW, dtype=torch.uint8)
            for h in outH_index:
                H_mask[0, h, 0] = 1
            for w in outW_index:
                W_mask[0, 0, w] = 1
        else:
            H_mask=torch.zeros(0, inH, 1, dtype=torch.uint8)
            W_mask = torch.zeros(1, 0, inW, dtype=torch.uint8)

        pad_ = torch.ones(scale_floor, inH, 1, dtype=torch.uint8)
        H_mask = torch.cat((pad_, H_mask), dim=0).view(self.inH * self.scale_ceil, 1)
        pad_ = torch.ones(1, scale_floor, inW, dtype=torch.uint8)
        W_mask = torch.cat((pad_, W_mask), dim=1).view(1, self.inW * self.scale_ceil)

        H_mask = torch.cat([H_mask] * self.inW * self.scale_ceil, dim=1)
        W_mask = torch.cat([W_mask] * self.inH * self.scale_ceil, dim=0)
        mask = (H_mask + W_mask).eq(2)
        return mask.view(-1)

    def mask_select(self, input: torch.Tensor, mask: torch.ByteTensor):  ###保留input形状的掩码运算
        if len(mask.size()) > 1:
            raise Exception("Function:mask_select ，mask 参数 维度应为一维")
        if mask.size(0) != input.size(-1):
            raise Exception("Function:mask_select ，input.shape[0]={} ，mask.shape[0]={}, 他们不一致".format(input.shape[-1],
                                                                                                      mask.shape[0]))
        input_size = (list(input.size()[:-1]))[::-1]
        for size in input_size:
            torch.stack([mask] * size, dim=0)
        output = torch.masked_select(input, mask)
        output = output.view(*(input.size()[:-1]), -1)
        return output


class Fea2Weight(nn.Module):
    def __init__(self, feature_num, inC, kernel_size=3, outC=3):
        super(Fea2Weight, self).__init__()
        self.inC = inC
        self.kernel_size = kernel_size
        self.outC = outC
        self.meta_block = nn.Sequential(
            nn.Linear(feature_num, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, self.kernel_size * self.kernel_size * self.inC * self.outC),
        )

    def forward(self, x):
        output = self.meta_block(x)
        return output


class Wconv(nn.Module):
    def __init__(self, Feature_num, inC,outC=-1, kernel_size=3, padding=0):
        super(Wconv, self).__init__()
        self.inC = inC
        self.outC = outC if outC!=-1 else inC
        self.kernel = kernel_size
        self.padding = padding
        self.F2W = Fea2Weight(Feature_num, inC, kernel_size, self.outC)
        self.conv2d = nn.functional.conv2d
        self.Norm = nn.LayerNorm([self.inC, self.kernel, self.kernel])

    def get_device(self):
        return torch.device("cuda" if next(self.parameters()).is_cuda else "cpu")

    def forward(self, input, features):
        features = features.to(self.get_device())
        if len(features.shape) == 1:
            features = features.unsqueeze(0)
        N = features.size(0)
        weights = self.F2W(features)
        weights = weights.view(N, self.outC, self.inC, self.kernel, self.kernel)
        weights = self.Norm(weights) / self.inC
        return [self.conv2d(input, weights[i], padding=self.padding) for i in range(N)]


if __name__ == "__main__":
    img = torch.randn(1, 1, 10, 10)
    up = PadSR(1,1)
    print(up(img, 2))
