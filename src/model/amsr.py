from model import common
import torch.nn as nn
import torch
import torch.nn.functional as F
from model.ConvLSTM import ConvLSTM_Cell
url = {
}
def make_model(args, parent=False):
    return AMSR(args)

class Attention(nn.Module):
    def __init__(self,args, conv=common.default_conv):
        super(Attention,self).__init__()
        n_resblocks = args.n_resblocks
        self.n_feats = args.n_feats
        kernel_size = 3
        scale = args.scale[0]
        act = nn.ReLU(True)
        m_attention = [
            common.ResBlock(
                conv, self.n_feats, kernel_size, act=act, res_scale=1
            ) for _ in range(3)
        ]
        m_attention.insert(0,conv(4, self.n_feats, kernel_size))
        m_attention.append(conv(self.n_feats, self.n_feats, kernel_size))
        self.attention_res = nn.Sequential(*m_attention)
        self.attention_LSTM=ConvLSTM_Cell(self.n_feats,self.n_feats,)
        self.attention_tail=conv(self.n_feats, 1, kernel_size)
    def get_device(self):
        return torch.device("cuda" if next(self.parameters()).is_cuda else "cpu")
    def forward(self,input:torch.Tensor):
        LSTM_HL=torch.zeros(input.size(0),self.n_feats,input.size(2),input.size(3)).to(self.get_device())
        LSTM_CL=torch.zeros(input.size(0),self.n_feats,input.size(2),input.size(3)).to(self.get_device())
        Map=torch.empty(input.size(0),1,input.size(2),input.size(3)).to(self.get_device())
        torch.nn.init.constant_(Map,0.5)
        Maps=[]
        for i in range(5):
            output=torch.cat((input,Map),dim=1)
            output=self.attention_res(output)
            LSTM_HL,LSTM_CL=self.attention_LSTM(output,LSTM_HL,LSTM_CL)
            Map=self.attention_tail(LSTM_HL)
            Maps.append(Map)
        return Maps




class AMSR(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(AMSR, self).__init__()

        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3 
        scale = args.scale[0]
        act = nn.ReLU(True)
        url_name = 'r{}f{}x{}'.format(n_resblocks, n_feats, scale)
        if url_name in url:
            self.url = url[url_name]
        else:
            self.url = None
        self.sub_mean = common.MeanShift(args.rgb_range)
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)

        # define head module
        m_head = [conv(4, n_feats, kernel_size)]

        # define body module
        m_body1_L = [
            common.ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=args.res_scale
            ) for _ in range(2)
        ]
        m_body1_L.append(conv(n_feats, n_feats, kernel_size))


        m_body2_L = [
            common.Upsampler(conv, 0.5, n_feats, act=False),
        ]
        m_body2_L.extend([
            common.ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=args.res_scale
            ) for _ in range(2)
        ])
        m_body2_L.append(conv(n_feats, n_feats, kernel_size))


        m_body3_L = [
            common.Upsampler(conv, 0.5, n_feats, act=False),
        ]
        m_body3_L.extend([
            common.ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=args.res_scale
            ) for _ in range(2)
        ])
        m_body3_L.append(conv(n_feats, n_feats, kernel_size))


        m_body3_R = [
            common.ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=args.res_scale
            ) for _ in range(2)
        ]
        m_body3_R.append(conv(n_feats, n_feats, kernel_size))
        m_body3_R.append(common.Upsampler(conv, 2, n_feats, act=False))

        m_body2_R=[conv(n_feats*2, n_feats, kernel_size)]
        m_body2_R.extend([
            common.ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=args.res_scale
            ) for _ in range(2)
        ])
        m_body2_R.append(common.Upsampler(conv, 2, n_feats, act=False))

        m_body1_R=[conv(n_feats*2, n_feats, kernel_size)]
        m_body1_R.extend([
            common.ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=args.res_scale
            ) for _ in range(2)
        ])
        m_body1_R.append(conv(n_feats, n_feats, kernel_size))


        # define tail module
        m_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, args.n_colors, kernel_size)
        ]
        self.attention=Attention(args,conv)
        self.head = nn.Sequential(*m_head)
        self.body1_L = nn.Sequential(*m_body1_L)
        self.body2_L = nn.Sequential(*m_body2_L)
        self.body3_L = nn.Sequential(*m_body3_L)
        self.body3_R = nn.Sequential(*m_body3_R)
        self.body2_R = nn.Sequential(*m_body2_R)
        self.body1_R = nn.Sequential(*m_body1_R)
        self.tail = nn.Sequential(*m_tail)
    def get_device(self):
        return torch.device("cuda" if next(self.parameters()).is_cuda else "cpu")
    def forward(self, x):
        x_H=x.size(-2)//2//2
        x_W=x.size(-1)//2//2
        x = self.sub_mean(x)
        maps=self.attention(x)
        x=torch.cat((x,maps[-1]),dim=1)
        x = self.head(x)
        body1x_L=self.body1_L(x)
        body2x_L=self.body2_L(body1x_L)
        body3x_L=self.body3_L(body2x_L)
        body3x_R=self.body3_R(body3x_L)
        body2x_R=self.body2_R(torch.cat([body2x_L[:,:,:x_H*2,:x_W*2],body3x_R],dim=1))
        body1x_R=self.body1_R(torch.cat([body1x_L[:,:,:x_H*4,:x_W*4],body2x_R],dim=1))
        body1x_R=F.interpolate(body1x_R, size=(x.size(-2), x.size(-1)), mode="bicubic",align_corners=True)
        body1x_R += x
        x = self.tail(body1x_R)
        x = self.add_mean(x)
        return x,maps

    def load_state_dict(self, state_dict, strict=True):

        print("----------------------------------------------------------------------")
        own_state = self.state_dict()
        '''import torch
        attention_load = torch.load("../experiment/egisr_baseline_x2_3/model/model_best.pt")
        for name, param in attention_load.items():
            if name.find("atten")>=0 and name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                    print(name+" load success")
                except Exception:
                    print(name+" load error")'''
        for name, param in state_dict.items():
            if name in own_state:  # name.find("atten")<0 and
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                    print(name + " load success")
                except Exception:
                    print(name + " load error")
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))


if __name__ == '__main__':
    from option import args
    egisr=AMSR(args)
    imgs=torch.randn(10,3,110,110)
    size=egisr(imgs)[0].size()
    print(size)
