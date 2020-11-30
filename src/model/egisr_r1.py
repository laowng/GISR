from model import common
import torch.nn as nn
import torch
from model.ConvLSTM import ConvLSTM_Cell
url = {
}

def make_model(args, parent=False):
    return EGISR(args)

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
        for i in range(3):
            output=torch.cat((input,Map),dim=1)
            output=self.attention_res(output)
            LSTM_HL,LSTM_CL=self.attention_LSTM(output,LSTM_HL,LSTM_CL)
            Map=self.attention_tail(LSTM_HL)
            Maps.append(Map)
        return Maps




class EGISR(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(EGISR, self).__init__()

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
        m_body = [
            common.ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=args.res_scale
            ) for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        m_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, args.n_colors, kernel_size)
        ]
        self.attention=Attention(args,conv)
        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)
    def get_device(self):
        return torch.device("cuda" if next(self.parameters()).is_cuda else "cpu")
    def forward(self, x):
        x = self.sub_mean(x)
        maps=self.attention(x)
        x=torch.cat((x,maps[-1]),dim=1)
        x = self.head(x)
        res = self.body(x)
        res += x
        x = self.tail(res)
        x = self.add_mean(x)

        return x,maps

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1 and name.find('head') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1 and name.find('head') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))
if __name__ == '__main__':
    from option import args
    egisr=EGISR(args)
    imgs=torch.randn(10,3,100,100)
    size=egisr(imgs).size()
    print(size)
