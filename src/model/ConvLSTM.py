import torch.nn as nn
import torch
class ConvLSTM_Cell(nn.Module):
    """
            input:[B,in_channels,H,W]
            hidden:[B,hidden_channels,H,W]
            Ct:[B,hidden_channels,H,W]
            :return ht, ct
        """

    def __init__(self, in_channels, hidden_channels, kernel_size=3):
        super(ConvLSTM_Cell, self).__init__()
        self.fConv = nn.Sequential(
            nn.Conv2d(in_channels + hidden_channels, hidden_channels, kernel_size, padding=kernel_size // 2),
            nn.Sigmoid()
        )
        self.iConv = nn.Sequential(
            nn.Conv2d(in_channels + hidden_channels, hidden_channels, kernel_size, padding=kernel_size // 2),
            nn.Sigmoid()
        )
        self.cConV = nn.Sequential(
            nn.Conv2d(in_channels + hidden_channels, hidden_channels, kernel_size, padding=kernel_size // 2),
            nn.Tanh()
        )
        self.oConv = nn.Sequential(
            nn.Conv2d(in_channels + hidden_channels, hidden_channels, kernel_size, padding=kernel_size // 2),
            nn.Sigmoid()
        )
        self.Tanh = nn.Tanh()

    def forward(self, input:torch.Tensor, hidden:torch.Tensor, cell:torch.Tensor):
        """
            input:[B,in_channels,H,W]
            hidden:[B,hidden_channels,H,W]
            cell:[B,hidden_channels,H,W]
            :return ht, ct
        """
        hInput = torch.cat([input, hidden], dim=1)
        ft = self.fConv(hInput)
        it = self.iConv(hInput)
        ct = self.cConV(hInput)
        cell_t = torch.add(torch.mul(cell, ft), torch.mul(it, ct))
        ot = self.oConv(hInput)
        ht = torch.mul(self.Tanh(cell_t), ot)
        return ht, ct


class ConvLSTM(nn.Module):
    def __init__(self, in_channels:int, hidden_channels:int, num_layers:int=1, bidirectional:bool=False):
        super(ConvLSTM, self).__init__()
        self.hidden_channels = hidden_channels
        self.in_channels = in_channels
        self.num_layers=num_layers
        self.bidirectional=bidirectional
        self.lstm0 = ConvLSTM_Cell(in_channels, hidden_channels)
        self.lstms = nn.ModuleList()
        for i in range(1,num_layers):
            self.lstms.append(ConvLSTM_Cell(hidden_channels, hidden_channels))
        if bidirectional:
            self.blstm0 = ConvLSTM_Cell(in_channels, hidden_channels)
            self.blstms = nn.ModuleList()
            for i in range(1, num_layers):
                self.blstms.append(ConvLSTM_Cell(hidden_channels, hidden_channels))
    def __devide__(self):
        return next(self.parameters()).device
    def init_hc(self,hidden:torch.Tensor=None, cell:torch.Tensor=None):
        self.layer_step=-1
        self.hidden=hidden
        self.cell=cell
    def get_net_hc(self):
        self.layer_step+=1
        assert self.layer_step<self.hidden.size(0)
        return self.hidden[self.layer_step],self.cell[self.layer_step]
    def forward(self, input:torch.Tensor, hidden:torch.Tensor=None, cell:torch.Tensor=None):
        """
            input:[SeqLen,B,in_channels,H,W]
            hidden:[num_layers,B,hidden_channels,H,W]
            cell:[num_layers,B,hidden_channels,H,W]
            :return [seq_len,B,hidden_channels,H,W]
        """
        SeqLen, B, in_channels, H, W = input.size(0), input.size(1), input.size(2), input.size(3), input.size(4)
        binput=input.clone()
        num=self.num_layers*2 if self.bidirectional else self.num_layers
        if hidden is None:
            hidden=torch.zeros(num,B,self.hidden_channels,H,W)
        if cell is None:
            cell=torch.zeros(num,B,self.hidden_channels,H,W)
        self.init_hc(hidden,cell)
        output = torch.empty(0, B, self.hidden_channels, H, W, dtype=torch.float32,device=self.__devide__())
        hn = torch.empty(0, B, self.hidden_channels, H, W, dtype=torch.float32,device=self.__devide__())
        cn = torch.empty(0, B, self.hidden_channels, H, W, dtype=torch.float32,device=self.__devide__())
        ht,ct=self.get_net_hc()
        for input_ in input:#对SqlLen的遍历
            ht, ct = self.lstm0(input_, ht, ct)
            output = torch.cat([output, ht.unsqueeze(0)], dim=0)
        hn=torch.cat([hn,ht.unsqueeze(0)])
        cn=torch.cat([cn,ct.unsqueeze(0)])
        for i in range(self.num_layers-1):
            ht,ct=self.get_net_hc()
            input = output.clone()
            output = torch.empty(0, B, self.hidden_channels, H, W, dtype=torch.float32,device=self.__devide__())
            for input_ in input:
                ht, ct = self.lstms[i](input_, ht, ct)
                output = torch.cat([output, ht.unsqueeze(0)], dim=0)
            hn = torch.cat([hn, ht.unsqueeze(0)])
            cn = torch.cat([cn, ct.unsqueeze(0)])
        #反向
        if self.bidirectional:
            bhn = torch.empty(0, B, self.hidden_channels, H, W, dtype=torch.float32,device=self.__devide__())
            bcn = torch.empty(0, B, self.hidden_channels, H, W, dtype=torch.float32,device=self.__devide__())
            bht,bct=self.get_net_hc()
            boutput = torch.empty(0, B, self.hidden_channels, H, W, dtype=torch.float32,device=self.__devide__())
            for b in range(1,binput.size(0)+1):
                bht, bct = self.lstm0(binput[-b], bht, bct)
                boutput = torch.cat([bht.unsqueeze(0),boutput], dim=0)
            bhn=torch.cat([bhn,bht.unsqueeze(0)])
            bcn=torch.cat([bcn,bct.unsqueeze(0)])
            for i in range(self.num_layers-1):
                bht,bct=self.get_net_hc()
                binput = boutput.clone()
                boutput = torch.empty(0, B, self.hidden_channels, H, W, dtype=torch.float32,device=self.__devide__())
                for binput_ in binput:
                    bht, bct = self.lstms[i](binput_, bht, bct)
                    boutput = torch.cat([boutput, bht.unsqueeze(0)], dim=0)
                bhn = torch.cat([bhn, bht.unsqueeze(0)])
                bcn = torch.cat([bcn, bct.unsqueeze(0)])
            output=torch.cat([output,boutput],dim=2)
            hn=torch.cat([hn,bhn],dim=0)
            cn=torch.cat([cn,bcn],dim=0)

        return output, (hn, cn)
if __name__ == '__main__':
    conv_lstm = ConvLSTM(3, 1, 2,bidirectional=True)
    """
                input:[SeqLen,B,in_channels,H,W]
                hidden:[num_layers,B,hidden_channels,H,W]
                cell:[num_layers,B,hidden_channels,H,W]
                :return [seq_len,B,hidden_channels,H,W]
    """
    input=torch.randn(10,5,3,100,200)
    output, (hn, cn)=conv_lstm(input)
    print(output.shape,hn.shape,cn.shape)
    pass
