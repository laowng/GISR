import math
import torch
import torch.nn as nn
import time
# from memory_profiler import profile
import torch.utils.checkpoint as cp
from model.feature_e import GLCM


class Fea2Weight(nn.Module):
    def __init__(self,feature_num,inC, kernel_size=3, outC=3):
        super(Fea2Weight,self).__init__()
        self.inC = inC
        self.kernel_size=kernel_size
        self.outC = outC
        self.meta_block=nn.Sequential(
            nn.Linear(feature_num,256),
            nn.ReLU(inplace=True),
            nn.Linear(256,self.kernel_size*self.kernel_size*self.inC*self.outC),
            # nn.LayerNorm(self.kernel_size*self.kernel_size*self.inC*self.outC)
        )
    def forward(self, x):
        output = self.meta_block(x)
        # print(next(self.meta_block.parameters()).device)
        return output

class UpSample(nn.Module):
    def __init__(self,inC,outC):
        super(UpSample,self).__init__()
        self.inC=inC
        self.outC = outC
        self.scale=-1
        self.inH=-1
        self.inW=-1
        self.outH=-1
        self.outW=-1
        self.unfold=nn.Unfold(3,padding=2,dilation=2)
        self.P2W = Fea2Weight(feature_num=3,inC=self.inC, outC=self.outC)
        self.Norm = nn.LayerNorm(3*3*self.inC)
    def get_device(self):
        return torch.device("cuda" if next(self.parameters()).is_cuda else "cpu")

    def forward(self,input:torch.Tensor,scale):
        inC=input.size(-3)
        if inC != self.inC:  # 输入合法检测
            raise Exception("inC!=self.inC")
        if self.inH!=input.size(-2) or self.inW!=input.size(-1)or self.scale != scale:
            self.inH=input.size(-2)
            self.inW=input.size(-1)
            self.scale = round(float(scale), 4)
            self.matmask = self.matmask_made(self.inH, self.inW)
        self.outH = math.ceil(self.scale * self.inH)
        self.outW = math.ceil(self.scale * self.inW)
        self.cycle=self.get_cirlen(scale)[0]
        posmat = self.Scale2Pos_mat(self.scale)

        weights = self.P2W(posmat.to(self.get_device()))
        weights = weights.view(weights.size(0), self.outC,-1 ).to(self.get_device())  # [cycleH*cycleW]xoutCx*inC]*[kH*kW
        weights = self.Norm(weights)/self.inC
        # weights = weights.view(weights.size(0), self.outC,-1 ).to(self.get_device())  # [cycleH*cycleW]xoutCx[kH*kW*inC]
        weights=weights.permute(0,2,1)# [cycleH*cycleW]x[kH*kW*inC]xoutC
        weights = weights.view(self.cycle,self.cycle,-1,self.outC)#cycleH x cycleW x[kH*kW*inC]xoutC

        up_x=self.up_x(input,self.matmask.to(self.get_device()))#N  xH_len x W_len x cycleH x cycleW x 1 x [inC * kH * kW]
        # t1=time.time()
        up_x=torch.matmul(up_x,weights)#N  xH_len x W_len x cycleH x cycleW x 1 x outC
        up_x=up_x.permute(0,6,1,3,2,4,5)#N x outC x H_len x cycleH x W_len x cycleW x 1
        up_x=up_x.contiguous().view(up_x.size(0),up_x.size(1),up_x.size(2)*up_x.size(3),up_x.size(4)*up_x.size(5))#N x outC x [H_len*cycleH] x [W_len*cycleW]
        up_x=up_x[:,:,:self.outH,:self.outW]
        # print(time.time()-t1)
        return up_x

    # @profile
    def up_x(self,x,maskmat):
        scale_int = math.ceil(self.scale)
        N,C,H,W = x.size()
        H_=self.outH%self.cycle
        W_=self.outW%self.cycle
        add_H=self.cycle-H_ if H_<self.cycle else 0
        add_W=self.cycle-W_ if W_<self.cycle else 0
        up_x = x.view(N,C,H,1,W,1)
        up_x = torch.cat([up_x]*scale_int,3)
        up_x = torch.cat([up_x]*scale_int,5).view(N,C,H*scale_int,W*scale_int)# N x inC x outH_ xoutW_
        # up_x=cp.checkpoint(self.unfold,up_x)
        up_x = self.unfold(up_x)#N x [inC * kH * kW] x [outH_*outW_]
        up_x = self.mask_select(up_x , maskmat)#N x [inC * kH * kW] x [outH*outW]
        up_x=up_x.view(N,-1,self.outH,self.outW)#N  x [inC * kH * kW] x outH x outW
        padding = nn.ZeroPad2d(padding=(0, add_W, 0, add_H))
        up_x = padding(up_x)
        up_x = up_x.view(N,-1,up_x.size(2)//self.cycle,self.cycle,up_x.size(3)//self.cycle,self.cycle,1)#N x [inC * kH * kW] x lenH x cycle x lenW x cycle x 1
        up_x=up_x.permute(0,2,4,3,5,6,1)
        return up_x#N x [outH*outW] x 1 x [inC * kH * kW]
    def get_cirlen(self,scale: float):  # 求posmat的循环部分
        # scale = round(float(scale), 4)
        for i in [10, 100, 1000, 10000]:
            x = i
            y = scale * x
            if y - math.floor(y) == 0:
                break
        """该函数返回两个数的最大公约数"""
        def GCU(m, n):
            if not m:
                return n
            elif not n:
                return m
            elif m is n:
                return m

            while m != n:
                if m > n:
                    m -= n
                else:
                    n -= m
            return m

        divisor = GCU(x, y)
        return int(y / divisor), int(x / divisor)
    def Scale2Pos_mat(self,scale: float):
          ###posmat制作
        outH = self.cycle
        outW = self.cycle
        h_project_coord = torch.arange(0, outH, 1).float().mul(1.0 / scale)
        w_project_coord = torch.arange(0, outW, 1).float().mul(1.0 / scale)
        int_h_project_coord = torch.floor(h_project_coord)
        int_w_project_coord = torch.floor(w_project_coord)
        offset_h = (h_project_coord - int_h_project_coord).unsqueeze(dim=1)
        offset_w = (w_project_coord - int_w_project_coord).unsqueeze(dim=0)
        offset_h = torch.cat([offset_h] * outW, dim=-1).unsqueeze(-1)
        offset_w = torch.cat([offset_w] * outH, dim=0).unsqueeze(-1)
        posmat = torch.cat((offset_h, offset_w), dim=-1).view(-1,2)

        # add_scale
        scale_mat = torch.tensor([[1 / scale]])
        scale_mat = torch.cat([scale_mat] * posmat.size(0), dim=0)
        posmat = torch.cat((scale_mat, posmat), dim=-1)
        return posmat
    def matmask_made(self,inh, inw):
        scale_int=math.ceil(self.scale)
        outH=math.ceil(self.scale*inh)
        outW=math.ceil(self.scale*inw)
        h_project_coord = torch.arange(0, outH, 1).float()
        w_project_coord = torch.arange(0, outW, 1).float()
        h_project_coord[-1] = h_project_coord[-1] - 0.1
        h_project_coord=h_project_coord.mul(1.0 / self.scale)
        int_h_project_coord = torch.floor(h_project_coord).int().tolist()
        w_project_coord[-1] = w_project_coord[-1] - 0.1
        w_project_coord=w_project_coord.mul(1.0 / self.scale)
        int_w_project_coord = torch.floor(w_project_coord).int().tolist()
        inH = inh
        inW = inw
        offset_h_mask = torch.zeros(inH, scale_int, 1)
        offset_w_mask = torch.zeros(1, inW, scale_int)

        number = 0
        sca = 0
        for coord in int_h_project_coord[:]:
            if coord != number:
                number += 1
                sca = 0
            offset_h_mask[coord, sca, 0] = 1
            sca += 1
        offset_h_mask = offset_h_mask.view(-1, 1)
        offset_h_mask = torch.cat([offset_h_mask] * inW * scale_int, dim=-1)

        number = 0
        sca = 0
        for coord in int_w_project_coord:
            if coord != number:
                number += 1
                sca = 0
            offset_w_mask[0, coord, sca] = 1
            sca += 1
        offset_w_mask = offset_w_mask.view(1, -1)
        offset_w_mask = torch.cat([offset_w_mask] * inH * scale_int, dim=0)
        maskmat = (offset_h_mask + offset_w_mask).eq(2).contiguous().view(-1, 1).squeeze()

        return maskmat
    def mask_select(self,input:torch.Tensor,mask:torch.ByteTensor):###保留input形状的掩码运算
        if len(mask.size())>1:
            raise Exception("Function:mask_select ，mask 参数 维度应为一维")
        if mask.size(0)!=input.size(-1):
            raise Exception("Function:mask_select ，input.shape[0]={} ，mask.shape[0]={}, 他们不一致".format(input.shape[-1],mask.shape[0]))
        input_size=(list(input.size()[:-1]))[::-1]
        for size in input_size:
            torch.stack([mask] * size, dim=0)
        output=torch.masked_select(input,mask)
        output=output.view(*(input.size()[:-1]),-1)
        return output

class Wconv1(nn.Module):
    def __init__(self,Feature_num,inC,outC=3,kernel_size=3, padding=0):
        super(Wconv1,self).__init__()
        self.inC=inC
        self.outC=outC
        self.kernel=kernel_size
        self.padding=padding
        self.F2W=Fea2Weight(Feature_num,inC,kernel_size,outC)
        self.conv2d=nn.functional.conv2d
        self.Norm = nn.LayerNorm([self.inC,self.kernel, self.kernel])
    def get_device(self):
        return torch.device("cuda" if next(self.parameters()).is_cuda else "cpu")
    # @profile
    def forward(self,input_):
        input, features=input_[0],input_[1]
        N=features.size(0)
        features = features.to(self.get_device())
        if N!=input.size(0):
            print()
            raise Exception("features.size(0)!=input.size(0) {} {}".format(features.size(),input.size()))
        weights=self.F2W(features)
        weights=weights.view(N,self.outC,self.inC,self.kernel,self.kernel)
        weights=self.Norm(weights)/self.inC
        return [torch.cat([self.conv2d(input[i:i+1],weights[i],padding=self.padding) for i in range(N)],dim=0),features]
class GLCMconv1(nn.Module):
    def __init__(self,inC,outC=3,kernel_size=3, padding=0):
        super(GLCMconv1,self).__init__()
        self.inC=inC
        self.outC=outC
        self.kernel=kernel_size
        self.padding=padding
        self.fetra=GLCM(1)
        self.conv=Wconv1(6,inC,outC,kernel_size,padding=padding)
    def get_device(self):
        return torch.device("cuda" if next(self.parameters()).is_cuda else "cpu")
    # @profile
    def forward(self,input):
        features = self.fetra(input)
        features = features.to(self.get_device())
        return self.conv(input,features)

class Wconv2(nn.Module):
    def __init__(self,Feature_num,inC,outC=3,kernel_size=3, padding=0):
        super(Wconv2,self).__init__()
        self.inC=inC
        self.outC=outC
        self.kernel=kernel_size
        self.padding=padding
        self.F2W=Fea2Weight(Feature_num,inC,kernel_size,outC)
        self.conv2d=nn.functional.conv2d
        self.Norm = nn.LayerNorm([self.inC,self.kernel, self.kernel])
    def get_device(self):
        return torch.device("cuda" if next(self.parameters()).is_cuda else "cpu")

    def forward(self,input,features):
        if 1!=features.size(0):
            raise Exception("1!=input.size(0)")
        features=features.to(self.get_device())
        if len(features.shape)==1:
            features=features.unsqueeze(0)
        weights=self.F2W(features)
        weights=weights.view(self.outC,self.inC,self.kernel,self.kernel)
        weights = self.Norm(weights)/self.inC
        # print(weights)
        return self.conv2d(input,weights,padding=self.padding)

if __name__=="__main__":
    img=torch.randn(10,3,100,100)
    up=UpSample(3,3)
    wcon1=Wconv1(1,3,64,3,1)
    wcon2=Wconv2(1,3,64,3,1)
    gconv1=GLCMconv1(3,3,3,1)
    features=torch.randn(10,1,dtype=torch.float32)
    t1 = time.time()
    img_=wcon1(img,features)

    print(wcon2(img,torch.Tensor([2.0])).shape)



    print(time.time() - t1)





