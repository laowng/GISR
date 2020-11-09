import model.Meta_ as Meta
import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt
import numpy as np
class cal_time():
    def __init__(self):
        self.time_start=time.time()
    def start(self):
        self.time_start=time.time()
    def end(self):
        return round(time.time()-self.time_start,3)
ctime=cal_time()
# torch.set_grad_enabled(False)
def plot_psnr(y1,y2,y3,path_name):
    axis = np.linspace(1, len(y1), len(y1))
    xlabel=[]
    for i in range(1,len(y1)+1):
        xlabel.append(2**i)
    xlabel=np.array(xlabel)
    fig = plt.figure()
    plt.plot(
        axis,
        np.array(y1),
        label="GFeaConv",
        color="red",
    )
    plt.plot(
        axis,
        np.array(y2),
        label="Conv",
        color = "blue"
    )
    plt.plot(
        axis,
        np.array(y3),
        label="Conv-Conv",
        color = "yellow"
    )
    plt.xticks(axis,xlabel)
    plt.legend()
    plt.xlabel('channels')
    plt.ylabel('time cost(s)')
    plt.grid(True)
    plt.savefig("./{}.pdf".format(path_name))
    plt.close(fig)
if __name__=="__main__":
    gfea=torch.randn(10,5).cuda()
    time_conv=[]
    time_geaconv=[]
    time_convx2=[]
    for i in range(1,9):
        print("\r"+str(i),end="")
        gfeaconv = Meta.Wconv1(5, 2**i, 2**i, 3, padding=1).cuda()
        conv = nn.Conv2d(2**i, 2**i, 3, padding=1).cuda()
        imgs = torch.randn(10, 2**i, 100, 100).cuda()
        conv_time=0
        convx2_time=0
        gfeaconv_time=0
        for i in range(1000):
            ctime.start()
            conv(imgs)
            conv_time+=ctime.end()
            ctime.start()
            conv(conv(imgs))
            convx2_time += ctime.end()
            ctime.start()
            gfeaconv(imgs, gfea)
            gfeaconv_time+=ctime.end()
        gfeaconv_time=round(gfeaconv_time/1000,5)
        conv_time=round(conv_time/1000,5)
        convx2_time=round(convx2_time/1000,5)
        time_convx2.append(convx2_time)
        time_geaconv.append(gfeaconv_time)
        time_conv.append(conv_time)
    print(time_geaconv)
    print(time_conv)
    plot_psnr(time_geaconv,time_conv,time_convx2,"gpu_time")

    gfea = torch.randn(10, 5)
    time_conv = []
    time_geaconv = []
    time_convx2 = []
    for i in range(1, 9):
        print("\r" + str(i), end="")
        gfeaconv = Meta.Wconv1(5, 2 ** i, 2 ** i, 3, padding=1)
        conv = nn.Conv2d(2 ** i, 2 ** i, 3, padding=1)
        imgs = torch.randn(10, 2 ** i, 100, 100)
        conv_time = 0
        convx2_time = 0
        gfeaconv_time = 0
        for i in range(1000):
            ctime.start()
            conv(imgs)
            conv_time += ctime.end()
            ctime.start()
            conv(conv(imgs))
            convx2_time += ctime.end()
            ctime.start()
            gfeaconv(imgs, gfea)
            gfeaconv_time += ctime.end()
        gfeaconv_time = round(gfeaconv_time / 1000, 5)
        conv_time = round(conv_time / 1000, 5)
        convx2_time = round(convx2_time / 1000, 5)
        time_convx2.append(convx2_time)
        time_geaconv.append(gfeaconv_time)
        time_conv.append(conv_time)
    print(time_geaconv)
    print(time_conv)
    plot_psnr(time_geaconv, time_conv,time_convx2, "cpu_time")

