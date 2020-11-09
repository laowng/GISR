import torch
import kornia
import numpy as np
from skimage.feature import greycomatrix,greycoprops
class GLCM():
    def __init__(self,rgb_range,norm=True,**kwargs):
        self.rgb_range=rgb_range
        self.device=torch.device("cpu")
        self.togray = kornia.color.RgbToGrayscale()
        self.kwargs=kwargs
        if "distances" not in self.kwargs:
            self.kwargs["distances"]=[2]
        if "angles" not in self.kwargs:
            self.kwargs["angles"]=[np.pi/4]
        self.kwargs["normed"]=True
        self.features=["contrast","homogeneity","ASM","correlation","sobel"]
        self.norm=norm
        self.norm_data=np.array([1000,1,0.01,1,10],dtype=np.float)
        self.sobel_f=kornia.filters.Sobel()
        self.sobel_data=[]
    def quantize(self,img:torch.Tensor):
        pixel_range = 255 / self.rgb_range
        return img.mul(pixel_range).clamp(0, 255).round().to(dtype=torch.uint8)
    def sobel(self,imgs):
        sobel_b=self.sobel_f(imgs)
        self.sobel_data = sobel_b.sum(dim=tuple(range(1, len(sobel_b.shape))))
    def get_features(self,img):
        comatix=greycomatrix(img,**self.kwargs)
        features=[]
        for f in self.features:
            if f =="sobel":
                feature=next(self.iter_sobel_data)/(self.img_h*self.img_w)
            else:
                feature=greycoprops(comatix, f).squeeze()
            features.append(feature)
        features=np.array(features,dtype=np.float)
        if self.norm:
            features = (features / self.norm_data)#.clip(0.1, 1)
        return features
    def __call__(self, imgs:torch.Tensor):
        if imgs.is_cuda:
            imgs=imgs.to(self.device)
        self.channels,_,self.img_h,self.img_w=imgs.size()
        gray_imgs=self.togray(imgs)
        # if len(gray_imgs.shape)==2:
        #     gray_imgs=gray_imgs.unsqueeze()
        if "sobel" in self.features:
            self.sobel(gray_imgs)
            self.iter_sobel_data=iter(self.sobel_data)
        gray_imgs=self.quantize(gray_imgs).view(self.channels,self.img_h,self.img_w)
        features=[]
        for img in gray_imgs:
            features.append(self.get_features(img))
        return torch.tensor(features,dtype=torch.float32)

if __name__=="__main__":
    imgs=torch.randn(1,3,100,100)
    glcm=GLCM(rgb_range=1)
    print(glcm(imgs))

