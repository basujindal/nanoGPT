import open_clip
import torch
import matplotlib.pyplot as plt
import os
import json
from PIL import Image
from tqdm import trange
import numpy as np
from utils import time_gpu
import pickle
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

model, _, preprocess = open_clip.create_model_and_transforms('ViT-bigG-14', pretrained='laion2b_s39b_b160k')
print(sum(p.numel() for p in model.parameters() if p.requires_grad)/1e9)

model.to(torch.bfloat16)
device  = torch.device('cuda:1')
model = model.to(device)

class Normalize:
    """Applies the :class:`~torchvision.transforms.Normalize` transform to a batch of images.

    .. note::
        This transform acts out of place by default, i.e., it does not mutate the input tensor.

    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation in-place.
        dtype (torch.dtype,optional): The data type of tensors to which the transform will be applied.
        device (torch.device,optional): The device of tensors to which the transform will be applied.

    """

    def __init__(self, mean, std, inplace=False, dtype=torch.float, device='cpu'):
        self.mean = torch.as_tensor(mean, dtype=dtype, device=device)[None, :, None, None]
        self.std = torch.as_tensor(std, dtype=dtype, device=device)[None, :, None, None]
        self.inplace = inplace
        
    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor of size (N, C, H, W) to be normalized.

        Returns:
            Tensor: Normalized Tensor.
        """
        if not self.inplace:
            tensor = tensor.clone()

        tensor.sub_(self.mean).div_(self.std)
        return tensor

trans = Compose([
    Resize(size=224),
    CenterCrop(size=(224, 224)),
    ToTensor(),
    # Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
])

pth = '../cptData/llava/train2014'
dir_imgs = os.listdir(pth)

trans2 = Compose([
    Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711),device=device),
])

def get_imgs(i, batch_size=10):
    img = dir_imgs[i]
    image = trans(Image.open(os.path.join(pth, img))).to(device).unsqueeze(0)
    
    if image.size()[1] != 3:
        image = image.repeat(1, 3, 1, 1)
    
    for j in range(batch_size-1):
        if i+j+1 >= len(dir_imgs):
            break
        img = preprocess(Image.open(os.path.join(pth, dir_imgs[i+j+1]))).unsqueeze(0).to(device)
        if img.size()[1] != 3:
            img = image.repeat(1, 3, 1, 1)
        image = torch.cat((image, img), 0)
        
    image = trans2(image)
    return image

all_features = torch.zeros((len(dir_imgs), 1280), dtype=torch.bfloat16).to(device)

batch_size = 512

for i in trange(len(dir_imgs)//batch_size + 1):
    images = get_imgs(i*batch_size, batch_size)
    with torch.no_grad():
        image_features = model.encode_image(images.to(device).to(torch.bfloat16))
        all_features[i*batch_size:(i+1)*batch_size] = image_features
            
torch.save(all_features, '../cptData/llava/all_features.pt')

