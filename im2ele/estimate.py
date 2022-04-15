from torchvision.transforms import Compose
from torchvision.transforms import transforms

from im2ele.models.transforms import ToTensor, Normalize, CenterCrop 
from PIL import Image
import numpy as np

import torch
import cv2

def estimateim2ele(img, msize, im2elemodel, device):
    # Adapted from IM2ELEVATION source code: https://github.com/speed8928/IMELE
    # Crop to 440x440 to match cropped training 
    crop = CenterCrop([440, 440])
    img_pil = crop(Image.fromarray((img * 255).astype(np.uint8)))
    img_resize = img_pil.resize((msize, msize), resample=Image.BICUBIC, box=None, reducing_gap=None)

    im2ele_transform = transforms.Compose([
                                            ToTensor(),
                                            Normalize([0.485, 0.456, 0.406],
                                                        [0.229, 0.224, 0.225])
                                            ])
    img_torch = im2ele_transform(img_resize)
    
    with torch.no_grad():
        sample = torch.unsqueeze(img_torch, 0).to(device)  
        prediction = im2elemodel(sample)
    
    prediction = prediction.squeeze().cpu().numpy()
    prediction = cv2.resize(prediction, (440, 440), interpolation=cv2.INTER_CUBIC)

    return prediction