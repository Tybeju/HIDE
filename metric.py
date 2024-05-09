import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

class SSIM:
    def __init__(self, data_range):
        self.data_range = data_range

    def __call__(self, output, target):
        return ssim(output, target, data_range=self.data_range, multichannel=True)

def calculate_metrics(output, target):
    data_range = target.max() - target.min()
    ssim_val = SSIM(data_range)(output, target)
    psnr_val = psnr(target, output, data_range=data_range)
    return psnr_val, ssim_val
