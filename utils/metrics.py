import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from typing import Tuple
import torch


class SSIM:
    # TODO: not sure if works correctly - verify/find better
    def __init__(self, data_range: float = 1, win_size: int = 7) -> None:
        self.data_range = data_range
        self.win_size = win_size  # Allows customization of window size

    def __call__(self, output: np.ndarray, target: np.ndarray) -> float:
        return ssim(output, target, data_range=self.data_range, multichannel=True, channel_axis=-1, win_size=self.win_size)


def calculate_metrics(output: np.ndarray, target: np.ndarray) -> Tuple[float, float]:
    if isinstance(output, torch.Tensor):
        output = output.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()

    output = output.transpose(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
    target = target.transpose(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)

    data_range = np.max(target) - np.min(target)
    ssim_val = SSIM(data_range)(output[0], target[0])
    psnr_val = psnr(target[0], output[0], data_range=data_range)
    return psnr_val, ssim_val


