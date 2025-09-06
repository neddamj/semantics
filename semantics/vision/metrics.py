import torch
import torch.nn as nn

from pytorch_msssim import SSIM, MS_SSIM    # https://github.com/VainF/pytorch-msssim

class PSNRMetric(nn.Module):
    def __init__(self, data_range=1.0, reduction='mean'):
        super().__init__()
        self.data_range = data_range
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mse = nn.functional.mse_loss(pred, target, reduction=self.reduction)
        psnr = 10.0 * torch.log10((self.data_range ** 2) / (mse + 1e-12))
        return psnr
    
class SSIMMetric(nn.Module):
    def __init__(self, data_range=1.0, size_average=True, channel=3):
        super().__init__()
        self.ssim = SSIM(data_range=data_range, size_average=size_average, channel=channel)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.ssim(pred, target)
    
class MSSSIMMetric(nn.Module):
    def __init__(self, data_range=1.0, size_average=True, channel=3):
        super().__init__()
        self.msssim = MS_SSIM(data_range=data_range, size_average=size_average, channel=channel)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.msssim(pred, target)