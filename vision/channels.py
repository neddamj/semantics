import torch
import torch.nn as nn
import math

class _Channel(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = mean
        self.std = std
        self.real = torch.normal(mean, std)
        self.imag = torch.normal(mean, std)
        self.noise = self.real + 1j * self.imag

    @staticmethod
    def _ensure_complex(x: torch.Tensor) -> torch.Tensor:
        if not torch.is_complex(x):
            raise ValueError("Channel expects a complex tensor (torch.cfloat/torch.cdouble).")
        return x

    @staticmethod
    def _normalize_to_power(x: torch.Tensor, target_power: float = 1.0):
        """
        Normalize complex signal x to have average power = target_power.
        Average power of a complex tensor is E[|x|^2].
        """
        # E[|x|^2] = mean(real^2 + imag^2)
        pwr = (x.real.pow(2) + x.imag.pow(2)).mean().clamp_min(1e-12)
        scale = math.sqrt(target_power) / torch.sqrt(pwr)
        return x * scale, pwr  # returns normalized x and original power

    def _make_complex_noise(self, shape, device, dtype, mean, std):
        real = torch.normal(mean=mean, std=std, size=shape, device=device, dtype=torch.float32)
        imag = torch.normal(mean=mean, std=std, size=shape, device=device, dtype=torch.float32)
        # Promote to the matching complex dtype
        ctype = torch.cfloat if dtype == torch.cfloat else torch.cdouble
        return (real + 1j * imag).to(dtype=ctype, device=device)

    def forward(self, x: torch.Tensor, channel_param: float | None = None, avg_power: float | None = None):
        """
        Args:
            x: complex tensor (cfloat/cdouble), arbitrary shape.
            channel_param: SNR in dB (Es/N0). If None, uses self.std as noise std.
            avg_power: if provided, use this as the *pre-known* average power of x and skip re-normalization.
        """
        x = self._ensure_complex(x)
        device, dtype = x.device, x.dtype

        if avg_power is not None:
            # Use caller-provided average power; normalize x to unit power using it.
            # For a complex signal, avg_power should equal E[|x|^2].
            x_tx = x / torch.sqrt(torch.as_tensor(avg_power, device=device, dtype=x.real.dtype).clamp_min(1e-12))
            used_power = float(avg_power)
        else:
            x_tx, used_power = self._normalize_to_power(x, target_power=1.0)

        # --- Noise std from SNR (if provided) ---
        # For complex AWGN, per real/imag component variance is sigma^2 = 1/(2 * SNR_linear)
        if channel_param is not None:
            snr_linear = 10.0 ** (float(channel_param) / 10.0)
            sigma = math.sqrt(1.0 / (2.0 * snr_linear))
        else:
            sigma = self.std  # fallback to ctor default

        self.noise = self._make_complex_noise(x_tx.shape, device, dtype, mean=0.0, std=sigma)

        # Send the signal through the channel
        y = self.send(x_tx)

        # Restore the signal to the measured original power.
        restore_scale = torch.sqrt(torch.as_tensor(used_power, device=device, dtype=x.real.dtype))
        return y * restore_scale

    def send(self, x):
        # This method should be overridden by subclasses
        raise NotImplementedError
    
class GaussianNoiseChannel(_Channel):
    def send(self, x):
        return x + self.noise 

class RayleighNoiseChannel(_Channel):
    def send(self, x):
        h = self._make_complex_noise(x.shape, x.device, x.dtype, mean=0.0, std=1.0 / math.sqrt(2.0))
        return h * x + self.noise
    
class ErrorFreeChannel(_Channel):
    def send(self, x):
        return x