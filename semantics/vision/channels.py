import torch
import torch.nn as nn
import math

class _Channel(nn.Module):
    def __init__(self, mean, std, snr=None, avg_power=None):
        super().__init__()
        self.mean = mean
        self.std = std
        self.snr = snr
        self.avg_power = avg_power

    @staticmethod
    def _normalize_to_power(x: torch.Tensor, target_power: float = 1.0):
        pwr = (x.real.pow(2) + x.imag.pow(2)).mean().clamp_min(1e-12)
        scale = math.sqrt(target_power) / torch.sqrt(pwr)
        return x * scale, pwr

    @staticmethod
    def _make_complex_noise(shape, device, dtype, mean, std):
        real = torch.normal(mean=mean, std=std, size=shape, device=device, dtype=torch.float32)
        imag = torch.normal(mean=mean, std=std, size=shape, device=device, dtype=torch.float32)
        ctype = torch.cfloat if dtype == torch.cfloat else torch.cdouble
        return (real + 1j * imag).to(dtype=ctype, device=device)

    @staticmethod
    def _pack_real_to_complex(x_real: torch.Tensor) -> torch.Tensor:
        if torch.is_complex(x_real):
            return x_real  # already complex
        if x_real.shape[-1] % 2 != 0:
            raise ValueError("Last dimension must be even to pack real -> complex (I/Q pairing).")
        x2 = x_real.view(*x_real.shape[:-1], -1, 2)
        return x2[..., 0].to(torch.cfloat) + 1j * x2[..., 1].to(torch.cfloat)

    @staticmethod
    def _unpack_complex_to_real(x_c: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        x2 = torch.stack([x_c.real.to(dtype), x_c.imag.to(dtype)], dim=-1)  # (..., M, 2)
        return x2.reshape(*x2.shape[:-2], -1)  # (..., 2M)

    def forward(self, x: torch.Tensor):
        input_was_complex = torch.is_complex(x)
        device = x.device

        # If real, pack into complex on the last dimension
        if input_was_complex:
            x_c = x
        else:
            x_c = self._pack_real_to_complex(x)

        # Normalize (or use provided power)
        if self.avg_power is not None:
            x_tx = x_c / torch.sqrt(torch.as_tensor(self.avg_power, device=device, dtype=x_c.real.dtype).clamp_min(1e-12))
            used_power = float(self.avg_power)
        else:
            x_tx, used_power = self._normalize_to_power(x_c, target_power=1.0)

        # Noise std from SNR if provided
        if self.snr is not None:
            snr_linear = 10.0 ** (float(self.snr) / 10.0)
            sigma = math.sqrt(1.0 / (2.0 * snr_linear))
        else:
            sigma = self.std

        self.noise = self._make_complex_noise(x_tx.shape, x_tx.device, x_tx.dtype, mean=0.0, std=sigma)

        # Pass through channel and restore original power
        y_c = self.send(x_tx)
        y_c = y_c * torch.sqrt(torch.as_tensor(used_power, device=device, dtype=y_c.real.dtype))

        # Return in the same "type/shape family" as the input
        if input_was_complex:
            return y_c
        else:
            # Unpack back to the original real shape
            y_real = self._unpack_complex_to_real(y_c, dtype=x.dtype if not torch.is_complex(x) else torch.float32)
            # Ensure shape exactly matches input
            if y_real.shape != x.shape:
                raise RuntimeError(f"Shape mismatch after channel: got {y_real.shape}, expected {x.shape}")
            return y_real

    def send(self, x):
        raise NotImplementedError
    
class AWGNChannel(_Channel):
    def send(self, x):
        return x + self.noise 

class RayleighNoiseChannel(_Channel):
    def send(self, x):
        h = self._make_complex_noise(x.shape, x.device, x.dtype, mean=0.0, std=1.0)# / math.sqrt(2.0))
        h = h.real ** 2 + h.imag ** 2
        h = torch.sqrt(h).clamp_min(1e-12) / math.sqrt(2.0)
        return h * x + self.noise
    
class ErrorFreeChannel(_Channel):
    def forward(self, x):
        return x