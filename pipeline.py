from typing import Protocol
import torch.nn as nn

class _Encoder(Protocol):
    def forward(self, x): ...

class _Decoder(Protocol):
    def forward(self, z): ...

class _Channel(Protocol):
    def forward(self, z): ...

class Pipeline(nn.Module):
    def __init__(self, encoder: _Encoder, channel: _Channel, decoder: _Decoder):
        super().__init__()
        self.encoder, self.channel, self.decoder = encoder, channel, decoder

    def forward(self, x):
        z = self.encoder(x)
        z_noisy = self.channel(z)
        return self.decoder(z_noisy)
