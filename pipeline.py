import torch.nn as nn

class _Encoder(nn.Module):
    def forward(self, x):
        raise NotImplementedError

class _Decoder(nn.Module):
    def forward(self, z):
        raise NotImplementedError

class _Channel(nn.Module):
    def forward(self, z):
        raise NotImplementedError

class Pipeline(nn.Module):
    def __init__(self, encoder: _Encoder, channel: _Channel, decoder: _Decoder):
        super().__init__()
        self.encoder, self.channel, self.decoder = encoder, channel, decoder

    def forward(self, x):
        z = self.encoder(x)
        z_noisy = self.channel(z)
        return self.decoder(z_noisy)
