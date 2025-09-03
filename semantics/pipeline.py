import torch.nn as nn

class Pipeline(nn.Module):
    def __init__(self, encoder, channel, decoder):
        super().__init__()
        self.encoder, self.channel, self.decoder = encoder, channel, decoder

    def forward(self, x):
        z = self.encoder(x)
        z_noisy = self.channel(z)
        return self.decoder(z_noisy)
