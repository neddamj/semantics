import torch.nn as nn

class Pipeline(nn.Module):
    def __init__(self, encoder, channel, decoder):
        super().__init__()
        self.encoder, self.channel, self.decoder = encoder, channel, decoder

    def forward(self, x):
        # Encode the input message
        encoder_output = self.encoder(x)
        if isinstance(encoder_output, tuple):
            z = encoder_output[0]
            aux_encoder = encoder_output[1:]
        else:
            z = encoder_output
            aux_encoder = ()

        # Pass data through the channel
        z_noisy = self.channel(z)

        # Decode the received message
        decoder_output = self.decoder(z_noisy)
        if isinstance(decoder_output, tuple):
            recon = decoder_output[0]
            aux_decoder = decoder_output[1:]
        else:
            recon = decoder_output
            aux_decoder = ()

        aux = {
            "encoder": aux_encoder,
            "decoder": aux_decoder
        }
        
        return recon, aux
