# Semantic Communication in PyTorch

#### This repository provides tools that can be used to implement semantic communication workflows in PyTorch.

Example Usage
```python
from semantics.pipeline import Pipeline
import semantics.vision as sv

import torch

# Configuration parameters
batch_size = 4
dim = 64
img_size = 32
patch_size = 2
window_size = 4
num_heads = 4
modulation = True
num_channels = 3
channel_mean = 0.0
channel_std = 0.1
channel_snr = None
channel_avg_power = None

encoder_cfg = {
    'img_size': img_size, 
    'patch_size': patch_size, 
    'embed_dims': [64, 128],
    'depths': [2, 2],
    'num_heads': [4, 8], 
    'C_out': 32, 
    'window_size': 4, 
    'use_modulation': modulation,
    'in_chans': num_channels
}

decoder_cfg = {
    'img_size': img_size, 
    'patch_size': patch_size, 
    'embed_dims': [128, 64],
    'depths': [2, 2], 
    'num_heads': [8, 4], 
    'C_in': 32, 
    'window_size': 4, 
    'use_modulation': modulation,
    'out_chans': num_channels
}

channel_config = {
    'mean': channel_mean,
    'std': channel_std,
    'snr': channel_snr,
    'avg_power': channel_avg_power
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

encoder = sv.encoder.WITTEncoder(**encoder_cfg).to(device)
decoder = sv.decoder.WITTDecoder(**decoder_cfg).to(device)
channel = sv.channels.ErrorFreeChannel(**channel_config).to(device)
pipeline = Pipeline(encoder, channel, decoder).to(device)

# Semantic Communication Example
input_image = torch.randn(batch_size, num_channels, img_size, img_size).to(device)
with torch.no_grad():
    # Run image through the entire pipeline step-by-step
    encoded_img = encoder(input_image)
    channel_out = channel(encoded_img)
    output_image = decoder(channel_out)

    # Run image through the entire pipeline at once
    pipeline_out = pipeline(input_image)

print("Input image shape:", input_image.shape)
print("Encoded image shape:", encoded_img.shape)
print("Channel output shape:", channel_out.shape)
print("Output image shape:", output_image.shape)
print("Pipeline output shape:", pipeline_out.shape)

# The output of the individual components is the same as the output of the pipeline
torch.all(output_image == pipeline_out)  # Should be True
```

### Roadmap:
- [x] Ability to train semantic communication models
- [ ] Train models and store their weights somewhere
- [ ] Have the ability to download pretrained models
- [ ] Make into python package for easy usage
- [ ] Implement more model architectures