# Semantic Communication in PyTorch

This repository provides utilities that can be used to implement semantic communication workflows in PyTorch.

## Getting Started

``` bash
pip install semantics-pytorch
```

### Example Usage

```python
from semantics.pipeline import Pipeline
import semantics.vision as sv

import torch

# Configuration parameters
batch_size = 128
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

encoder = sv.WITTEncoder(
    img_size = img_size, 
    patch_size = patch_size, 
    embed_dims = [32, 64, 128, 256],
    depths = [2, 2, 2, 2],
    num_heads = [4, 8, 8, 8], 
    C_out = 32, 
    window_size = 4, 
    use_modulation = modulation,
    in_chans = num_channels
).to(device)

decoder = sv.WITTDecoder(
    img_size = img_size, 
    patch_size = patch_size, 
    embed_dims = [256, 128, 64, 32],
    depths = [2, 2, 2, 2], 
    num_heads = [8, 8, 8, 4], 
    C_in = 32, 
    window_size = 4, 
    use_modulation = modulation,
    out_chans = num_channels
).to(device)

channel = sv.AWGNChannel(
    mean = channel_mean,
    std = channel_std,
    snr = channel_snr,
    avg_power = channel_avg_power
).to(device)

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

### Training Semantic Communication Models

Training models can be accomplished easily via the Trainer workflow. An example of training on the CIFAR-10 dataset can be seen below

```python
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from semantics.pipeline import Pipeline
from semantics.train import Trainer, TrainerConfig
import semantics.vision as sv

# Configuration parameters
batch_size = 128
dim = 128
img_size = 32
modulation = True
num_channels = 3
channel_mean = 0.0
channel_std = 0.1
channel_snr = None
channel_avg_power = None

encoder_cfg = {
    'in_ch': num_channels,
    'k': dim,
    'reparameterize': False
}

decoder_cfg = {
    'out_ch': num_channels,
    'k': dim,
    'reparameterize': True
}

channel_config = {
    'mean': channel_mean,
    'std': channel_std,
    'snr': channel_snr,
    'avg_power': channel_avg_power
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

encoder = sv.VSCCEncoder(**encoder_cfg).to(device)
decoder = sv.VSCCDecoder(**decoder_cfg).to(device)
channel = sv.AWGNChannel(**channel_config).to(device)
pipeline = Pipeline(encoder, channel, decoder).to(device)

# Data
transform = transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor()])
train_ds = datasets.CIFAR10("./data", train=True,  download=True, transform=transform)
val_ds   = datasets.CIFAR10("./data", train=False, download=True, transform=transform)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

# Optimizer and Loss
optimizer = Adam(pipeline.parameters(), lr=3e-4)
criterion = torch.nn.L1Loss()

# Simple metrics
metrics = {
        "psnr": sv.PSNRMetric(),
        'ssim': sv.SSIMMetric(data_range=1.0, size_average=True, channel=3)
    }

# Train
cfg = TrainerConfig(
    num_epochs=20,
    use_amp=True,          # turn on mixed precision
    amp_dtype="auto",      # auto-select bf16/fp16
    grad_accum_steps=1,    # increase if batches are small
    clip_grad_norm=1.0,    # optional safety
    compile_model=False,   # set True if PyTorch 2.x and stable graph
)
trainer = Trainer(
    pipeline=pipeline,
    optimizer=optimizer,
    train_loader=train_loader,
    val_loader=val_loader,
    loss_fn=criterion,
    config=cfg,
    metrics=metrics,
)
trainer.train()
```

More examples for inference and training of semantic communication models can be found in the example folder

### Roadmap

- [x] Ability to train semantic communication models
- [x] Add metrics to the package
- [x] Make into python package for easy usage
- [x] Implement more model architectures
- [ ] Train models and store their weights somewhere
- [ ] Have the ability to download pretrained models
