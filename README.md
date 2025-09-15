# Semantic Communication in PyTorch

This repository provides a modular PyTorch toolkit for building, training and evaluating semantic communication systems end-to-end.

## Installation

To get started, install this on PyPI:
``` bash
pip install semantics-pytorch
```

## Quickstart

```python
import torch
from semantics.pipeline import Pipeline
import semantics.vision as sv

# Define encoder/decoder
encoder = sv.WITTEncoder(img_size=32, patch_size=2, embed_dims=[32,64,128,256], depths=[2,2,2,2], num_heads=[4,8,8,16])
decoder = sv.WITTDecoder(img_size=32, patch_size=2, embed_dims=[256,128,64,32], depths=[2,2,2,2], num_heads=[16,8,8,4])

# Define channel
channel = sv.AWGNChannel(mean=0.0, std=0.1)

# Build pipeline
pipe = Pipeline(encoder, channel, decoder)

# Run a forward pass
x = torch.randn(8, 3, 32, 32)  # dummy batch
out, _ = pipe(x)
print(out.shape)
```

## Training Semantic Communication Models

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

## More Examples
For additional usage examples—including how to set up datasets, train pipelines, and run inference—see the `examples/` folder. These scripts provide workflows you can adapt to your own research or experiments.

### Roadmap

- [x] Ability to train semantic communication models
- [x] Add metrics to the package
- [x] Make into python package for easy usage
- [x] Implement more model architectures
- [x] Ability to train and run 'semantic' classifiers
- [ ] Train models and store their weights somewhere
- [ ] Have the ability to download pretrained models
