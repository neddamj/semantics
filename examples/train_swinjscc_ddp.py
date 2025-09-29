"""
Distributed example: train Swin-JSCC on two GPUs using torchrun.
Launch: torchrun --standalone --nproc_per_node=2 train_swinjscc_ddp.py --batch-size 128
"""
import argparse
import os
from typing import Tuple

import torch
import torch.distributed as dist
from torch.optim import Adam
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms

from semantics.pipeline import Pipeline
from semantics.train import Trainer, TrainerConfig
import semantics.vision as sv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Swin-JSCC with DDP")
    parser.add_argument("--data-root", default="./data", help="Dataset root directory")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Per-process batch size")
    parser.add_argument("--grad-accum", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=3e-4, help="Adam learning rate")
    parser.add_argument("--clip-grad", type=float, default=1.0, help="Gradient norm clip (0 disables)")
    parser.add_argument("--amp-dtype", default="auto", choices=["auto", "bf16", "fp16"], help="AMP precision")
    parser.add_argument("--no-amp", action="store_true", help="Disable automatic mixed precision")
    parser.add_argument("--ckpt", default="./checkpoints/swinjscc_ddp.pt", help="Checkpoint output path")
    parser.add_argument("--backend", default="nccl", help="torch.distributed backend")
    parser.add_argument("--sync-bn", action="store_true", help="Convert BatchNorm layers to SyncBatchNorm")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader worker processes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--find-unused", dest="find_unused", action="store_true",
                        help="Enable DDP unused parameter detection (default: on)")
    parser.add_argument("--no-find-unused", dest="find_unused", action="store_false",
                        help="Disable DDP unused parameter detection")
    parser.set_defaults(find_unused=True)
    return parser.parse_args()


def init_distributed(backend: str) -> Tuple[bool, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    is_distributed = world_size > 1
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if is_distributed and torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    if is_distributed and not dist.is_initialized():
        dist.init_process_group(backend=backend, init_method="env://")
    return is_distributed, rank, local_rank


def build_dataloaders(data_root: str, batch_size: int, num_workers: int, is_distributed: bool):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])

    def make_dataset(train: bool):
        if not is_distributed:
            return datasets.CIFAR10(data_root, train=train, download=True, transform=transform)

        rank = dist.get_rank()
        if rank == 0:
            ds = datasets.CIFAR10(data_root, train=train, download=True, transform=transform)
            dist.barrier()
            return ds

        dist.barrier()
        return datasets.CIFAR10(data_root, train=train, download=False, transform=transform)

    train_dataset = make_dataset(train=True)
    val_dataset = make_dataset(train=False)

    train_sampler = DistributedSampler(train_dataset, shuffle=True) if is_distributed else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if is_distributed else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader


def build_pipeline(device: torch.device):
    img_size = (32, 32)
    embed_dims = [64, 128]
    depths = [2, 2]
    num_heads = [4, 8]
    snr = 10.0
    rate = 16
    model_name = 'SwinJSCC_w/o_SAandRA'
    encoder = sv.SwinJSCCEncoder(
        img_size=img_size,
        patch_size=2,
        in_chans=3,
        embed_dims=embed_dims,
        depths=depths,
        num_heads=num_heads,
        C=32,
        window_size=4,
        model=model_name,
        snr=snr,
        rate=rate,
    )

    decoder = sv.SwinJSCCDecoder(
        img_size=img_size,
        embed_dims=list(reversed(embed_dims)),
        depths=depths,
        num_heads=list(reversed(num_heads)),
        C=32,
        window_size=4,
        model=model_name,
        snr=snr,
    )

    channel = sv.AWGNChannel(mean=0.0, std=0.1, snr=snr, avg_power=None)
    return Pipeline(encoder.to(device), channel.to(device), decoder.to(device))


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True

    is_distributed, rank, local_rank = init_distributed(args.backend)
    device = torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else torch.device("cpu")

    train_loader, val_loader = build_dataloaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        is_distributed=is_distributed,
    )

    pipeline = build_pipeline(device)
    optimizer = Adam(pipeline.parameters(), lr=args.lr)
    loss_fn = torch.nn.L1Loss()
    metrics = {
        "psnr": sv.PSNRMetric(),
        "ssim": sv.SSIMMetric(data_range=1.0, size_average=True, channel=3),
    }

    cfg = TrainerConfig(
        num_epochs=args.epochs,
        device=str(device),
        ckpt_path=args.ckpt,
        use_amp=not args.no_amp,
        amp_dtype=args.amp_dtype,
        grad_accum_steps=args.grad_accum,
        clip_grad_norm=args.clip_grad,
        distributed=is_distributed,
        dist_backend=args.backend,
        local_rank=local_rank,
        sync_batchnorm=args.sync_bn,
        ddp_find_unused_parameters=args.find_unused,
    )

    trainer = Trainer(
        pipeline=pipeline,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        config=cfg,
        metrics=metrics,
    )

    trainer.train()

    if is_distributed and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

    if (not is_distributed) or rank == 0:
        print("Training complete. Checkpoint saved to", args.ckpt)


if __name__ == "__main__":
    main()