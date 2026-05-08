#!/usr/bin/env python3

from __future__ import annotations

import csv
import hashlib
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import SSIM
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image

from semantics.classifiers import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from semantics.pipeline import Pipeline
from semantics.vision.channels import AWGNChannel, ErrorFreeChannel, RayleighNoiseChannel
from semantics.vision.models.swinjscc import SwinJSCCDecoder, SwinJSCCEncoder
from semantics.vision.models.vscc import VSCCDecoder, VSCCEncoder
from semantics.vision.models.witt import WITTDecoder, WITTEncoder


DEFAULT_DATA_ROOT = "data"
DEFAULT_CHECKPOINT_DIR = "checkpoints"
DEFAULT_OUTPUT_ROOT = "outputs/robustness"
DEFAULT_CLASSIFIER_CHECKPOINT = "checkpoints/cifar10_resnet18_classifier.pt"
DEFAULT_TRAIN_SNR_DB = 10.0
DEFAULT_EVAL_SNR_DB = [20.0, 10.0, 5.0, 0.0]
DEFAULT_SAMPLE_COUNT = 8
MODEL_CHOICES = ("swinjscc", "witt", "vscc")
TRAIN_CHANNEL_CHOICES = ("awgn", "rayleigh")
EVAL_CHANNEL_CHOICES = ("error_free", "awgn", "rayleigh")
ATTACK_LOCATIONS = ("clean", "input", "latent")
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


@dataclass(frozen=True)
class ChannelCondition:
    name: str
    kind: str
    snr_db: Optional[float]
    slug: str


@dataclass(frozen=True)
class TrainingDefaults:
    epochs: int
    batch_size: int
    lr: float
    optimizer_name: str
    use_amp: bool
    compile_model: bool = False


SEMANTIC_DEFAULTS = {
    "swinjscc": TrainingDefaults(
        epochs=10,
        batch_size=128,
        lr=3e-4,
        optimizer_name="adam",
        use_amp=False,
    ),
    "witt": TrainingDefaults(
        epochs=50,
        batch_size=128,
        lr=3e-4,
        optimizer_name="adamw",
        use_amp=True,
    ),
    "vscc": TrainingDefaults(
        epochs=20,
        batch_size=128,
        lr=3e-4,
        optimizer_name="adam",
        use_amp=True,
    ),
}


CLASSIFIER_DEFAULTS = TrainingDefaults(
    epochs=5,
    batch_size=128,
    lr=3e-4,
    optimizer_name="adamw",
    use_amp=True,
)


CLASSIFIER_ARCHES = {
    "resnet18": ResNet18,
    "resnet34": ResNet34,
    "resnet50": ResNet50,
    "resnet101": ResNet101,
    "resnet152": ResNet152,
}


def resolve_path(raw_path: str | Path) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def resolve_device(raw_device: Optional[str]) -> torch.device:
    if raw_device:
        return torch.device(raw_device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def stable_int_seed(*parts: object, modulo: int = 2**31 - 1) -> int:
    payload = "::".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    return int(digest[:15], 16) % modulo


def format_snr(snr_db: float) -> str:
    return f"{snr_db:g}"


def snr_slug(snr_db: float) -> str:
    return format_snr(snr_db).replace("-", "neg").replace(".", "p")


def semantic_checkpoint_name(
    model_name: str,
    train_channel: str,
    train_snr_db: float,
    swin_bottleneck: int = 16,
) -> str:
    base = f"{model_name}_dataset_cifar10_train_{train_channel}_snr_{snr_slug(train_snr_db)}db"
    if model_name == "swinjscc":
        base = f"{base}_bottleneck_{swin_bottleneck}"
    return f"{base}.pt"


def default_semantic_checkpoint_path(
    model_name: str,
    train_channel: str,
    train_snr_db: float,
    checkpoint_dir: str | Path = DEFAULT_CHECKPOINT_DIR,
    swin_bottleneck: int = 16,
) -> Path:
    directory = resolve_path(checkpoint_dir)
    return directory / semantic_checkpoint_name(
        model_name=model_name,
        train_channel=train_channel,
        train_snr_db=train_snr_db,
        swin_bottleneck=swin_bottleneck,
    )


def resolve_semantic_checkpoint_path(
    model_name: str,
    train_channel: str,
    train_snr_db: float,
    checkpoint_dir: str | Path = DEFAULT_CHECKPOINT_DIR,
    swin_bottleneck: int = 16,
) -> Path:
    default_path = default_semantic_checkpoint_path(
        model_name=model_name,
        train_channel=train_channel,
        train_snr_db=train_snr_db,
        checkpoint_dir=checkpoint_dir,
        swin_bottleneck=swin_bottleneck,
    )
    if default_path.is_file():
        return default_path

    legacy_candidates = []
    if (
        model_name == "swinjscc"
        and train_channel == "awgn"
        and abs(train_snr_db - 10.0) < 1e-9
    ):
        legacy_candidates.append(
            resolve_path(f"checkpoints/swinjscc_dataset_cifar10_bottleneck_{swin_bottleneck}")
        )

    for candidate in legacy_candidates:
        if candidate.is_file():
            return candidate
    return default_path


def default_classifier_checkpoint_path(
    checkpoint: str | Path = DEFAULT_CLASSIFIER_CHECKPOINT,
) -> Path:
    return resolve_path(checkpoint)


def build_channel_conditions(snr_values: Sequence[float]) -> list[ChannelCondition]:
    conditions = [
        ChannelCondition(name="ErrorFree", kind="error_free", snr_db=None, slug="error_free"),
    ]
    for snr_db in snr_values:
        snr_text = format_snr(snr_db)
        conditions.append(
            ChannelCondition(
                name=f"AWGN @ {snr_text} dB",
                kind="awgn",
                snr_db=snr_db,
                slug=f"awgn_snr_{snr_slug(snr_db)}db",
            )
        )
    for snr_db in snr_values:
        snr_text = format_snr(snr_db)
        conditions.append(
            ChannelCondition(
                name=f"Rayleigh @ {snr_text} dB",
                kind="rayleigh",
                snr_db=snr_db,
                slug=f"rayleigh_snr_{snr_slug(snr_db)}db",
            )
        )
    return conditions


def instantiate_channel(
    kind: str,
    snr_db: Optional[float],
    device: torch.device,
):
    if kind == "error_free":
        channel = ErrorFreeChannel(mean=0.0, std=0.0, snr=None, avg_power=None)
    elif kind == "awgn":
        channel = AWGNChannel(mean=0.0, std=0.0, snr=snr_db, avg_power=None)
    elif kind == "rayleigh":
        channel = RayleighNoiseChannel(mean=0.0, std=0.0, snr=snr_db, avg_power=None)
    else:
        raise ValueError(f"Unknown channel kind: {kind}")
    return channel.to(device)


def split_primary_output(output):
    if isinstance(output, tuple):
        return output[0], output[1:]
    return output, ()


def forward_semantic_pipeline(
    pipeline: Pipeline,
    x: Optional[torch.Tensor] = None,
    latent: Optional[torch.Tensor] = None,
):
    if x is None and latent is None:
        raise ValueError("Either x or latent must be provided.")

    if latent is None:
        latent, aux_encoder = split_primary_output(pipeline.encoder(x))
    else:
        aux_encoder = ()

    latent_noisy = pipeline.channel(latent)
    recon, aux_decoder = split_primary_output(pipeline.decoder(latent_noisy))
    aux = {
        "encoder": aux_encoder,
        "decoder": aux_decoder,
    }
    return recon, aux, latent, latent_noisy


def semantic_training_defaults(model_name: str) -> TrainingDefaults:
    try:
        return SEMANTIC_DEFAULTS[model_name]
    except KeyError as exc:
        raise ValueError(f"Unknown semantic model: {model_name}") from exc


def build_semantic_pipeline(
    model_name: str,
    device: torch.device,
    train_channel: str,
    train_snr_db: float,
    swin_bottleneck: int = 16,
) -> Pipeline:
    if model_name == "swinjscc":
        encoder = SwinJSCCEncoder(
            img_size=(32, 32),
            patch_size=2,
            in_chans=3,
            embed_dims=[64, 128],
            depths=[2, 2],
            num_heads=[4, 8],
            C=swin_bottleneck,
            window_size=4,
            model="SwinJSCC_w/o_SAandRA",
            snr=train_snr_db,
            rate=16,
        )
        decoder = SwinJSCCDecoder(
            img_size=(32, 32),
            embed_dims=[128, 64],
            depths=[2, 2],
            num_heads=[8, 4],
            C=swin_bottleneck,
            window_size=4,
            model="SwinJSCC_w/o_SAandRA",
            snr=train_snr_db,
        )
    elif model_name == "witt":
        encoder = WITTEncoder(
            img_size=32,
            patch_size=2,
            embed_dims=[32, 64, 128, 256],
            depths=[2, 2, 2, 2],
            num_heads=[4, 8, 8, 8],
            C_out=32,
            window_size=4,
            use_modulation=True,
            snr=train_snr_db,
            in_chans=3,
        )
        decoder = WITTDecoder(
            img_size=32,
            patch_size=2,
            embed_dims=[256, 128, 64, 32],
            depths=[2, 2, 2, 2],
            num_heads=[8, 8, 8, 4],
            C_in=32,
            window_size=4,
            use_modulation=True,
            snr=train_snr_db,
            out_chans=3,
        )
    elif model_name == "vscc":
        encoder = VSCCEncoder(
            in_ch=3,
            k=128,
            reparameterize=False,
        )
        decoder = VSCCDecoder(
            out_ch=3,
            k=128,
            reparameterize=True,
        )
    else:
        raise ValueError(f"Unknown semantic model: {model_name}")

    channel = instantiate_channel(train_channel, train_snr_db, device)
    return Pipeline(encoder.to(device), channel.to(device), decoder.to(device)).to(device)


def build_optimizer(
    parameters,
    optimizer_name: str,
    lr: float,
):
    name = optimizer_name.lower()
    if name == "adam":
        return torch.optim.Adam(parameters, lr=lr)
    if name == "adamw":
        return torch.optim.AdamW(parameters, lr=lr)
    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def build_classifier(
    arch: str,
    device: torch.device,
    pretrained: bool = False,
) -> nn.Module:
    try:
        classifier_cls = CLASSIFIER_ARCHES[arch]
    except KeyError as exc:
        raise ValueError(f"Unknown classifier arch: {arch}") from exc

    model = classifier_cls(
        num_classes=10,
        in_channels=3,
        pretrained=pretrained,
        freeze_backbone=False,
        dropout=0.2,
    )
    return model.to(device)


def classifier_train_transforms() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
        ]
    )


def classifier_eval_transforms() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
        ]
    )


def semantic_training_transforms() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ]
    )


def _build_cifar10_dataset(
    data_root: Path,
    train: bool,
    transform,
    download: bool,
):
    try:
        return datasets.CIFAR10(
            str(data_root),
            train=train,
            download=download,
            transform=transform,
        )
    except RuntimeError as exc:
        split = "training" if train else "validation"
        raise SystemExit(
            f"CIFAR-10 {split} data not found under {data_root}. "
            "Run with --download in an environment that can fetch the dataset."
        ) from exc


def subset_dataset(dataset, num_images: Optional[int], full_val: bool):
    if full_val or num_images is None:
        return dataset
    if num_images <= 0:
        raise SystemExit("--num-images must be positive unless --full-val is set.")
    limit = min(num_images, len(dataset))
    return Subset(dataset, range(limit))


def make_dataloader(
    dataset,
    batch_size: int,
    device: torch.device,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    if batch_size <= 0:
        raise SystemExit("--batch-size must be positive.")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )


def build_classifier_dataloaders(
    data_root: str | Path,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    download: bool,
):
    root = resolve_path(data_root)
    train_dataset = _build_cifar10_dataset(
        root,
        train=True,
        transform=classifier_train_transforms(),
        download=download,
    )
    val_dataset = _build_cifar10_dataset(
        root,
        train=False,
        transform=classifier_eval_transforms(),
        download=download,
    )
    train_loader = make_dataloader(
        train_dataset,
        batch_size=batch_size,
        device=device,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = make_dataloader(
        val_dataset,
        batch_size=max(batch_size, 256),
        device=device,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_dataset, val_dataset, train_loader, val_loader


def build_semantic_dataloaders(
    data_root: str | Path,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    download: bool,
):
    root = resolve_path(data_root)
    transform = semantic_training_transforms()
    train_dataset = _build_cifar10_dataset(root, train=True, transform=transform, download=download)
    val_dataset = _build_cifar10_dataset(root, train=False, transform=transform, download=download)
    train_loader = make_dataloader(
        train_dataset,
        batch_size=batch_size,
        device=device,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = make_dataloader(
        val_dataset,
        batch_size=batch_size,
        device=device,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_dataset, val_dataset, train_loader, val_loader


def build_eval_dataset(
    data_root: str | Path,
    num_images: Optional[int],
    full_val: bool,
    download: bool,
):
    root = resolve_path(data_root)
    transform = semantic_training_transforms()
    dataset = _build_cifar10_dataset(root, train=False, transform=transform, download=download)
    return subset_dataset(dataset, num_images=num_images, full_val=full_val)


def collect_sample_images(dataset, sample_count: int) -> Optional[torch.Tensor]:
    if sample_count < 0:
        raise SystemExit("--sample-count cannot be negative.")
    if sample_count == 0:
        return None
    count = min(sample_count, len(dataset))
    images = [dataset[index][0] for index in range(count)]
    if not images:
        return None
    return torch.stack(images, dim=0)


class CIFAR10Normalizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("mean", torch.tensor(CIFAR10_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(CIFAR10_STD).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


class ReconstructionMetricSuite:
    def __init__(self, device: torch.device):
        self.ssim = SSIM(data_range=1.0, size_average=False, channel=3).to(device)

    def measure(self, recon: torch.Tensor, target: torch.Tensor):
        mse_per_image = F.mse_loss(recon, target, reduction="none").flatten(1).mean(dim=1)
        psnr_per_image = 10.0 * torch.log10(1.0 / mse_per_image.clamp_min(1e-12))
        ssim_values = self.ssim(recon, target)
        if not torch.is_tensor(ssim_values):
            ssim_values = torch.as_tensor(ssim_values, device=recon.device)
        if ssim_values.ndim == 0:
            ssim_values = ssim_values.unsqueeze(0)
        return mse_per_image, psnr_per_image, ssim_values


def load_checkpoint_state(
    module: nn.Module,
    checkpoint_path: str | Path,
    device: torch.device,
    state_key: str = "pipeline",
) -> dict:
    path = resolve_path(checkpoint_path)
    if not path.is_file():
        raise SystemExit(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=device)
    state = checkpoint
    if isinstance(checkpoint, dict) and state_key in checkpoint:
        state = checkpoint[state_key]
    if not isinstance(state, dict):
        raise SystemExit(f"Checkpoint {path} does not contain a valid state dict under '{state_key}'.")

    try:
        module.load_state_dict(state)
    except RuntimeError as exc:
        raise SystemExit(f"Failed to load checkpoint {path}.\n{exc}") from exc
    return checkpoint if isinstance(checkpoint, dict) else {}


def read_checkpoint_metadata(checkpoint_path: str | Path, device: torch.device) -> dict:
    path = resolve_path(checkpoint_path)
    checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, dict):
        return {}
    metadata = checkpoint.get("metadata")
    return metadata if isinstance(metadata, dict) else {}


def attach_checkpoint_metadata(checkpoint_path: str | Path, metadata: dict) -> None:
    path = resolve_path(checkpoint_path)
    checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        return
    existing = checkpoint.get("metadata")
    merged = {}
    if isinstance(existing, dict):
        merged.update(existing)
    merged.update(metadata)
    checkpoint["metadata"] = merged
    torch.save(checkpoint, path)


def save_sample_grid(
    originals: Optional[torch.Tensor],
    reconstructions: Optional[torch.Tensor],
    output_path: Path,
) -> None:
    if originals is None or reconstructions is None:
        return
    ensure_dir(output_path.parent)
    grid = make_grid(
        torch.cat([originals, reconstructions], dim=0),
        nrow=originals.size(0),
        padding=2,
    )
    save_image(grid, output_path)


def write_csv(rows: Sequence[dict], csv_path: Path) -> None:
    ensure_dir(csv_path.parent)
    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_results_table(rows: Sequence[dict]) -> None:
    if not rows:
        print("No results collected.")
        return

    headers = ["Model", "Train", "Eval", "Attack", "Images", "ClsAcc", "AtkSucc", "PSNR", "SSIM"]
    body = []
    for row in rows:
        body.append(
            [
                str(row.get("model", "")),
                str(row.get("train_channel", "")),
                str(row.get("condition", "")),
                str(row.get("attack_location", "")),
                str(row.get("num_images", "")),
                _format_metric(row.get("classifier_acc")),
                _format_metric(row.get("attack_success_rate")),
                _format_metric(row.get("psnr")),
                _format_metric(row.get("ssim")),
            ]
        )

    widths = []
    for index, header in enumerate(headers):
        column_width = max(len(header), max(len(line[index]) for line in body))
        widths.append(column_width)

    divider = "-+-".join("-" * width for width in widths)
    print(" | ".join(header.ljust(widths[idx]) for idx, header in enumerate(headers)))
    print(divider)
    for line in body:
        print(" | ".join(value.ljust(widths[idx]) for idx, value in enumerate(line)))


def _format_metric(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def save_summary_plots(rows: Sequence[dict], output_dir: Path) -> list[Path]:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib.pyplot as plt

    plots_dir = output_dir / "plots"
    ensure_dir(plots_dir)
    saved_paths: list[Path] = []
    metrics = {
        "classifier_acc": "Classifier Accuracy",
        "attack_success_rate": "Attack Success Rate",
        "psnr": "PSNR",
        "ssim": "SSIM",
    }

    filtered = [
        row for row in rows
        if row.get("eval_channel") in ("awgn", "rayleigh") and row.get("snr_db") is not None
    ]
    if not filtered:
        return saved_paths

    groups = {}
    for row in filtered:
        key = (row["model"], row["train_channel"], row["eval_channel"])
        groups.setdefault(key, []).append(row)

    for (model_name, train_channel, eval_channel), group_rows in groups.items():
        for metric_key, metric_label in metrics.items():
            figure, axis = plt.subplots(figsize=(7, 4))
            plotted = False
            for attack_location in ATTACK_LOCATIONS:
                location_rows = [
                    row for row in group_rows if row.get("attack_location") == attack_location
                ]
                location_rows.sort(key=lambda item: float(item["snr_db"]), reverse=True)
                x_values = [float(row["snr_db"]) for row in location_rows]
                y_values = [float(row[metric_key]) for row in location_rows if row.get(metric_key) is not None]
                if len(x_values) != len(y_values) or not x_values:
                    continue
                axis.plot(x_values, y_values, marker="o", label=attack_location)
                plotted = True

            if not plotted:
                plt.close(figure)
                continue

            axis.set_title(f"{model_name} trained on {train_channel}, eval {eval_channel}")
            axis.set_xlabel("Eval SNR (dB)")
            axis.set_ylabel(metric_label)
            axis.grid(True, alpha=0.3)
            axis.invert_xaxis()
            axis.legend()
            filename = (
                f"{metric_key}_{model_name}_train_{train_channel}_eval_{eval_channel}.png"
            )
            output_path = plots_dir / filename
            figure.tight_layout()
            figure.savefig(output_path)
            plt.close(figure)
            saved_paths.append(output_path)

    return saved_paths


def save_channel_adversary_plots(rows: Sequence[dict], output_dir: Path) -> list[Path]:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib.pyplot as plt
    import numpy as np

    def condition_order(channel: Optional[str], snr_db: Optional[float]) -> tuple[int, float]:
        if channel == "error_free":
            return (0, 0.0)
        if channel == "awgn":
            return (1, -float(snr_db if snr_db is not None else -1e9))
        if channel == "rayleigh":
            return (2, -float(snr_db if snr_db is not None else -1e9))
        return (3, 0.0)

    def mode_label(raw_mode: str) -> str:
        return "EoT-PGD" if raw_mode == "eot_pgd" else raw_mode.upper()

    plots_dir = output_dir / "plots"
    ensure_dir(plots_dir)
    saved_paths: list[Path] = []

    matched_rows = [
        row
        for row in rows
        if row.get("attack_location") in ("input", "latent")
        and row.get("adversary_mode") in ("pgd", "eot_pgd")
        and row.get("attack_condition") == row.get("eval_condition")
        and row.get("attack_channel") == row.get("eval_channel")
        and row.get("attack_snr_db") == row.get("eval_snr_db")
        and row.get("eval_channel") in ("awgn", "rayleigh")
        and row.get("eval_snr_db") is not None
        and row.get("attack_success_rate") is not None
    ]

    matched_groups = {}
    for row in matched_rows:
        key = (
            row["model"],
            row["train_channel"],
            row["attack_location"],
            row["eval_channel"],
        )
        matched_groups.setdefault(key, []).append(row)

    for (model_name, train_channel, attack_location, eval_channel), group_rows in matched_groups.items():
        figure, axis = plt.subplots(figsize=(7, 4))
        plotted = False
        for adversary_mode in ("pgd", "eot_pgd"):
            mode_rows = [
                row for row in group_rows
                if row.get("adversary_mode") == adversary_mode
            ]
            mode_rows.sort(key=lambda item: float(item["eval_snr_db"]), reverse=True)
            x_values = [float(row["eval_snr_db"]) for row in mode_rows]
            y_values = [float(row["attack_success_rate"]) for row in mode_rows]
            if not x_values:
                continue
            axis.plot(x_values, y_values, marker="o", label=mode_label(adversary_mode))
            plotted = True

        if not plotted:
            plt.close(figure)
            continue

        axis.set_title(
            f"{model_name} trained on {train_channel}, {attack_location} attack, eval {eval_channel}"
        )
        axis.set_xlabel("Matched Eval SNR (dB)")
        axis.set_ylabel("Attack Success Rate")
        axis.set_ylim(0.0, 1.0)
        axis.grid(True, alpha=0.3)
        axis.invert_xaxis()
        axis.legend()
        output_path = (
            plots_dir
            / f"attack_success_rate_compare_{model_name}_train_{train_channel}_{attack_location}_{eval_channel}.png"
        )
        figure.tight_layout()
        figure.savefig(output_path)
        plt.close(figure)
        saved_paths.append(output_path)

    heatmap_rows = [
        row
        for row in rows
        if row.get("attack_location") in ("input", "latent")
        and row.get("adversary_mode") in ("pgd", "eot_pgd")
        and row.get("attack_condition") is not None
        and row.get("eval_condition") is not None
        and row.get("attack_success_rate") is not None
    ]

    heatmap_groups = {}
    for row in heatmap_rows:
        key = (
            row["model"],
            row["train_channel"],
            row["attack_location"],
            row["adversary_mode"],
        )
        heatmap_groups.setdefault(key, []).append(row)

    for (model_name, train_channel, attack_location, adversary_mode), group_rows in heatmap_groups.items():
        attack_condition_rows = {}
        eval_condition_rows = {}
        for row in group_rows:
            attack_key = (
                row["attack_condition"],
                row.get("attack_channel"),
                row.get("attack_snr_db"),
            )
            eval_key = (
                row["eval_condition"],
                row.get("eval_channel"),
                row.get("eval_snr_db"),
            )
            attack_condition_rows[attack_key] = row
            eval_condition_rows[eval_key] = row

        attack_keys = sorted(
            attack_condition_rows.keys(),
            key=lambda item: condition_order(item[1], item[2]),
        )
        eval_keys = sorted(
            eval_condition_rows.keys(),
            key=lambda item: condition_order(item[1], item[2]),
        )
        if not attack_keys or not eval_keys:
            continue

        attack_index = {key: idx for idx, key in enumerate(attack_keys)}
        eval_index = {key: idx for idx, key in enumerate(eval_keys)}
        matrix = np.full((len(attack_keys), len(eval_keys)), np.nan, dtype=float)

        for row in group_rows:
            attack_key = (
                row["attack_condition"],
                row.get("attack_channel"),
                row.get("attack_snr_db"),
            )
            eval_key = (
                row["eval_condition"],
                row.get("eval_channel"),
                row.get("eval_snr_db"),
            )
            matrix[attack_index[attack_key], eval_index[eval_key]] = float(row["attack_success_rate"])

        figure_width = max(7.0, 0.8 * len(eval_keys))
        figure_height = max(4.5, 0.6 * len(attack_keys))
        figure, axis = plt.subplots(figsize=(figure_width, figure_height))
        image = axis.imshow(matrix, aspect="auto", vmin=0.0, vmax=1.0, cmap="viridis")
        axis.set_title(
            f"{model_name} trained on {train_channel}, {attack_location}, {mode_label(adversary_mode)}"
        )
        axis.set_xlabel("Evaluation condition")
        axis.set_ylabel("Attack condition")
        axis.set_xticks(range(len(eval_keys)))
        axis.set_xticklabels([key[0] for key in eval_keys], rotation=45, ha="right")
        axis.set_yticks(range(len(attack_keys)))
        axis.set_yticklabels([key[0] for key in attack_keys])
        colorbar = figure.colorbar(image, ax=axis)
        colorbar.set_label("Attack Success Rate")
        output_path = (
            plots_dir
            / f"transfer_heatmap_{model_name}_train_{train_channel}_{attack_location}_{adversary_mode}.png"
        )
        figure.tight_layout()
        figure.savefig(output_path)
        plt.close(figure)
        saved_paths.append(output_path)

    return saved_paths
