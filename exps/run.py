#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from exps.attacks import evaluate_variant, measure_signal_and_perturbation, pgd_input_attack, pgd_latent_attack
from exps.common import (
    ATTACK_LOCATIONS,
    CIFAR10Normalizer,
    CLASSIFIER_ARCHES,
    CLASSIFIER_DEFAULTS,
    DEFAULT_CHECKPOINT_DIR,
    DEFAULT_CLASSIFIER_CHECKPOINT,
    DEFAULT_DATA_ROOT,
    DEFAULT_EVAL_SNR_DB,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_SAMPLE_COUNT,
    DEFAULT_TRAIN_SNR_DB,
    MODEL_CHOICES,
    TRAIN_CHANNEL_CHOICES,
    ReconstructionMetricSuite,
    attach_checkpoint_metadata,
    build_channel_conditions,
    build_classifier,
    build_classifier_dataloaders,
    build_eval_dataset,
    build_optimizer,
    build_semantic_dataloaders,
    build_semantic_pipeline,
    default_classifier_checkpoint_path,
    default_semantic_checkpoint_path,
    ensure_dir,
    instantiate_channel,
    load_checkpoint_state,
    make_dataloader,
    print_results_table,
    read_checkpoint_metadata,
    resolve_device,
    resolve_path,
    resolve_semantic_checkpoint_path,
    save_sample_grid,
    save_channel_adversary_plots,
    save_summary_plots,
    semantic_training_defaults,
    set_seed,
    stable_int_seed,
    write_csv,
)
from semantics.train import Trainer, TrainerConfig


DEFAULT_CHANNEL_ADVERSARY_OUTPUT_ROOT = "outputs/channel_adversary_comparison"
DEFAULT_CHANNEL_ADVERSARY_CLASSIFIER_CHECKPOINT = "checkpoints/cifar10_resnet18_classifier_strong.pt"
CHANNEL_ADVERSARY_ATTACK_LOCATIONS = ("input", "latent")
CHANNEL_ADVERSARY_MODES = ("pgd", "eot_pgd")


@dataclass
class RowAccumulator:
    model_name: str
    train_channel: str
    condition_name: str
    eval_channel: str
    snr_db: Optional[float]
    attack_location: str
    requested_spr_db: Optional[float]
    semantic_checkpoint: str
    classifier_checkpoint: str
    total_images: int = 0
    correct_predictions: int = 0
    clean_correct_candidates: int = 0
    attack_successes: int = 0
    mse_sum: float = 0.0
    psnr_sum: float = 0.0
    ssim_sum: float = 0.0
    signal_power_sum: float = 0.0
    perturbation_power_sum: float = 0.0
    achieved_spr_sum: float = 0.0
    perturbation_samples: int = 0
    sample_image: Optional[str] = None
    extra_fields: dict[str, object] = field(default_factory=dict)

    def update(
        self,
        evaluation,
        labels_cpu: torch.Tensor,
        clean_correct_mask: Optional[torch.Tensor] = None,
        attack_success_mask: Optional[torch.Tensor] = None,
        signal_power: Optional[torch.Tensor] = None,
        perturbation_power: Optional[torch.Tensor] = None,
        achieved_spr_db: Optional[torch.Tensor] = None,
    ) -> None:
        predictions = evaluation.mean_logits.argmax(dim=1)
        self.total_images += int(labels_cpu.size(0))
        self.correct_predictions += int(predictions.eq(labels_cpu).sum().item())
        self.mse_sum += float(evaluation.mse_per_image.sum().item())
        self.psnr_sum += float(evaluation.psnr_per_image.sum().item())
        self.ssim_sum += float(evaluation.ssim_per_image.sum().item())

        if clean_correct_mask is not None:
            self.clean_correct_candidates += int(clean_correct_mask.sum().item())
        if attack_success_mask is not None:
            self.attack_successes += int(attack_success_mask.sum().item())

        if signal_power is not None and perturbation_power is not None and achieved_spr_db is not None:
            self.signal_power_sum += float(signal_power.sum().item())
            self.perturbation_power_sum += float(perturbation_power.sum().item())
            self.achieved_spr_sum += float(achieved_spr_db.sum().item())
            self.perturbation_samples += int(signal_power.numel())

    def finalize(self) -> dict:
        classifier_acc = self.correct_predictions / self.total_images if self.total_images else 0.0
        if self.attack_location == "clean":
            attack_success_rate = 0.0
            signal_power = None
            perturbation_power = 0.0
            achieved_spr_db = None
        else:
            attack_success_rate = (
                self.attack_successes / self.clean_correct_candidates
                if self.clean_correct_candidates
                else 0.0
            )
            signal_power = (
                self.signal_power_sum / self.perturbation_samples
                if self.perturbation_samples
                else None
            )
            perturbation_power = (
                self.perturbation_power_sum / self.perturbation_samples
                if self.perturbation_samples
                else None
            )
            achieved_spr_db = (
                self.achieved_spr_sum / self.perturbation_samples
                if self.perturbation_samples
                else None
            )

        row = {
            "model": self.model_name,
            "train_channel": self.train_channel,
            "condition": self.condition_name,
            "eval_channel": self.eval_channel,
            "snr_db": self.snr_db,
            "attack_location": self.attack_location,
            "spr_db": self.requested_spr_db,
            "num_images": self.total_images,
            "classifier_acc": classifier_acc,
            "attack_success_rate": attack_success_rate,
            "mse": self.mse_sum / self.total_images if self.total_images else None,
            "psnr": self.psnr_sum / self.total_images if self.total_images else None,
            "ssim": self.ssim_sum / self.total_images if self.total_images else None,
            "signal_power": signal_power,
            "perturbation_power": perturbation_power,
            "achieved_spr_db": achieved_spr_db,
            "clean_correct_reference": self.clean_correct_candidates if self.attack_location != "clean" else None,
            "attack_successes": self.attack_successes if self.attack_location != "clean" else 0,
            "sample_image": self.sample_image,
            "semantic_checkpoint": self.semantic_checkpoint,
            "classifier_checkpoint": self.classifier_checkpoint,
        }
        if self.extra_fields:
            row.update(self.extra_fields)
        return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train semantic models and run input-vs-latent robustness sweeps."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    classifier_defaults = CLASSIFIER_DEFAULTS

    train_classifier = subparsers.add_parser(
        "train-classifier",
        help="Train the CIFAR-10 classifier used as the semantic metric.",
    )
    train_classifier.add_argument("--arch", default="resnet18", choices=sorted(CLASSIFIER_ARCHES))
    train_classifier.add_argument("--checkpoint", default=DEFAULT_CLASSIFIER_CHECKPOINT)
    train_classifier.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    train_classifier.add_argument("--epochs", type=int, default=classifier_defaults.epochs)
    train_classifier.add_argument("--batch-size", type=int, default=classifier_defaults.batch_size)
    train_classifier.add_argument("--lr", type=float, default=classifier_defaults.lr)
    train_classifier.add_argument("--device", default=None)
    train_classifier.add_argument("--num-workers", type=int, default=4)
    train_classifier.add_argument("--seed", type=int, default=42)
    train_classifier.add_argument("--download", action="store_true")
    train_classifier.add_argument("--pretrained", action="store_true")
    train_classifier.add_argument("--amp-dtype", default="auto", choices=["auto", "bf16", "fp16"])
    train_classifier.add_argument("--no-amp", action="store_true")

    train_semantic = subparsers.add_parser(
        "train-semantic",
        help="Train one semantic communication checkpoint for a given architecture and training channel.",
    )
    train_semantic.add_argument("--model", required=True, choices=MODEL_CHOICES)
    train_semantic.add_argument("--train-channel", required=True, choices=TRAIN_CHANNEL_CHOICES)
    train_semantic.add_argument("--train-snr-db", type=float, default=DEFAULT_TRAIN_SNR_DB)
    train_semantic.add_argument("--checkpoint", default=None)
    train_semantic.add_argument("--checkpoint-dir", default=DEFAULT_CHECKPOINT_DIR)
    train_semantic.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    train_semantic.add_argument("--epochs", type=int, default=None)
    train_semantic.add_argument("--batch-size", type=int, default=None)
    train_semantic.add_argument("--lr", type=float, default=None)
    train_semantic.add_argument("--device", default=None)
    train_semantic.add_argument("--num-workers", type=int, default=4)
    train_semantic.add_argument("--seed", type=int, default=42)
    train_semantic.add_argument("--download", action="store_true")
    train_semantic.add_argument("--amp-dtype", default="auto", choices=["auto", "bf16", "fp16"])
    train_semantic.add_argument("--no-amp", action="store_true")
    train_semantic.add_argument("--swin-bottleneck", type=int, default=16)

    run_robustness = subparsers.add_parser(
        "run-robustness",
        help="Run the clean/input-attack/latent-attack sweep across channel conditions.",
    )
    run_robustness.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        choices=["all", *MODEL_CHOICES],
    )
    run_robustness.add_argument(
        "--train-channels",
        nargs="+",
        default=list(TRAIN_CHANNEL_CHOICES),
        choices=list(TRAIN_CHANNEL_CHOICES),
    )
    run_robustness.add_argument("--train-snr-db", type=float, default=DEFAULT_TRAIN_SNR_DB)
    run_robustness.add_argument("--checkpoint-dir", default=DEFAULT_CHECKPOINT_DIR)
    run_robustness.add_argument("--classifier-checkpoint", default=DEFAULT_CLASSIFIER_CHECKPOINT)
    run_robustness.add_argument("--classifier-arch", default=None, choices=sorted(CLASSIFIER_ARCHES))
    run_robustness.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    run_robustness.add_argument("--batch-size", type=int, default=32)
    run_robustness.add_argument("--num-images", type=int, default=1024)
    run_robustness.add_argument("--full-val", action="store_true")
    run_robustness.add_argument("--device", default=None)
    run_robustness.add_argument("--num-workers", type=int, default=0)
    run_robustness.add_argument("--download", action="store_true")
    run_robustness.add_argument("--output-dir", default=DEFAULT_OUTPUT_ROOT)
    run_robustness.add_argument("--sample-count", type=int, default=DEFAULT_SAMPLE_COUNT)
    run_robustness.add_argument(
        "--eval-snr-db",
        nargs="+",
        type=float,
        default=DEFAULT_EVAL_SNR_DB,
    )
    run_robustness.add_argument("--spr-db", type=float, default=10.0)
    run_robustness.add_argument("--pgd-steps", type=int, default=10)
    run_robustness.add_argument("--pgd-step-scale", type=float, default=1.5)
    run_robustness.add_argument("--eot-samples", type=int, default=4)
    run_robustness.add_argument("--seed", type=int, default=42)
    run_robustness.add_argument("--smoke", action="store_true")
    run_robustness.add_argument("--swin-bottleneck", type=int, default=16)
    run_robustness.add_argument("--no-random-start", action="store_true")

    run_channel_adversary_comparison = subparsers.add_parser(
        "run-channel-adversary-comparison",
        help="Compare fixed-channel PGD against channel-aware EoT-PGD across attack and eval conditions.",
    )
    run_channel_adversary_comparison.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        choices=["all", *MODEL_CHOICES],
    )
    run_channel_adversary_comparison.add_argument(
        "--train-channels",
        nargs="+",
        default=list(TRAIN_CHANNEL_CHOICES),
        choices=list(TRAIN_CHANNEL_CHOICES),
    )
    run_channel_adversary_comparison.add_argument("--train-snr-db", type=float, default=DEFAULT_TRAIN_SNR_DB)
    run_channel_adversary_comparison.add_argument("--checkpoint-dir", default=DEFAULT_CHECKPOINT_DIR)
    run_channel_adversary_comparison.add_argument(
        "--classifier-checkpoint",
        default=DEFAULT_CHANNEL_ADVERSARY_CLASSIFIER_CHECKPOINT,
    )
    run_channel_adversary_comparison.add_argument(
        "--classifier-arch",
        default=None,
        choices=sorted(CLASSIFIER_ARCHES),
    )
    run_channel_adversary_comparison.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    run_channel_adversary_comparison.add_argument("--batch-size", type=int, default=32)
    run_channel_adversary_comparison.add_argument("--num-images", type=int, default=1024)
    run_channel_adversary_comparison.add_argument("--full-val", action="store_true")
    run_channel_adversary_comparison.add_argument("--device", default=None)
    run_channel_adversary_comparison.add_argument("--num-workers", type=int, default=0)
    run_channel_adversary_comparison.add_argument("--download", action="store_true")
    run_channel_adversary_comparison.add_argument(
        "--output-dir",
        default=DEFAULT_CHANNEL_ADVERSARY_OUTPUT_ROOT,
    )
    run_channel_adversary_comparison.add_argument(
        "--sample-count",
        type=int,
        default=DEFAULT_SAMPLE_COUNT,
    )
    run_channel_adversary_comparison.add_argument(
        "--eval-snr-db",
        nargs="+",
        type=float,
        default=DEFAULT_EVAL_SNR_DB,
    )
    run_channel_adversary_comparison.add_argument("--spr-db", type=float, default=10.0)
    run_channel_adversary_comparison.add_argument("--pgd-steps", type=int, default=10)
    run_channel_adversary_comparison.add_argument("--pgd-step-scale", type=float, default=1.5)
    run_channel_adversary_comparison.add_argument("--attack-eot-samples", type=int, default=4)
    run_channel_adversary_comparison.add_argument("--eval-eot-samples", type=int, default=8)
    run_channel_adversary_comparison.add_argument(
        "--attack-locations",
        nargs="+",
        default=list(CHANNEL_ADVERSARY_ATTACK_LOCATIONS),
        choices=list(CHANNEL_ADVERSARY_ATTACK_LOCATIONS),
    )
    run_channel_adversary_comparison.add_argument("--seed", type=int, default=42)
    run_channel_adversary_comparison.add_argument("--smoke", action="store_true")
    run_channel_adversary_comparison.add_argument("--swin-bottleneck", type=int, default=16)
    run_channel_adversary_comparison.add_argument("--no-random-start", action="store_true")
    return parser.parse_args()


def classifier_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    predictions = logits.argmax(dim=1)
    return float((predictions == targets).float().mean().item())


def build_reconstruction_metrics(device: torch.device):
    suite = ReconstructionMetricSuite(device)

    def psnr_metric(pred: torch.Tensor, target: torch.Tensor) -> float:
        _, psnr_values, _ = suite.measure(pred.float().clamp(0.0, 1.0), target.float())
        return float(psnr_values.mean().item())

    def ssim_metric(pred: torch.Tensor, target: torch.Tensor) -> float:
        _, _, ssim_values = suite.measure(pred.float().clamp(0.0, 1.0), target.float())
        return float(ssim_values.mean().item())

    return {"psnr": psnr_metric, "ssim": ssim_metric}


def expand_models(selected_models: list[str]) -> list[str]:
    if "all" in selected_models:
        return list(MODEL_CHOICES)
    return selected_models


def apply_smoke_settings(args: argparse.Namespace) -> None:
    args.full_val = False
    args.num_images = min(args.num_images, 32)
    args.sample_count = min(args.sample_count, 4)
    args.batch_size = min(args.batch_size, 8)
    args.pgd_steps = min(args.pgd_steps, 2)
    if hasattr(args, "eot_samples"):
        args.eot_samples = min(args.eot_samples, 2)
    if hasattr(args, "attack_eot_samples"):
        args.attack_eot_samples = min(args.attack_eot_samples, 2)
    if hasattr(args, "eval_eot_samples"):
        args.eval_eot_samples = min(args.eval_eot_samples, 2)
    if args.output_dir == DEFAULT_OUTPUT_ROOT:
        args.output_dir = "outputs/robustness_smoke"
    elif args.output_dir == DEFAULT_CHANNEL_ADVERSARY_OUTPUT_ROOT:
        args.output_dir = "outputs/channel_adversary_comparison_smoke"


def classifier_checkpoint_path(args: argparse.Namespace) -> Path:
    return default_classifier_checkpoint_path(args.checkpoint)


def semantic_checkpoint_path(args: argparse.Namespace) -> Path:
    if args.checkpoint:
        return resolve_path(args.checkpoint)
    return default_semantic_checkpoint_path(
        model_name=args.model,
        train_channel=args.train_channel,
        train_snr_db=args.train_snr_db,
        checkpoint_dir=args.checkpoint_dir,
        swin_bottleneck=args.swin_bottleneck,
    )


def train_classifier_command(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    set_seed(args.seed)

    checkpoint_path = classifier_checkpoint_path(args)
    ensure_dir(checkpoint_path.parent)

    print(f"Using device: {device}")
    print(f"Training classifier: {args.arch}")
    print(f"Saving checkpoint to: {checkpoint_path}")

    _, _, train_loader, val_loader = build_classifier_dataloaders(
        data_root=args.data_root,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        download=args.download,
    )
    classifier = build_classifier(args.arch, device=device, pretrained=args.pretrained)
    optimizer = build_optimizer(classifier.parameters(), optimizer_name="adamw", lr=args.lr)
    trainer = Trainer(
        pipeline=classifier,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=torch.nn.CrossEntropyLoss(),
        config=TrainerConfig(
            num_epochs=args.epochs,
            device=str(device),
            ckpt_path=str(checkpoint_path),
            use_amp=not args.no_amp,
            amp_dtype=args.amp_dtype,
            grad_accum_steps=1,
            clip_grad_norm=1.0,
            compile_model=False,
            task="supervised",
        ),
        metrics={"acc": classifier_accuracy},
    )
    trainer.train()
    attach_checkpoint_metadata(
        checkpoint_path,
        {
            "kind": "classifier",
            "arch": args.arch,
            "num_classes": 10,
            "pretrained": bool(args.pretrained),
            "normalization": {"mean": [*CIFAR10Normalizer().mean.flatten().tolist()], "std": [*CIFAR10Normalizer().std.flatten().tolist()]},
        },
    )
    print(f"Saved classifier checkpoint to {checkpoint_path}")


def train_semantic_command(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    set_seed(args.seed)

    defaults = semantic_training_defaults(args.model)
    batch_size = args.batch_size if args.batch_size is not None else defaults.batch_size
    epochs = args.epochs if args.epochs is not None else defaults.epochs
    lr = args.lr if args.lr is not None else defaults.lr

    checkpoint_path = semantic_checkpoint_path(args)
    ensure_dir(checkpoint_path.parent)

    print(f"Using device: {device}")
    print(f"Training semantic model: {args.model}")
    print(f"Training channel: {args.train_channel} @ {args.train_snr_db:g} dB")
    print(f"Saving checkpoint to: {checkpoint_path}")

    _, _, train_loader, val_loader = build_semantic_dataloaders(
        data_root=args.data_root,
        device=device,
        batch_size=batch_size,
        num_workers=args.num_workers,
        download=args.download,
    )
    pipeline = build_semantic_pipeline(
        model_name=args.model,
        device=device,
        train_channel=args.train_channel,
        train_snr_db=args.train_snr_db,
        swin_bottleneck=args.swin_bottleneck,
    )
    optimizer = build_optimizer(pipeline.parameters(), optimizer_name=defaults.optimizer_name, lr=lr)
    trainer = Trainer(
        pipeline=pipeline,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=torch.nn.L1Loss(),
        config=TrainerConfig(
            num_epochs=epochs,
            device=str(device),
            ckpt_path=str(checkpoint_path),
            use_amp=(not args.no_amp) and defaults.use_amp,
            amp_dtype=args.amp_dtype,
            grad_accum_steps=1,
            clip_grad_norm=1.0,
            compile_model=defaults.compile_model,
            task="reconstruction",
        ),
        metrics=build_reconstruction_metrics(device),
    )
    trainer.train()
    attach_checkpoint_metadata(
        checkpoint_path,
        {
            "kind": "semantic",
            "model_name": args.model,
            "train_channel": args.train_channel,
            "train_snr_db": args.train_snr_db,
            "swin_bottleneck": args.swin_bottleneck,
        },
    )
    print(f"Saved semantic checkpoint to {checkpoint_path}")


def load_classifier_for_eval(
    checkpoint_path: Path,
    device: torch.device,
    classifier_arch_override: Optional[str],
):
    metadata = read_checkpoint_metadata(checkpoint_path, device)
    arch = classifier_arch_override or metadata.get("arch") or "resnet18"
    classifier = build_classifier(arch=arch, device=device, pretrained=False)
    load_checkpoint_state(classifier, checkpoint_path, device=device, state_key="pipeline")
    classifier.eval()
    return classifier, arch


def eot_seed_list(
    base_seed: int,
    model_name: str,
    train_channel: str,
    condition_slug: str,
    batch_index: int,
    eot_samples: int,
    noisy: bool,
) -> list[int]:
    sample_count = max(eot_samples, 1) if noisy else 1
    return [
        stable_int_seed(
            "robustness",
            base_seed,
            model_name,
            train_channel,
            condition_slug,
            batch_index,
            sample_index,
        )
        for sample_index in range(sample_count)
    ]


def save_condition_sample(
    output_dir: Path,
    model_name: str,
    train_channel: str,
    condition_slug: str,
    attack_location: str,
    originals: torch.Tensor,
    reconstructions: torch.Tensor,
) -> Path:
    output_path = (
        output_dir
        / "samples"
        / model_name
        / f"train_{train_channel}"
        / f"{condition_slug}_{attack_location}.png"
    )
    save_sample_grid(originals, reconstructions, output_path)
    return output_path


def noisy_channel_conditions(snr_values: list[float]):
    return [
        condition
        for condition in build_channel_conditions(snr_values)
        if condition.kind != "error_free"
    ]


def channel_adversary_opt_seed_count(adversary_mode: str, attack_eot_samples: int) -> int:
    if adversary_mode == "pgd":
        return 1
    return max(attack_eot_samples, 1)


def channel_adversary_eval_seed_count(condition, eval_eot_samples: int) -> int:
    if condition.kind == "error_free":
        return 1
    return max(eval_eot_samples, 1)


def channel_adversary_opt_seed_list(
    *,
    base_seed: int,
    model_name: str,
    train_channel: str,
    attack_location: str,
    adversary_mode: str,
    attack_condition_slug: str,
    batch_index: int,
    attack_eot_samples: int,
) -> list[int]:
    sample_count = channel_adversary_opt_seed_count(adversary_mode, attack_eot_samples)
    return [
        stable_int_seed(
            "channel_adversary_opt",
            base_seed,
            model_name,
            train_channel,
            attack_location,
            adversary_mode,
            attack_condition_slug,
            batch_index,
            sample_index,
        )
        for sample_index in range(sample_count)
    ]


def channel_adversary_eval_seed_list(
    *,
    base_seed: int,
    model_name: str,
    train_channel: str,
    attack_location: str,
    adversary_mode: str,
    attack_condition_slug: str,
    eval_condition_slug: str,
    batch_index: int,
    eval_eot_samples: int,
    noisy: bool,
) -> list[int]:
    sample_count = max(eval_eot_samples, 1) if noisy else 1
    return [
        stable_int_seed(
            "channel_adversary_eval",
            base_seed,
            model_name,
            train_channel,
            attack_location,
            adversary_mode,
            attack_condition_slug,
            eval_condition_slug,
            batch_index,
            sample_index,
        )
        for sample_index in range(sample_count)
    ]


def save_channel_adversary_sample(
    *,
    output_dir: Path,
    model_name: str,
    train_channel: str,
    eval_condition_slug: str,
    attack_location: str,
    adversary_mode: str,
    attack_condition_slug: Optional[str],
    originals: torch.Tensor,
    reconstructions: torch.Tensor,
) -> Path:
    if attack_location == "clean":
        filename = f"{eval_condition_slug}_clean.png"
    else:
        filename = (
            f"{attack_condition_slug}_{eval_condition_slug}_{attack_location}_{adversary_mode}.png"
        )
    output_path = output_dir / "samples" / model_name / f"train_{train_channel}" / filename
    save_sample_grid(originals, reconstructions, output_path)
    return output_path


def print_channel_adversary_results_table(rows: list[dict]) -> None:
    if not rows:
        print("No results collected.")
        return

    def format_metric(value) -> str:
        if value is None:
            return ""
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value)

    headers = [
        "Model",
        "Train",
        "Attack",
        "Mode",
        "AtkCond",
        "EvalCond",
        "AtkSucc",
        "ClsAcc",
    ]
    body = []
    for row in rows:
        body.append(
            [
                str(row.get("model", "") or ""),
                str(row.get("train_channel", "") or ""),
                str(row.get("attack_location", "") or ""),
                str(row.get("adversary_mode", "") or ""),
                str(row.get("attack_condition", "") or ""),
                str(row.get("eval_condition", "") or ""),
                format_metric(row.get("attack_success_rate")),
                format_metric(row.get("classifier_acc")),
            ]
        )

    widths = []
    for index, header in enumerate(headers):
        widths.append(max(len(header), max(len(line[index]) for line in body)))

    divider = "-+-".join("-" * width for width in widths)
    print(" | ".join(header.ljust(widths[idx]) for idx, header in enumerate(headers)))
    print(divider)
    for line in body:
        print(" | ".join(value.ljust(widths[idx]) for idx, value in enumerate(line)))


def evaluate_channel_adversary_comparison(
    *,
    pipeline,
    classifier,
    classifier_normalizer,
    metric_suite,
    loader,
    device: torch.device,
    model_name: str,
    train_channel: str,
    semantic_checkpoint_path: Path,
    classifier_checkpoint_path: Path,
    eval_conditions,
    attack_conditions,
    args: argparse.Namespace,
    output_dir: Path,
) -> list[dict]:
    pipeline.eval()

    accumulators: dict[tuple[str, str, Optional[str], str], RowAccumulator] = {}
    for eval_condition in eval_conditions:
        eval_seed_count = channel_adversary_eval_seed_count(eval_condition, args.eval_eot_samples)
        accumulators[("clean", "clean", None, eval_condition.slug)] = RowAccumulator(
            model_name=model_name,
            train_channel=train_channel,
            condition_name=eval_condition.name,
            eval_channel=eval_condition.kind,
            snr_db=eval_condition.snr_db,
            attack_location="clean",
            requested_spr_db=None,
            semantic_checkpoint=str(semantic_checkpoint_path),
            classifier_checkpoint=str(classifier_checkpoint_path),
            extra_fields={
                "adversary_mode": "clean",
                "attack_condition": None,
                "attack_channel": None,
                "attack_snr_db": None,
                "eval_condition": eval_condition.name,
                "eval_snr_db": eval_condition.snr_db,
                "attack_eot_samples": None,
                "eval_eot_samples": eval_seed_count,
            },
        )

    for attack_location in args.attack_locations:
        for adversary_mode in CHANNEL_ADVERSARY_MODES:
            attack_seed_count = channel_adversary_opt_seed_count(
                adversary_mode,
                args.attack_eot_samples,
            )
            for attack_condition in attack_conditions:
                for eval_condition in eval_conditions:
                    eval_seed_count = channel_adversary_eval_seed_count(
                        eval_condition,
                        args.eval_eot_samples,
                    )
                    condition_name = (
                        f"{eval_condition.name} | attack {attack_condition.name} | {adversary_mode}"
                    )
                    accumulators[
                        (attack_location, adversary_mode, attack_condition.slug, eval_condition.slug)
                    ] = RowAccumulator(
                        model_name=model_name,
                        train_channel=train_channel,
                        condition_name=condition_name,
                        eval_channel=eval_condition.kind,
                        snr_db=eval_condition.snr_db,
                        attack_location=attack_location,
                        requested_spr_db=args.spr_db,
                        semantic_checkpoint=str(semantic_checkpoint_path),
                        classifier_checkpoint=str(classifier_checkpoint_path),
                        extra_fields={
                            "adversary_mode": adversary_mode,
                            "attack_condition": attack_condition.name,
                            "attack_channel": attack_condition.kind,
                            "attack_snr_db": attack_condition.snr_db,
                            "eval_condition": eval_condition.name,
                            "eval_snr_db": eval_condition.snr_db,
                            "attack_eot_samples": attack_seed_count,
                            "eval_eot_samples": eval_seed_count,
                        },
                    )

    attack_fns = {
        "input": pgd_input_attack,
        "latent": pgd_latent_attack,
    }

    for batch_index, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=device.type == "cuda")
        labels = labels.to(device, non_blocking=device.type == "cuda")
        labels_cpu = labels.detach().cpu()
        originals_cpu = images.detach().cpu()
        clean_correct_masks: dict[str, torch.Tensor] = {}

        for eval_condition in eval_conditions:
            pipeline.channel = instantiate_channel(eval_condition.kind, eval_condition.snr_db, device)
            eval_seeds = channel_adversary_eval_seed_list(
                base_seed=args.seed,
                model_name=model_name,
                train_channel=train_channel,
                attack_location="clean",
                adversary_mode="clean",
                attack_condition_slug="clean",
                eval_condition_slug=eval_condition.slug,
                batch_index=batch_index,
                eval_eot_samples=args.eval_eot_samples,
                noisy=eval_condition.kind != "error_free",
            )
            clean_eval = evaluate_variant(
                pipeline=pipeline,
                classifier=classifier,
                classifier_normalizer=classifier_normalizer,
                metric_suite=metric_suite,
                target_images=images,
                seeds=eval_seeds,
                device=device,
                adv_inputs=images,
            )
            clean_predictions = clean_eval.mean_logits.argmax(dim=1)
            clean_correct_mask = clean_predictions.eq(labels_cpu)
            clean_correct_masks[eval_condition.slug] = clean_correct_mask
            clean_accumulator = accumulators[("clean", "clean", None, eval_condition.slug)]
            clean_accumulator.update(clean_eval, labels_cpu)
            if clean_accumulator.sample_image is None and args.sample_count > 0:
                limit = min(args.sample_count, clean_eval.first_reconstruction.size(0))
                sample_path = save_channel_adversary_sample(
                    output_dir=output_dir,
                    model_name=model_name,
                    train_channel=train_channel,
                    eval_condition_slug=eval_condition.slug,
                    attack_location="clean",
                    adversary_mode="clean",
                    attack_condition_slug=None,
                    originals=originals_cpu[:limit],
                    reconstructions=clean_eval.first_reconstruction[:limit],
                )
                clean_accumulator.sample_image = str(sample_path)

        for attack_condition in attack_conditions:
            pipeline.channel = instantiate_channel(attack_condition.kind, attack_condition.snr_db, device)

            for attack_location in args.attack_locations:
                attack_fn = attack_fns[attack_location]
                attack_artifacts = {}
                attack_stats = {}

                for adversary_mode in CHANNEL_ADVERSARY_MODES:
                    opt_seeds = channel_adversary_opt_seed_list(
                        base_seed=args.seed,
                        model_name=model_name,
                        train_channel=train_channel,
                        attack_location=attack_location,
                        adversary_mode=adversary_mode,
                        attack_condition_slug=attack_condition.slug,
                        batch_index=batch_index,
                        attack_eot_samples=args.attack_eot_samples,
                    )
                    artifact = attack_fn(
                        pipeline=pipeline,
                        classifier=classifier,
                        classifier_normalizer=classifier_normalizer,
                        inputs=images,
                        target_labels=labels,
                        seeds=opt_seeds,
                        spr_db=args.spr_db,
                        steps=args.pgd_steps,
                        step_scale=args.pgd_step_scale,
                        device=device,
                        random_start=not args.no_random_start,
                    )
                    signal_power, perturbation_power, achieved_spr = measure_signal_and_perturbation(
                        artifact.signal,
                        artifact.perturbation,
                    )
                    attack_artifacts[adversary_mode] = artifact
                    attack_stats[adversary_mode] = (
                        signal_power.detach().cpu(),
                        perturbation_power.detach().cpu(),
                        achieved_spr.detach().cpu(),
                    )

                for adversary_mode, artifact in attack_artifacts.items():
                    signal_power_cpu, perturbation_power_cpu, achieved_spr_cpu = attack_stats[adversary_mode]
                    for eval_condition in eval_conditions:
                        pipeline.channel = instantiate_channel(
                            eval_condition.kind,
                            eval_condition.snr_db,
                            device,
                        )
                        eval_seeds = channel_adversary_eval_seed_list(
                            base_seed=args.seed,
                            model_name=model_name,
                            train_channel=train_channel,
                            attack_location=attack_location,
                            adversary_mode=adversary_mode,
                            attack_condition_slug=attack_condition.slug,
                            eval_condition_slug=eval_condition.slug,
                            batch_index=batch_index,
                            eval_eot_samples=args.eval_eot_samples,
                            noisy=eval_condition.kind != "error_free",
                        )
                        eval_kwargs = (
                            {"adv_inputs": artifact.adv_inputs}
                            if attack_location == "input"
                            else {"adv_latent": artifact.adv_latent}
                        )
                        attack_eval = evaluate_variant(
                            pipeline=pipeline,
                            classifier=classifier,
                            classifier_normalizer=classifier_normalizer,
                            metric_suite=metric_suite,
                            target_images=images,
                            seeds=eval_seeds,
                            device=device,
                            **eval_kwargs,
                        )
                        predictions = attack_eval.mean_logits.argmax(dim=1)
                        clean_correct_mask = clean_correct_masks[eval_condition.slug]
                        success_mask = clean_correct_mask & predictions.ne(labels_cpu)
                        accumulator = accumulators[
                            (
                                attack_location,
                                adversary_mode,
                                attack_condition.slug,
                                eval_condition.slug,
                            )
                        ]
                        accumulator.update(
                            attack_eval,
                            labels_cpu,
                            clean_correct_mask=clean_correct_mask,
                            attack_success_mask=success_mask,
                            signal_power=signal_power_cpu,
                            perturbation_power=perturbation_power_cpu,
                            achieved_spr_db=achieved_spr_cpu,
                        )
                        if (
                            accumulator.sample_image is None
                            and args.sample_count > 0
                            and attack_condition.slug == eval_condition.slug
                        ):
                            limit = min(args.sample_count, attack_eval.first_reconstruction.size(0))
                            sample_path = save_channel_adversary_sample(
                                output_dir=output_dir,
                                model_name=model_name,
                                train_channel=train_channel,
                                eval_condition_slug=eval_condition.slug,
                                attack_location=attack_location,
                                adversary_mode=adversary_mode,
                                attack_condition_slug=attack_condition.slug,
                                originals=originals_cpu[:limit],
                                reconstructions=attack_eval.first_reconstruction[:limit],
                            )
                            accumulator.sample_image = str(sample_path)

    return [accumulator.finalize() for accumulator in accumulators.values()]


def evaluate_condition(
    *,
    pipeline,
    classifier,
    classifier_normalizer,
    metric_suite,
    loader,
    device: torch.device,
    model_name: str,
    train_channel: str,
    semantic_checkpoint_path: Path,
    classifier_checkpoint_path: Path,
    condition,
    args: argparse.Namespace,
    output_dir: Path,
) -> list[dict]:
    pipeline.channel = instantiate_channel(condition.kind, condition.snr_db, device)
    pipeline.eval()

    accumulators = {
        "clean": RowAccumulator(
            model_name=model_name,
            train_channel=train_channel,
            condition_name=condition.name,
            eval_channel=condition.kind,
            snr_db=condition.snr_db,
            attack_location="clean",
            requested_spr_db=None,
            semantic_checkpoint=str(semantic_checkpoint_path),
            classifier_checkpoint=str(classifier_checkpoint_path),
        ),
        "input": RowAccumulator(
            model_name=model_name,
            train_channel=train_channel,
            condition_name=condition.name,
            eval_channel=condition.kind,
            snr_db=condition.snr_db,
            attack_location="input",
            requested_spr_db=args.spr_db,
            semantic_checkpoint=str(semantic_checkpoint_path),
            classifier_checkpoint=str(classifier_checkpoint_path),
        ),
        "latent": RowAccumulator(
            model_name=model_name,
            train_channel=train_channel,
            condition_name=condition.name,
            eval_channel=condition.kind,
            snr_db=condition.snr_db,
            attack_location="latent",
            requested_spr_db=args.spr_db,
            semantic_checkpoint=str(semantic_checkpoint_path),
            classifier_checkpoint=str(classifier_checkpoint_path),
        ),
    }

    for batch_index, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=device.type == "cuda")
        labels = labels.to(device, non_blocking=device.type == "cuda")
        labels_cpu = labels.detach().cpu()
        seeds = eot_seed_list(
            base_seed=args.seed,
            model_name=model_name,
            train_channel=train_channel,
            condition_slug=condition.slug,
            batch_index=batch_index,
            eot_samples=args.eot_samples,
            noisy=condition.kind != "error_free",
        )

        clean_eval = evaluate_variant(
            pipeline=pipeline,
            classifier=classifier,
            classifier_normalizer=classifier_normalizer,
            metric_suite=metric_suite,
            target_images=images,
            seeds=seeds,
            device=device,
            adv_inputs=images,
        )
        clean_predictions = clean_eval.mean_logits.argmax(dim=1)
        clean_correct_mask = clean_predictions.eq(labels_cpu)
        accumulators["clean"].update(clean_eval, labels_cpu)
        if accumulators["clean"].sample_image is None and args.sample_count > 0:
            limit = min(args.sample_count, clean_eval.first_reconstruction.size(0))
            sample_path = save_condition_sample(
                output_dir=output_dir,
                model_name=model_name,
                train_channel=train_channel,
                condition_slug=condition.slug,
                attack_location="clean",
                originals=images[:limit].detach().cpu(),
                reconstructions=clean_eval.first_reconstruction[:limit],
            )
            accumulators["clean"].sample_image = str(sample_path)

        input_attack = pgd_input_attack(
            pipeline=pipeline,
            classifier=classifier,
            classifier_normalizer=classifier_normalizer,
            inputs=images,
            target_labels=labels,
            seeds=seeds,
            spr_db=args.spr_db,
            steps=args.pgd_steps,
            step_scale=args.pgd_step_scale,
            device=device,
            random_start=not args.no_random_start,
        )
        input_eval = evaluate_variant(
            pipeline=pipeline,
            classifier=classifier,
            classifier_normalizer=classifier_normalizer,
            metric_suite=metric_suite,
            target_images=images,
            seeds=seeds,
            device=device,
            adv_inputs=input_attack.adv_inputs,
        )
        input_predictions = input_eval.mean_logits.argmax(dim=1)
        input_success_mask = clean_correct_mask & input_predictions.ne(labels_cpu)
        signal_power, perturbation_power, achieved_spr = measure_signal_and_perturbation(
            input_attack.signal,
            input_attack.perturbation,
        )
        accumulators["input"].update(
            input_eval,
            labels_cpu,
            clean_correct_mask=clean_correct_mask,
            attack_success_mask=input_success_mask,
            signal_power=signal_power.detach().cpu(),
            perturbation_power=perturbation_power.detach().cpu(),
            achieved_spr_db=achieved_spr.detach().cpu(),
        )
        if accumulators["input"].sample_image is None and args.sample_count > 0:
            limit = min(args.sample_count, input_eval.first_reconstruction.size(0))
            sample_path = save_condition_sample(
                output_dir=output_dir,
                model_name=model_name,
                train_channel=train_channel,
                condition_slug=condition.slug,
                attack_location="input",
                originals=images[:limit].detach().cpu(),
                reconstructions=input_eval.first_reconstruction[:limit],
            )
            accumulators["input"].sample_image = str(sample_path)

        latent_attack = pgd_latent_attack(
            pipeline=pipeline,
            classifier=classifier,
            classifier_normalizer=classifier_normalizer,
            inputs=images,
            target_labels=labels,
            seeds=seeds,
            spr_db=args.spr_db,
            steps=args.pgd_steps,
            step_scale=args.pgd_step_scale,
            device=device,
            random_start=not args.no_random_start,
        )
        latent_eval = evaluate_variant(
            pipeline=pipeline,
            classifier=classifier,
            classifier_normalizer=classifier_normalizer,
            metric_suite=metric_suite,
            target_images=images,
            seeds=seeds,
            device=device,
            adv_latent=latent_attack.adv_latent,
        )
        latent_predictions = latent_eval.mean_logits.argmax(dim=1)
        latent_success_mask = clean_correct_mask & latent_predictions.ne(labels_cpu)
        signal_power, perturbation_power, achieved_spr = measure_signal_and_perturbation(
            latent_attack.signal,
            latent_attack.perturbation,
        )
        accumulators["latent"].update(
            latent_eval,
            labels_cpu,
            clean_correct_mask=clean_correct_mask,
            attack_success_mask=latent_success_mask,
            signal_power=signal_power.detach().cpu(),
            perturbation_power=perturbation_power.detach().cpu(),
            achieved_spr_db=achieved_spr.detach().cpu(),
        )
        if accumulators["latent"].sample_image is None and args.sample_count > 0:
            limit = min(args.sample_count, latent_eval.first_reconstruction.size(0))
            sample_path = save_condition_sample(
                output_dir=output_dir,
                model_name=model_name,
                train_channel=train_channel,
                condition_slug=condition.slug,
                attack_location="latent",
                originals=images[:limit].detach().cpu(),
                reconstructions=latent_eval.first_reconstruction[:limit],
            )
            accumulators["latent"].sample_image = str(sample_path)

    return [accumulators[location].finalize() for location in ATTACK_LOCATIONS]


def run_robustness_command(args: argparse.Namespace) -> None:
    if args.smoke:
        apply_smoke_settings(args)

    device = resolve_device(args.device)
    set_seed(args.seed)

    output_dir = resolve_path(args.output_dir)
    ensure_dir(output_dir)
    print(f"Using device: {device}")
    print(f"Writing outputs to: {output_dir}")

    eval_dataset = build_eval_dataset(
        data_root=args.data_root,
        num_images=args.num_images,
        full_val=args.full_val,
        download=args.download,
    )
    eval_loader = make_dataloader(
        eval_dataset,
        batch_size=args.batch_size,
        device=device,
        shuffle=False,
        num_workers=args.num_workers,
    )

    classifier_checkpoint = resolve_path(args.classifier_checkpoint)
    classifier, classifier_arch = load_classifier_for_eval(
        checkpoint_path=classifier_checkpoint,
        device=device,
        classifier_arch_override=args.classifier_arch,
    )
    classifier_normalizer = CIFAR10Normalizer().to(device)
    metric_suite = ReconstructionMetricSuite(device)

    print(f"Loaded classifier checkpoint: {classifier_checkpoint}")
    print(f"Classifier architecture: {classifier_arch}")
    print(f"Evaluating {len(eval_dataset)} CIFAR-10 validation images")

    all_results = []
    conditions = build_channel_conditions(args.eval_snr_db)

    for model_name in expand_models(args.models):
        for train_channel in args.train_channels:
            semantic_checkpoint = resolve_semantic_checkpoint_path(
                model_name=model_name,
                train_channel=train_channel,
                train_snr_db=args.train_snr_db,
                checkpoint_dir=args.checkpoint_dir,
                swin_bottleneck=args.swin_bottleneck,
            )
            print()
            print(f"Loading semantic checkpoint: {semantic_checkpoint}")
            pipeline = build_semantic_pipeline(
                model_name=model_name,
                device=device,
                train_channel=train_channel,
                train_snr_db=args.train_snr_db,
                swin_bottleneck=args.swin_bottleneck,
            )
            load_checkpoint_state(pipeline, semantic_checkpoint, device=device, state_key="pipeline")
            pipeline.eval()

            for condition in conditions:
                print(f"[{model_name} | trained on {train_channel}] {condition.name}")
                condition_rows = evaluate_condition(
                    pipeline=pipeline,
                    classifier=classifier,
                    classifier_normalizer=classifier_normalizer,
                    metric_suite=metric_suite,
                    loader=eval_loader,
                    device=device,
                    model_name=model_name,
                    train_channel=train_channel,
                    semantic_checkpoint_path=semantic_checkpoint,
                    classifier_checkpoint_path=classifier_checkpoint,
                    condition=condition,
                    args=args,
                    output_dir=output_dir,
                )
                all_results.extend(condition_rows)

            del pipeline
            if device.type == "cuda":
                torch.cuda.empty_cache()

    csv_path = output_dir / "results.csv"
    write_csv(all_results, csv_path)
    print()
    print_results_table(all_results)
    print()
    print(f"Saved results CSV to {csv_path}")

    plot_paths = save_summary_plots(all_results, output_dir)
    if plot_paths:
        print("Saved summary plots:")
        for plot_path in plot_paths:
            print(f"  {plot_path}")


def run_channel_adversary_comparison_command(args: argparse.Namespace) -> None:
    if args.smoke:
        apply_smoke_settings(args)

    device = resolve_device(args.device)
    set_seed(args.seed)

    output_dir = resolve_path(args.output_dir)
    ensure_dir(output_dir)
    print(f"Using device: {device}")
    print(f"Writing outputs to: {output_dir}")

    eval_dataset = build_eval_dataset(
        data_root=args.data_root,
        num_images=args.num_images,
        full_val=args.full_val,
        download=args.download,
    )
    eval_loader = make_dataloader(
        eval_dataset,
        batch_size=args.batch_size,
        device=device,
        shuffle=False,
        num_workers=args.num_workers,
    )

    classifier_checkpoint = resolve_path(args.classifier_checkpoint)
    classifier, classifier_arch = load_classifier_for_eval(
        checkpoint_path=classifier_checkpoint,
        device=device,
        classifier_arch_override=args.classifier_arch,
    )
    classifier_normalizer = CIFAR10Normalizer().to(device)
    metric_suite = ReconstructionMetricSuite(device)

    print(f"Loaded classifier checkpoint: {classifier_checkpoint}")
    print(f"Classifier architecture: {classifier_arch}")
    print(f"Evaluating {len(eval_dataset)} CIFAR-10 validation images")

    all_results = []
    eval_conditions = build_channel_conditions(args.eval_snr_db)
    attack_conditions = noisy_channel_conditions(args.eval_snr_db)

    for model_name in expand_models(args.models):
        for train_channel in args.train_channels:
            semantic_checkpoint = resolve_semantic_checkpoint_path(
                model_name=model_name,
                train_channel=train_channel,
                train_snr_db=args.train_snr_db,
                checkpoint_dir=args.checkpoint_dir,
                swin_bottleneck=args.swin_bottleneck,
            )
            print()
            print(f"Loading semantic checkpoint: {semantic_checkpoint}")
            pipeline = build_semantic_pipeline(
                model_name=model_name,
                device=device,
                train_channel=train_channel,
                train_snr_db=args.train_snr_db,
                swin_bottleneck=args.swin_bottleneck,
            )
            load_checkpoint_state(pipeline, semantic_checkpoint, device=device, state_key="pipeline")
            pipeline.eval()
            print(
                f"[{model_name} | trained on {train_channel}] "
                f"{len(args.attack_locations)} attack locations, "
                f"{len(attack_conditions)} attack conditions, "
                f"{len(eval_conditions)} eval conditions"
            )
            model_rows = evaluate_channel_adversary_comparison(
                pipeline=pipeline,
                classifier=classifier,
                classifier_normalizer=classifier_normalizer,
                metric_suite=metric_suite,
                loader=eval_loader,
                device=device,
                model_name=model_name,
                train_channel=train_channel,
                semantic_checkpoint_path=semantic_checkpoint,
                classifier_checkpoint_path=classifier_checkpoint,
                eval_conditions=eval_conditions,
                attack_conditions=attack_conditions,
                args=args,
                output_dir=output_dir,
            )
            all_results.extend(model_rows)

            del pipeline
            if device.type == "cuda":
                torch.cuda.empty_cache()

    csv_path = output_dir / "results.csv"
    write_csv(all_results, csv_path)
    print()
    print_channel_adversary_results_table(all_results)
    print()
    print(f"Saved results CSV to {csv_path}")

    plot_paths = save_channel_adversary_plots(all_results, output_dir)
    if plot_paths:
        print("Saved comparison plots:")
        for plot_path in plot_paths:
            print(f"  {plot_path}")


def main() -> None:
    args = parse_args()
    if args.command == "train-classifier":
        train_classifier_command(args)
        return
    if args.command == "train-semantic":
        train_semantic_command(args)
        return
    if args.command == "run-robustness":
        run_robustness_command(args)
        return
    if args.command == "run-channel-adversary-comparison":
        run_channel_adversary_comparison_command(args)
        return
    raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
