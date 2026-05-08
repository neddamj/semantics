#!/usr/bin/env python3

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn.functional as F

from exps.common import forward_semantic_pipeline


@dataclass
class BatchEvaluation:
    mean_logits: torch.Tensor
    mean_reconstruction: torch.Tensor
    first_reconstruction: torch.Tensor
    mse_per_image: torch.Tensor
    psnr_per_image: torch.Tensor
    ssim_per_image: torch.Tensor


@dataclass
class AttackArtifact:
    adv_inputs: Optional[torch.Tensor]
    adv_latent: Optional[torch.Tensor]
    perturbation: torch.Tensor
    signal: torch.Tensor


def set_torch_seed(seed: int, device: torch.device) -> None:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def mean_power_per_sample(x: torch.Tensor) -> torch.Tensor:
    flat = x.reshape(x.size(0), -1)
    return flat.pow(2).mean(dim=1)


def l2_budget_from_spr(signal: torch.Tensor, spr_db: float) -> torch.Tensor:
    signal_power = mean_power_per_sample(signal)
    perturb_power = signal_power / (10.0 ** (spr_db / 10.0))
    return torch.sqrt(perturb_power * signal[0].numel())


def project_to_spr_ball(
    perturbation: torch.Tensor,
    signal: torch.Tensor,
    spr_db: float,
) -> torch.Tensor:
    radii = l2_budget_from_spr(signal, spr_db)
    flat = perturbation.reshape(perturbation.size(0), -1)
    norms = flat.norm(p=2, dim=1).clamp_min(1e-12)
    scales = torch.minimum(torch.ones_like(norms), radii / norms)
    projected = flat * scales.unsqueeze(1)
    return projected.view_as(perturbation)


def perturbation_step(gradient: torch.Tensor, step_sizes: torch.Tensor) -> torch.Tensor:
    grad_flat = gradient.reshape(gradient.size(0), -1)
    grad_norm = grad_flat.norm(p=2, dim=1).clamp_min(1e-12)
    normalized = gradient / grad_norm.view(-1, *([1] * (gradient.ndim - 1)))
    return normalized * step_sizes.view(-1, *([1] * (gradient.ndim - 1)))


def measure_signal_and_perturbation(
    signal: torch.Tensor,
    perturbation: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    signal_power = mean_power_per_sample(signal)
    perturbation_power = mean_power_per_sample(perturbation)
    achieved_spr_db = 10.0 * torch.log10(signal_power / perturbation_power.clamp_min(1e-12))
    return signal_power, perturbation_power, achieved_spr_db


def _random_start_like(
    signal: torch.Tensor,
    spr_db: float,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    set_torch_seed(seed, device)
    noise = torch.randn_like(signal)
    noise = project_to_spr_ball(noise, signal, spr_db)
    radii = l2_budget_from_spr(signal, spr_db)
    scales = torch.rand(signal.size(0), device=device)
    flat = noise.reshape(noise.size(0), -1)
    norms = flat.norm(p=2, dim=1).clamp_min(1e-12)
    scaled = flat * ((scales * radii) / norms).unsqueeze(1)
    return scaled.view_as(signal)


def eot_attack_loss(
    pipeline,
    classifier,
    classifier_normalizer,
    target_labels: torch.Tensor,
    seeds: Iterable[int],
    device: torch.device,
    adv_inputs: Optional[torch.Tensor] = None,
    adv_latent: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    losses = []
    for seed in seeds:
        set_torch_seed(seed, device)
        recon, _, _, _ = forward_semantic_pipeline(
            pipeline,
            x=adv_inputs,
            latent=adv_latent,
        )
        recon = recon.clamp(0.0, 1.0)
        logits = classifier(classifier_normalizer(recon))
        losses.append(F.cross_entropy(logits, target_labels))
    return torch.stack(losses, dim=0).mean()


@torch.no_grad()
def evaluate_variant(
    pipeline,
    classifier,
    classifier_normalizer,
    metric_suite,
    target_images: torch.Tensor,
    seeds: Iterable[int],
    device: torch.device,
    adv_inputs: Optional[torch.Tensor] = None,
    adv_latent: Optional[torch.Tensor] = None,
) -> BatchEvaluation:
    logits_sum = None
    recon_sum = None
    first_reconstruction = None
    mse_sum = None
    psnr_sum = None
    ssim_sum = None
    seed_count = 0

    for seed in seeds:
        set_torch_seed(seed, device)
        recon, _, _, _ = forward_semantic_pipeline(
            pipeline,
            x=adv_inputs,
            latent=adv_latent,
        )
        recon = recon.clamp(0.0, 1.0)
        logits = classifier(classifier_normalizer(recon))
        mse_values, psnr_values, ssim_values = metric_suite.measure(recon, target_images)

        logits_sum = logits if logits_sum is None else logits_sum + logits
        recon_sum = recon if recon_sum is None else recon_sum + recon
        mse_sum = mse_values if mse_sum is None else mse_sum + mse_values
        psnr_sum = psnr_values if psnr_sum is None else psnr_sum + psnr_values
        ssim_sum = ssim_values if ssim_sum is None else ssim_sum + ssim_values
        if first_reconstruction is None:
            first_reconstruction = recon.detach().cpu()
        seed_count += 1

    return BatchEvaluation(
        mean_logits=(logits_sum / seed_count).detach().cpu(),
        mean_reconstruction=(recon_sum / seed_count).detach().cpu(),
        first_reconstruction=first_reconstruction,
        mse_per_image=(mse_sum / seed_count).detach().cpu(),
        psnr_per_image=(psnr_sum / seed_count).detach().cpu(),
        ssim_per_image=(ssim_sum / seed_count).detach().cpu(),
    )


def pgd_input_attack(
    pipeline,
    classifier,
    classifier_normalizer,
    inputs: torch.Tensor,
    target_labels: torch.Tensor,
    seeds: list[int],
    spr_db: float,
    steps: int,
    step_scale: float,
    device: torch.device,
    random_start: bool = True,
) -> AttackArtifact:
    signal = inputs.detach()
    if random_start:
        perturbation = _random_start_like(signal, spr_db, seed=seeds[0] + 17, device=device)
    else:
        perturbation = torch.zeros_like(signal)
    perturbation = perturbation.detach()
    step_sizes = step_scale * l2_budget_from_spr(signal, spr_db) / max(steps, 1)

    for _ in range(max(steps, 0)):
        perturbation.requires_grad_(True)
        adv_inputs = (signal + perturbation).clamp(0.0, 1.0)
        effective_delta = adv_inputs - signal
        loss = eot_attack_loss(
            pipeline=pipeline,
            classifier=classifier,
            classifier_normalizer=classifier_normalizer,
            target_labels=target_labels,
            seeds=seeds,
            device=device,
            adv_inputs=adv_inputs,
        )
        gradient = torch.autograd.grad(loss, perturbation)[0]
        with torch.no_grad():
            perturbation = effective_delta + perturbation_step(gradient, step_sizes)
            adv_inputs = (signal + perturbation).clamp(0.0, 1.0)
            perturbation = adv_inputs - signal
            perturbation = project_to_spr_ball(perturbation, signal, spr_db)

    final_adv = (signal + perturbation).clamp(0.0, 1.0).detach()
    final_delta = final_adv - signal
    return AttackArtifact(
        adv_inputs=final_adv,
        adv_latent=None,
        perturbation=final_delta.detach(),
        signal=signal.detach(),
    )


def pgd_latent_attack(
    pipeline,
    classifier,
    classifier_normalizer,
    inputs: torch.Tensor,
    target_labels: torch.Tensor,
    seeds: list[int],
    spr_db: float,
    steps: int,
    step_scale: float,
    device: torch.device,
    random_start: bool = True,
) -> AttackArtifact:
    pipeline.eval()
    with torch.no_grad():
        _, _, clean_latent, _ = forward_semantic_pipeline(pipeline, x=inputs)
    signal = clean_latent.detach()

    if random_start:
        perturbation = _random_start_like(signal, spr_db, seed=seeds[0] + 97, device=device)
    else:
        perturbation = torch.zeros_like(signal)
    perturbation = perturbation.detach()
    step_sizes = step_scale * l2_budget_from_spr(signal, spr_db) / max(steps, 1)

    for _ in range(max(steps, 0)):
        perturbation.requires_grad_(True)
        adv_latent = signal + perturbation
        loss = eot_attack_loss(
            pipeline=pipeline,
            classifier=classifier,
            classifier_normalizer=classifier_normalizer,
            target_labels=target_labels,
            seeds=seeds,
            device=device,
            adv_latent=adv_latent,
        )
        gradient = torch.autograd.grad(loss, perturbation)[0]
        with torch.no_grad():
            perturbation = perturbation + perturbation_step(gradient, step_sizes)
            perturbation = project_to_spr_ball(perturbation, signal, spr_db)

    final_adv_latent = (signal + perturbation).detach()
    return AttackArtifact(
        adv_inputs=None,
        adv_latent=final_adv_latent,
        perturbation=perturbation.detach(),
        signal=signal.detach(),
    )
