import os
import torch
import torch.nn as nn
from dataclasses import dataclass
from tqdm import tqdm
from contextlib import nullcontext

@dataclass
class TrainerConfig:
    num_epochs: int = 10
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ckpt_path: str = './checkpoints/best.pt'
    log_every: int = 100

    use_amp: bool = True                  # enable automatic mixed precision
    amp_dtype: str = "auto"               # "auto" | "bf16" | "fp16"
    grad_accum_steps: int = 1             # accumulate N steps before optimizer.step()
    clip_grad_norm: float = 0.0           # 0 disables
    compile_model: bool = False           # torch.compile for speed (PyTorch 2+)

class Trainer:
    def __init__(
            self,
            pipeline,
            optimizer,
            train_loader,
            val_loader = None,
            loss_fn = nn.MSELoss(),
            config = None,
            metrics = None,
            print_fn = print,
            lr_scheduler = None
    ):
        self.cfg = config
        self.pipeline = pipeline.to(self.cfg.device)
        if self.cfg.compile_model and hasattr(torch, "compile"):
            self.pipeline.encoder = torch.compile(self.pipeline.encoder)
            self.pipeline.decoder = torch.compile(self.pipeline.decoder)

        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.print = print_fn
        self.metrics = metrics if metrics is not None else {}
        self.lr_scheduler = lr_scheduler

        self.best_val_loss = float('inf')
        self.epoch = 0
        self.global_step = 0

        # History to access outside
        self.history = {"train_loss": [], "val_loss": []}
        for name in self.metrics:
            self.history[f"val_{name}"] = []

        self.device_type = "cuda" if self.cfg.device.startswith("cuda") and torch.cuda.is_available() else "cpu"
        self.autocast_dtype = self._resolve_amp_dtype(self.cfg.amp_dtype)
        self.amp_enabled = bool(self.cfg.use_amp) and (self.device_type in ("cuda", "cpu"))
        # GradScaler is enabled only for fp16 + CUDA - bf16/CPU doesn’t use scaler
        use_scaler = self.amp_enabled and self.device_type == "cuda" and self.autocast_dtype == torch.float16
        self.scaler = torch.amp.GradScaler(enabled=use_scaler)

    def _resolve_amp_dtype(self, amp_dtype_str):
        if not self.cfg.use_amp:
            return torch.float32
        if amp_dtype_str == "bf16":
            return torch.bfloat16
        if amp_dtype_str == "fp16":
            return torch.float16
        if self.device_type == "cuda":
            major, minor = torch.cuda.get_device_capability() if torch.cuda.is_available() else (0, 0)
            return torch.bfloat16 if major >= 8 else torch.float16
        else:
            # CPU autocast supports bf16
            return torch.bfloat16

    def _autocast_ctx(self):
        if self.amp_enabled:
            return torch.autocast(device_type=self.device_type, dtype=self.autocast_dtype)
        return nullcontext()

    def train(self):
        for epoch in range(self.cfg.num_epochs):
            self.epoch = epoch
            train_loss = self._train_step()
            logs = {'epoch': epoch, 'train_loss': train_loss}

            if self.val_loader is not None:
                val_logs = self._eval_step()
                logs.update(val_logs)

                if self.cfg.ckpt_path and val_logs['val_loss'] < self.best_val_loss:
                    self.best_val_loss = val_logs['val_loss']
                    self._save(self.cfg.ckpt_path)
            else:
                if self.cfg.ckpt_path:
                    self._save(self.cfg.ckpt_path)

            # history
            self.history["train_loss"].append(train_loss)
            if "val_loss" in logs:
                self.history["val_loss"].append(logs["val_loss"])
                for name in self.metrics:
                    key = f"val_{name}"
                    if key in logs:
                        self.history[key].append(logs[key])

            # optional scheduler (epoch-wise)
            if self.lr_scheduler is not None and hasattr(self.lr_scheduler, "step"):
                try:
                    self.lr_scheduler.step(logs.get("val_loss", train_loss))
                except TypeError:
                    self.lr_scheduler.step()

            self.print(
                f"[epoch {epoch:03d}] "
                + " ".join(f"{k}={v:.4f}" for k, v in logs.items() if isinstance(v, (int, float)))
            )

    def _train_step(self):
        self.pipeline.train()
        running = 0.0
        n = 0
        accum = max(1, int(self.cfg.grad_accum_steps))
        self.optimizer.zero_grad(set_to_none=True)

        progress = tqdm(self.train_loader, desc=f"Epoch {self.epoch+1}/{self.cfg.num_epochs}", leave=False)
        for i, batch in enumerate(progress):
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(self.cfg.device, non_blocking=True)

            with self._autocast_ctx():
                # Handle the pipeline's outputs dynamically
                outputs = self.pipeline(x)
                x_hat = outputs[0]  # Primary output
                aux = outputs[1] if len(outputs) > 1 else {}  # Auxiliary outputs

                loss = self.loss_fn(x_hat, x) / accum  # scale for grad accumulation

            if self.scaler.is_enabled():
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            step_boundary = ((i + 1) % accum == 0)
            if step_boundary:
                # If using scaler, unscale before clipping
                if self.scaler.is_enabled():
                    self.scaler.unscale_(self.optimizer)

                # Grad clipping
                if self.cfg.clip_grad_norm and self.cfg.clip_grad_norm > 0.0:
                    nn.utils.clip_grad_norm_(self.pipeline.parameters(), self.cfg.clip_grad_norm)

                # Optimizer step (scaled or not)
                if self.scaler.is_enabled():
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.optimizer.zero_grad(set_to_none=True)

            running += float(loss.item()) * accum  # undo the division for reporting
            n += 1
            self.global_step += 1

            avg_loss = running / n
            progress.set_postfix({"loss": f"{avg_loss:.4f}",
                                  "amp": f"{'bf16' if self.autocast_dtype==torch.bfloat16 else ('fp16' if self.autocast_dtype==torch.float16 else 'off')}"})

        return running / max(n, 1)

    @torch.no_grad()
    def _eval_step(self):
        self.pipeline.eval()
        total = 0.0
        n = 0
        metric_sums = {name: 0.0 for name in self.metrics}

        progress = tqdm(self.val_loader, desc="Validating", leave=False)

        # Use autocast in eval too for speed (safe with no_grad)
        with self._autocast_ctx():
            for batch in progress:
                x = batch[0] if isinstance(batch, (list, tuple)) else batch
                x = x.to(self.cfg.device, non_blocking=True)

                # Handle the pipeline's outputs dynamically
                outputs = self.pipeline(x)
                y_hat = outputs[0]  # Primary output
                aux = outputs[1] if len(outputs) > 1 else {}  # Auxiliary outputs

                loss = self.loss_fn(y_hat, x)

                total += float(loss.item())
                n += 1

                # metrics computed in float32 for stability
                y_hat_f = y_hat.float()
                x_f = x.float()
                for name, fn in self.metrics.items():
                    metric_sums[name] += float(fn(y_hat_f, x_f))

                avg_loss = total / n
                progress.set_postfix({"val_loss": f"{avg_loss:.4f}"})

        logs = {"val_loss": total / max(n, 1)}
        for name, s in metric_sums.items():
            logs[f"val_{name}"] = s / max(n, 1)
        return logs

    def _save(self, path):
        state = {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "pipeline": self.pipeline.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict() if hasattr(self.scaler, "state_dict") else None,
            "config": vars(self.cfg) if hasattr(self.cfg, "__dict__") else None,
        }
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(state, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.cfg.device)
        self.pipeline.load_state_dict(ckpt["pipeline"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        if "scaler" in ckpt and ckpt["scaler"] is not None and hasattr(self.scaler, "load_state_dict"):
            # Load scaler only if it was enabled (safe even if disabled; it’s a no-op)
            self.scaler.load_state_dict(ckpt["scaler"])
        self.epoch = ckpt.get("epoch", 0)
        self.global_step = ckpt.get("global_step", 0)