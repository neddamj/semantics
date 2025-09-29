import os
import torch
import torch.nn as nn
import torch.distributed as dist
from dataclasses import dataclass
from tqdm import tqdm
from contextlib import nullcontext
from torch.nn.parallel import DistributedDataParallel


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
    task: str = "reconstruction"          # "reconstruction" | "supervised"
    distributed: bool = False             # enable DistributedDataParallel
    dist_backend: str = "nccl"            # default DDP backend
    dist_init_method: str = "env://"      # process group init
    world_size: int = -1                  # override WORLD_SIZE if > 0
    rank: int = -1                        # override RANK if >= 0
    local_rank: int = -1                  # override LOCAL_RANK if >= 0
    sync_batchnorm: bool = False          # convert BatchNorm layers to SyncBatchNorm
    ddp_broadcast_buffers: bool = True    # mirror buffers across ranks
    ddp_find_unused_parameters: bool = False  # detect unused params (may slow training)


class Trainer:
    def __init__(
            self,
            pipeline,
            optimizer,
            train_loader,
            val_loader=None,
            loss_fn=nn.MSELoss(),
            config=None,
            metrics=None,
            print_fn=print,
            lr_scheduler=None
    ):
        self.cfg = config or TrainerConfig()
        if not isinstance(self.cfg.task, str) or self.cfg.task not in ("reconstruction", "supervised"):
            raise ValueError("TrainerConfig.task must be set to 'reconstruction' or 'supervised'")

        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.metrics = metrics if metrics is not None else {}
        self.lr_scheduler = lr_scheduler

        self.distributed = False
        self.world_size = 1
        self.rank = 0
        self.local_rank = 0
        self.is_main_process = True

        if self._should_distribute():
            self._setup_distributed()

        self.device = torch.device(self.cfg.device)

        model = pipeline
        if self.cfg.sync_batchnorm:
            param_lookup = {id(param): name for name, param in model.named_parameters()}
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
            if self.optimizer is not None:
                new_params = dict(model.named_parameters())
                for group in self.optimizer.param_groups:
                    updated = []
                    for param in group['params']:
                        name = param_lookup.get(id(param))
                        if name is None:
                            raise RuntimeError(
                                "Failed to map optimizer parameters after SyncBatchNorm conversion; "
                                "construct the optimizer after enabling sync_batchnorm."
                            )
                        updated.append(new_params[name])
                    group['params'] = updated

        model = model.to(self.device)

        if self.cfg.compile_model and hasattr(torch, "compile"):
            if hasattr(model, "encoder") and hasattr(model, "decoder"):
                model.encoder = torch.compile(model.encoder)
                model.decoder = torch.compile(model.decoder)
            else:
                model = torch.compile(model)

        self.model = model
        if self.distributed:
            ddp_kwargs = {
                "broadcast_buffers": self.cfg.ddp_broadcast_buffers,
                "find_unused_parameters": self.cfg.ddp_find_unused_parameters,
            }
            if self.device.type == "cuda":
                ddp_kwargs["device_ids"] = [self.local_rank]
                ddp_kwargs["output_device"] = self.local_rank
            self.pipeline = DistributedDataParallel(self.model, **ddp_kwargs)
        else:
            self.pipeline = self.model

        self.print = print_fn if self.is_main_process else (lambda *args, **kwargs: None)

        self.best_val_loss = float('inf')
        self.epoch = 0
        self.global_step = 0

        # History to access outside
        self.history = {"train_loss": [], "val_loss": []}
        for name in self.metrics:
            self.history[f"val_{name}"] = []

        self.device_type = self.device.type
        self.autocast_dtype = self._resolve_amp_dtype(self.cfg.amp_dtype)
        self.amp_enabled = bool(self.cfg.use_amp) and (self.device_type in ("cuda", "cpu"))
        # GradScaler is enabled only for fp16 + CUDA - bf16/CPU doesn’t use scaler
        use_scaler = self.amp_enabled and self.device_type == "cuda" and self.autocast_dtype == torch.float16
        self.scaler = torch.amp.GradScaler(enabled=use_scaler)

    def _should_distribute(self):
        cfg_flag = bool(getattr(self.cfg, "distributed", False))
        env_world = int(os.environ.get("WORLD_SIZE", "1"))
        return cfg_flag or env_world > 1

    def _setup_distributed(self):
        if not dist.is_available():
            raise RuntimeError("Distributed training requested but torch.distributed is unavailable.")

        backend = getattr(self.cfg, "dist_backend", "nccl")
        init_method = getattr(self.cfg, "dist_init_method", "env://")
        world_size = self.cfg.world_size if self.cfg.world_size > 0 else int(os.environ.get("WORLD_SIZE", "1"))
        rank = self.cfg.rank if self.cfg.rank >= 0 else int(os.environ.get("RANK", "0"))

        if world_size <= 1:
            self.cfg.distributed = False
            return

        if backend == "nccl" and not torch.cuda.is_available():
            raise RuntimeError("NCCL backend requires CUDA; set TrainerConfig.dist_backend='gloo' for CPU training.")

        if not dist.is_initialized():
            dist.init_process_group(
                backend=backend,
                init_method=init_method,
                world_size=world_size,
                rank=rank,
            )

        self.distributed = True
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.is_main_process = self.rank == 0

        local_rank = self.cfg.local_rank if self.cfg.local_rank >= 0 else int(os.environ.get("LOCAL_RANK", str(self.rank)))
        self.local_rank = local_rank if local_rank >= 0 else self.rank

        if torch.cuda.is_available():
            torch.cuda.set_device(self.local_rank)
            self.cfg.device = f"cuda:{self.local_rank}"
        else:
            self.cfg.device = "cpu"
        self.cfg.distributed = True

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

    def _unwrap_model(self):
        return self.pipeline.module if isinstance(self.pipeline, DistributedDataParallel) else self.pipeline

    def _grad_sync_context(self, should_sync):
        if not self.distributed or not hasattr(self.pipeline, "no_sync"):
            return nullcontext()
        return nullcontext() if should_sync else self.pipeline.no_sync()

    def _reduce_tensor(self, tensor, op="mean"):
        if not self.distributed:
            return tensor
        t = tensor.detach().clone()
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        if op == "mean":
            t /= self.world_size
        return t

    def _set_sampler_epoch(self, epoch):
        if not self.distributed:
            return
        sampler = getattr(self.train_loader, "sampler", None)
        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        if self.val_loader is not None:
            val_sampler = getattr(self.val_loader, "sampler", None)
            if val_sampler is not None and hasattr(val_sampler, "set_epoch"):
                val_sampler.set_epoch(epoch)

    def _autocast_ctx(self):
        if self.amp_enabled:
            return torch.autocast(device_type=self.device_type, dtype=self.autocast_dtype)
        return nullcontext()

    def _split_batch(self, batch):
        if self.cfg.task == "reconstruction":
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            target = x
            return x, target
        if not (isinstance(batch, (list, tuple)) and len(batch) >= 2):
            raise ValueError("Supervised task requires batches of (inputs, targets)")
        return batch[0], batch[1]

    def train(self):
        for epoch in range(self.cfg.num_epochs):
            self.epoch = epoch
            self._set_sampler_epoch(epoch)
            train_loss = self._train_step()
            logs = {'epoch': epoch, 'train_loss': train_loss}

            if self.val_loader is not None:
                if self.distributed:
                    dist.barrier()
                val_logs = self._eval_step()
                logs.update(val_logs)

                if self.cfg.ckpt_path and self.is_main_process and val_logs['val_loss'] < self.best_val_loss:
                    self.best_val_loss = val_logs['val_loss']
                    self._save(self.cfg.ckpt_path)
            else:
                if self.cfg.ckpt_path and self.is_main_process:
                    self._save(self.cfg.ckpt_path)

            # history
            if self.is_main_process:
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
        steps = 0
        accum = max(1, int(self.cfg.grad_accum_steps))
        self.optimizer.zero_grad(set_to_none=True)

        progress = tqdm(
            self.train_loader,
            desc=f"Epoch {self.epoch+1}/{self.cfg.num_epochs}",
            leave=False,
            disable=not self.is_main_process,
        )
        for i, batch in enumerate(progress):
            x, target = self._split_batch(batch)
            x = x.to(self.device, non_blocking=True)
            if isinstance(target, torch.Tensor):
                target = target.to(self.device, non_blocking=True)

            step_boundary = ((i + 1) % accum == 0)
            with self._grad_sync_context(step_boundary):
                with self._autocast_ctx():
                    outputs = self.pipeline(x)
                    if isinstance(outputs, (list, tuple)):
                        x_hat = outputs[0]
                        aux = outputs[1] if len(outputs) > 1 else {}
                    else:
                        x_hat = outputs
                        aux = {}
                    loss = self.loss_fn(x_hat, target) / accum

            loss_to_log = loss.detach() * accum
            reduced_loss = self._reduce_tensor(loss_to_log, op="mean")

            if self.scaler.is_enabled():
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            if step_boundary:
                if self.scaler.is_enabled():
                    self.scaler.unscale_(self.optimizer)

                if self.cfg.clip_grad_norm and self.cfg.clip_grad_norm > 0.0:
                    nn.utils.clip_grad_norm_(self._unwrap_model().parameters(), self.cfg.clip_grad_norm)

                if self.scaler.is_enabled():
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.optimizer.zero_grad(set_to_none=True)

            running += float(reduced_loss.item())
            steps += 1
            self.global_step += 1

            if self.is_main_process:
                avg_loss = running / max(steps, 1)
                if self.amp_enabled:
                    if self.autocast_dtype == torch.bfloat16:
                        amp_state = "bf16"
                    elif self.autocast_dtype == torch.float16:
                        amp_state = "fp16"
                    else:
                        amp_state = "fp32"
                else:
                    amp_state = "off"
                progress.set_postfix({"loss": f"{avg_loss:.4f}", "amp": amp_state})

        return running / max(steps, 1)

    @torch.no_grad()
    def _eval_step(self):
        self.pipeline.eval()
        total = 0.0
        n = 0
        metric_sums = {name: 0.0 for name in self.metrics}

        progress = tqdm(self.val_loader, desc="Validating", leave=False, disable=not self.is_main_process)

        # Use autocast in eval too for speed (safe with no_grad)
        with self._autocast_ctx():
            for batch in progress:
                x, target = self._split_batch(batch)
                x = x.to(self.device, non_blocking=True)
                if isinstance(target, torch.Tensor):
                    target = target.to(self.device, non_blocking=True)

                outputs = self.pipeline(x)
                if isinstance(outputs, (list, tuple)):
                    y_hat = outputs[0]
                    aux = outputs[1] if len(outputs) > 1 else {}
                else:
                    y_hat = outputs
                    aux = {}

                loss = self.loss_fn(y_hat, target)

                total += float(loss.item())
                n += 1

                y_hat_f = y_hat.float()
                target_f = target.float() if isinstance(target, torch.Tensor) and target.is_floating_point() else target
                for name, fn in self.metrics.items():
                    metric_sums[name] += float(fn(y_hat_f, target_f))

                if self.is_main_process:
                    avg_loss = total / max(n, 1)
                    progress.set_postfix({"val_loss": f"{avg_loss:.4f}"})

        total_tensor = torch.tensor(total, dtype=torch.float64, device=self.device)
        total_tensor = self._reduce_tensor(total_tensor, op="sum")
        count_tensor = torch.tensor(float(n), dtype=torch.float64, device=self.device)
        count_tensor = self._reduce_tensor(count_tensor, op="sum")
        denom = max(count_tensor.item(), 1.0)

        logs = {"val_loss": total_tensor.item() / denom}
        for name, s in metric_sums.items():
            metric_tensor = torch.tensor(s, dtype=torch.float64, device=self.device)
            metric_tensor = self._reduce_tensor(metric_tensor, op="sum")
            logs[f"val_{name}"] = metric_tensor.item() / denom
        return logs

    def _save(self, path):
        if self.distributed and not self.is_main_process:
            return
        state = {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "pipeline": self._unwrap_model().state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict() if hasattr(self.scaler, "state_dict") else None,
            "config": vars(self.cfg) if hasattr(self.cfg, "__dict__") else None,
        }
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(state, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self._unwrap_model().load_state_dict(ckpt["pipeline"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        if "scaler" in ckpt and ckpt["scaler"] is not None and hasattr(self.scaler, "load_state_dict"):
            self.scaler.load_state_dict(ckpt["scaler"])
        self.epoch = ckpt.get("epoch", 0)
        self.global_step = ckpt.get("global_step", 0)
