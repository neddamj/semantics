import os
import torch
import torch.nn as nn
from dataclasses import dataclass
from tqdm import tqdm

@dataclass
class TrainerConfig:
    num_epochs: int = 10
    learning_rate: float = 1e-3
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ckpt_path: str = './checkpoints/best.pt'
    log_every: int = 100

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
            print_fn = print
    ):
        self.cfg = config
        self.pipeline = pipeline.to(self.cfg.device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.print = print_fn
        self.metrics = metrics if metrics is not None else {}

        self.best_val_loss = float('inf')
        self.epoch = 0
        self.global_step = 0

        self.history = {"train_loss": [], "val_loss": []}
        for name in self.metrics:
            self.history[f"val_{name}"] = []

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
            
            # Save the train and val losses so they are accessible outside the trainer
            self.history["train_loss"].append(train_loss)
            if "val_loss" in logs:
                self.history["val_loss"].append(logs["val_loss"])
                for name in self.metrics:
                    key = f"val_{name}"
                    if key in logs:
                        self.history[key].append(logs[key])

            self.print(
                f"[epoch {epoch:03d}] "
                + " ".join(f"{k}={v:.4f}" for k, v in logs.items() if isinstance(v, (int, float)))
            )

    def _train_step(self):
        self.pipeline.train()
        running = 0.0
        n = 0

        # wrap train_loader in tqdm
        progress = tqdm(self.train_loader, desc=f"Epoch {self.epoch+1}/{self.cfg.num_epochs}", leave=False)
        for i, batch in enumerate(progress):
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(self.cfg.device, non_blocking=True)

            x_hat = self.pipeline(x)
            loss = self.loss_fn(x_hat, x)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

            running += float(loss.item())
            n += 1
            self.global_step += 1

            # update tqdm postfix
            avg_loss = running / n
            progress.set_postfix({"loss": f"{avg_loss:.4f}"})

        return running / max(n, 1)

    @torch.no_grad()
    def _eval_step(self):
        self.pipeline.eval()
        total = 0.0
        n = 0
        metric_sums = {name: 0.0 for name in self.metrics}

        progress = tqdm(self.val_loader, desc="Validating", leave=False)

        for batch in progress:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(self.cfg.device, non_blocking=True)

            y_hat = self.pipeline(x)
            loss = self.loss_fn(y_hat, x)

            total += float(loss.item())
            n += 1

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
        }
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(state, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.cfg.device)
        self.pipeline.load_state_dict(ckpt["pipeline"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.epoch = ckpt.get("epoch", 0)
        self.global_step = ckpt.get("global_step", 0)
