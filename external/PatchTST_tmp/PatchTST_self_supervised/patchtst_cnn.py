import os
import math
import argparse
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from src.models.patchTST import PatchTST

import wandb


################################################################################
# Dataset
################################################################################

DEFAULT_FORCE_COLS = [
    "left_fx", "left_fy", "left_fz", "left_tx", "left_ty", "left_tz",
    "right_fx", "right_fy", "right_fz", "right_tx", "right_ty", "right_tz"
]


class ForceContrastiveDataset(Dataset):
    """
    Returns (force_seq, bert_embedding)
    force_seq: [seq_len, nvars]
    bert_embedding: tensor [dim]
    """

    def __init__(
        self,
        data_dir: str,
        index_csv: str,
        seq_len: int = 80,
        use_cols: List[str] = None,
        mode: str = "train",
    ):
        super().__init__()
        self.data_dir = data_dir
        self.index_csv = os.path.join(data_dir, index_csv)
        self.seq_len = seq_len
        self.use_cols = use_cols or DEFAULT_FORCE_COLS
        self.index_name = "emb_index"

        if not os.path.exists(self.index_csv):
            raise FileNotFoundError(self.index_csv)
        df = pd.read_csv(self.index_csv)
        required = {"csv_path", "start", self.index_name}
        miss = required - set(df.columns)
        if miss:
            raise ValueError(f"Missing columns {miss} in {self.index_csv}")

        self.force_segments = []
        self.bert_embeddings = []

        csv_cache = {}
        unique_paths = df["csv_path"].unique()
        for p in unique_paths:
            full_p = os.path.join(data_dir, "csv", p)
            if os.path.exists(full_p):
                csv_cache[p] = pd.read_csv(full_p, usecols=self.use_cols).values.astype("float32")

        debug_limit = 100
        debug_prints = 0
        for _, row in df.iterrows():
            p = row["csv_path"]
            if p not in csv_cache:
                continue
            start = int(row["start"])
            seg = csv_cache[p][start : start + self.seq_len, :]
            if seg.shape[0] != self.seq_len:
                continue

            if not np.isfinite(seg).all():
                nonfinite = ~np.isfinite(seg)
                nf_total = int(nonfinite.sum())
                per_col = nonfinite.sum(axis=0).tolist()
                if debug_prints < debug_limit:
                    print(
                        f"[ForceContrastiveDataset] non-finite force segment csv={p} "
                        f"start={start} total={nf_total} per_col(head)={per_col[:min(12,len(per_col))]}"
                    )
                    debug_prints += 1
                continue

            emb_file = os.path.join(data_dir, "bert_emb", f"{row[self.index_name]}.pt")
            if not os.path.exists(emb_file):
                continue
            try:
                t_emb = torch.load(emb_file, map_location="cpu")
            except Exception as e:
                if debug_prints < debug_limit:
                    print(f"[ForceContrastiveDataset] failed to load text emb: {emb_file} err={e}")
                    debug_prints += 1
                continue

            if isinstance(t_emb, torch.Tensor):
                t_emb = t_emb.detach().cpu()
            else:
                try:
                    t_emb = torch.tensor(t_emb)
                except Exception:
                    if debug_prints < debug_limit:
                        print(f"[ForceContrastiveDataset] text emb not tensor-like: {emb_file}")
                        debug_prints += 1
                    continue

            if t_emb.ndim == 2 and t_emb.shape[0] == 1:
                t_emb = t_emb.squeeze(0)

            if not torch.isfinite(t_emb).all():
                nf_total = int((~torch.isfinite(t_emb)).sum().item())
                if debug_prints < debug_limit:
                    print(
                        f"[ForceContrastiveDataset] non-finite text emb emb_index={row[self.index_name]} "
                        f"file={emb_file} total={nf_total}"
                    )
                    debug_prints += 1
                continue

            self.force_segments.append(seg)
            self.bert_embeddings.append(t_emb.float())

        if len(self.force_segments) == 0:
            raise RuntimeError(f"No usable pairs in {data_dir}")

    def __len__(self):
        return len(self.force_segments)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.force_segments[idx])
        y = self.bert_embeddings[idx]
        return x, y


################################################################################
# Losses
################################################################################


class SigmoidLoss(nn.Module):
    """SigLIP-style symmetric sigmoid loss with learnable temperature & bias."""

    def __init__(self, initial_t_prime: float = 0.0, initial_b: float = 0.0):
        super().__init__()
        initial_t_prime = math.log(1 / 0.07)
        self.t_prime = nn.Parameter(torch.tensor(initial_t_prime))
        self.b = nn.Parameter(torch.tensor(initial_b))

    def forward(self, img_emb: torch.Tensor, txt_emb: torch.Tensor):
        n = img_emb.size(0)
        device = img_emb.device
        t = torch.exp(self.t_prime)
        logits = (img_emb @ txt_emb.T) * t + self.b
        labels = 2 * torch.eye(n, device=device) - torch.ones(n, n, device=device)
        loss = -torch.sum(F.logsigmoid(labels * logits)) / n
        return loss


class InfoNCELoss(nn.Module):
    """CLIP-style symmetric InfoNCE loss with learnable logit scale."""

    def __init__(
        self,
        initial_logit_scale: float = math.log(1 / 0.07),
        learnable: bool = True,
        max_scale: float = 100.0,
    ):
        super().__init__()
        if learnable:
            self.logit_scale_param = nn.Parameter(torch.tensor(initial_logit_scale))
        else:
            self.register_buffer("logit_scale_param", torch.tensor(initial_logit_scale))
        self.max_scale = max_scale

    def forward(self, img_emb: torch.Tensor, txt_emb: torch.Tensor):
        n = img_emb.size(0)
        device = img_emb.device
        img_emb = img_emb / (img_emb.norm(dim=-1, keepdim=True) + 1e-8)
        txt_emb = txt_emb / (txt_emb.norm(dim=-1, keepdim=True) + 1e-8)
        scale = self.logit_scale_param.exp().clamp(max=self.max_scale)
        logits_per_image = (img_emb @ txt_emb.T) * scale
        logits_per_text = logits_per_image.T
        labels = torch.arange(n, device=device)
        loss_i = F.cross_entropy(logits_per_image, labels)
        loss_t = F.cross_entropy(logits_per_text, labels)
        return (loss_i + loss_t) / 2.0


################################################################################
# Models
################################################################################


class ResidualTemporalBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(channels)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        h = self.conv1(x)
        h = self.bn1(h)
        h = self.act(h)
        h = self.drop(h)
        h = self.conv2(h)
        h = self.bn2(h)
        return self.act(x + self.drop(h))


class CNNContrastive(nn.Module):
    """1D CNN encoder for force time-series contrastive learning."""

    def __init__(
        self,
        nvars: int,
        base_channels: int,
        depth: int,
        kernel_size: int,
        dropout: float,
        out_embed_dim: int,
        head_dropout: float,
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(nvars, base_channels, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(base_channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        blocks = []
        for i in range(depth):
            dilation = 2 ** (i % 4)
            blocks.append(
                ResidualTemporalBlock(
                    channels=base_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )
        self.encoder = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)

        if base_channels == out_embed_dim:
            self.proj = nn.Linear(base_channels, out_embed_dim)
        else:
            self.proj = nn.Sequential(
                nn.Linear(base_channels, base_channels),
                nn.GELU(),
                nn.Dropout(head_dropout),
                nn.Linear(base_channels, out_embed_dim),
            )

    def forward(self, x: torch.Tensor):
        x = x.transpose(1, 2)
        z = self.stem(x)
        z = self.encoder(z)
        z = self.pool(z).squeeze(-1)
        emb = self.proj(z)
        emb = emb / (emb.norm(dim=-1, keepdim=True) + 1e-8)
        return emb


class PatchTSTContrastiveRef(nn.Module):
    """Reference PatchTST used only for parameter-count matching."""

    def __init__(
        self,
        nvars: int,
        context_points: int,
        patch_len: int,
        stride: int,
        n_layers: int,
        n_heads: int,
        d_model: int,
        d_ff: int,
        dropout: float,
        head_dropout: float,
        out_embed_dim: int,
    ):
        super().__init__()
        num_patch = (max(context_points, patch_len) - patch_len) // stride + 1
        self.backbone = PatchTST(
            c_in=nvars,
            target_dim=0,
            patch_len=patch_len,
            stride=stride,
            num_patch=num_patch,
            n_layers=n_layers,
            n_heads=n_heads,
            d_model=d_model,
            shared_embedding=True,
            d_ff=d_ff,
            dropout=dropout,
            head_dropout=head_dropout,
            act="gelu",
            head_type="pretrain",
            res_attention=False,
        ).backbone
        if d_model == out_embed_dim:
            self.proj = nn.Sequential(nn.Linear(d_model, out_embed_dim))
        else:
            self.proj = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Dropout(head_dropout),
                nn.Linear(d_model, out_embed_dim),
            )


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def choose_cnn_width(
    nvars: int,
    out_embed_dim: int,
    depth: int,
    kernel_size: int,
    dropout: float,
    head_dropout: float,
    target_params: int,
) -> Tuple[int, int]:
    candidates = [32, 48, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 384]
    best_width = candidates[0]
    best_params = -1
    best_gap = float("inf")

    for width in candidates:
        model = CNNContrastive(
            nvars=nvars,
            base_channels=width,
            depth=depth,
            kernel_size=kernel_size,
            dropout=dropout,
            out_embed_dim=out_embed_dim,
            head_dropout=head_dropout,
        )
        pcount = count_trainable_params(model)
        gap = abs(pcount - target_params)
        if gap < best_gap:
            best_gap = gap
            best_width = width
            best_params = pcount

    return best_width, best_params


################################################################################
# Scheduler
################################################################################


class WarmupCosineScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-5, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            warmup = max(1, self.warmup_epochs)
            return [base_lr * (self.last_epoch + 1) / warmup for base_lr in self.base_lrs]
        progress = (self.last_epoch - self.warmup_epochs) / max(1, (self.total_epochs - self.warmup_epochs))
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return [self.min_lr + (base_lr - self.min_lr) * cosine for base_lr in self.base_lrs]


################################################################################
# Training
################################################################################


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="/home/user/Genesis/data/")
    ap.add_argument("--index_csv", type=str, default="train_04272026_thin_15pct.csv")
    ap.add_argument("--context_points", type=int, default=80)
    ap.add_argument("--force_cols", type=str, default=",".join(DEFAULT_FORCE_COLS))
    ap.add_argument(
        "--val_from_train_pct",
        type=float,
        default=0.10,
        help="Use this percentage of training data as validation during training.",
    )

    # CNN
    ap.add_argument("--cnn_base_channels", type=int, default=0, help="<=0: auto-match PatchTST scale")
    ap.add_argument("--cnn_depth", type=int, default=6)
    ap.add_argument("--cnn_kernel_size", type=int, default=5)
    ap.add_argument("--cnn_dropout", type=float, default=0.2)
    ap.add_argument(
        "--pretrained_path",
        type=str,
        default="",
        help="Path to CNN SSL pretraining checkpoint. stem/encoder weights will be loaded.",
    )
    ap.add_argument("--from_scratch", action="store_true", help="Ignore pretrained checkpoint even if provided.")

    # Reference PatchTST (for param-size matching)
    ap.add_argument("--patch_len", type=int, default=10)
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--n_layers", type=int, default=3)
    ap.add_argument("--n_heads", type=int, default=16)
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--d_ff", type=int, default=768)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--head_dropout", type=float, default=0.2)

    ap.add_argument("--out_embed_dim", type=int, default=768)

    # Optim
    ap.add_argument("--epochs", type=int, default=3000)
    ap.add_argument("--warmup_epochs", type=int, default=30)
    ap.add_argument("--peak_lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--gradient_clipping", type=float, default=1.0)
    ap.add_argument("--min_lr", type=float, default=1e-5)

    # Loss
    ap.add_argument("--loss_type", type=str, default="infonce", choices=["sigmoid", "infonce"])
    ap.add_argument("--learnable_temp", action="store_true")

    # Misc
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--wandb_api_key", type=str, default="c85b817c62f441243d232b381088358e72fa2b19")
    ap.add_argument("--save_dir", type=str, default="saved_models/cnn_bert/")
    return ap.parse_args()


def set_seed(seed: int):
    if seed is None:
        return
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_cnn_encoder_weights(model: CNNContrastive, pretrained_path: str, device: torch.device):
    state = torch.load(pretrained_path, map_location=device)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]

    model_state = model.state_dict()
    loaded, skipped = 0, 0
    for k, v in state.items():
        if not (k.startswith("stem.") or k.startswith("encoder.")):
            continue
        if k in model_state and model_state[k].shape == v.shape:
            model_state[k] = v
            loaded += 1
        else:
            skipped += 1
    model.load_state_dict(model_state, strict=False)
    print(f"[load_cnn_encoder_weights] loaded={loaded} skipped={skipped} from {pretrained_path}")


def main():
    args = parse_args()
    set_seed(args.seed)

    train_data_dir = args.data_dir + "train_04272026/"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    wandb.login(key=args.wandb_api_key)
    wandb.init(project="cnn_bert", config=vars(args))
    run_name = wandb.run.name

    os.makedirs(args.save_dir, exist_ok=True)

    cols = args.force_cols.split(",") if args.force_cols else DEFAULT_FORCE_COLS
    dataset_full = ForceContrastiveDataset(train_data_dir, args.index_csv, args.context_points, cols, mode="train")
    nvars = dataset_full[0][0].shape[1]

    total_len = len(dataset_full)
    val_len = max(1, int(round(total_len * args.val_from_train_pct)))
    train_len = max(1, total_len - val_len)
    if train_len + val_len > total_len:
        val_len = total_len - train_len

    g = torch.Generator()
    g.manual_seed(args.seed if args.seed is not None else 42)
    train_subset, val_subset = torch.utils.data.random_split(dataset_full, [train_len, val_len], generator=g)

    loader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )

    ref_model = PatchTSTContrastiveRef(
        nvars=nvars,
        context_points=args.context_points,
        patch_len=args.patch_len,
        stride=args.stride,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_model=args.d_model,
        d_ff=args.d_ff,
        dropout=args.dropout,
        head_dropout=args.head_dropout,
        out_embed_dim=args.out_embed_dim,
    )
    target_params = count_trainable_params(ref_model)

    if args.cnn_base_channels > 0:
        cnn_width = args.cnn_base_channels
        tmp_model = CNNContrastive(
            nvars=nvars,
            base_channels=cnn_width,
            depth=args.cnn_depth,
            kernel_size=args.cnn_kernel_size,
            dropout=args.cnn_dropout,
            out_embed_dim=args.out_embed_dim,
            head_dropout=args.head_dropout,
        )
        cnn_params = count_trainable_params(tmp_model)
    else:
        cnn_width, cnn_params = choose_cnn_width(
            nvars=nvars,
            out_embed_dim=args.out_embed_dim,
            depth=args.cnn_depth,
            kernel_size=args.cnn_kernel_size,
            dropout=args.cnn_dropout,
            head_dropout=args.head_dropout,
            target_params=target_params,
        )

    print(f"Reference PatchTST params: {target_params:,}")
    print(f"CNN params: {cnn_params:,} (base_channels={cnn_width}, depth={args.cnn_depth})")

    model = CNNContrastive(
        nvars=nvars,
        base_channels=cnn_width,
        depth=args.cnn_depth,
        kernel_size=args.cnn_kernel_size,
        dropout=args.cnn_dropout,
        out_embed_dim=args.out_embed_dim,
        head_dropout=args.head_dropout,
    ).to(device)
    if (not args.from_scratch) and args.pretrained_path:
        load_cnn_encoder_weights(model, args.pretrained_path, device)
    else:
        print("[info] Training from scratch (no CNN pretrained encoder loaded).")

    if args.loss_type == "sigmoid":
        criterion = SigmoidLoss().to(device)
        crit_params = list(criterion.parameters())
    else:
        criterion = InfoNCELoss(learnable=args.learnable_temp).to(device)
        crit_params = [p for p in criterion.parameters() if p.requires_grad]

    optimizer = torch.optim.Adam(
        [{"params": model.parameters(), "lr": args.peak_lr}, {"params": crit_params, "lr": args.peak_lr}],
        lr=args.peak_lr,
        weight_decay=args.weight_decay,
    )
    scheduler = WarmupCosineScheduler(optimizer, args.warmup_epochs, args.epochs, min_lr=args.min_lr)

    best_val = float("inf")
    best_path = os.path.join(args.save_dir, f"{run_name}_best.pth")

    for epoch in range(args.epochs):
        model.train()
        scheduler.step()
        total_loss = 0.0

        for step, (force_seq, txt_emb) in enumerate(loader):
            force_seq = force_seq.to(device)
            txt_emb = txt_emb.to(device).float()
            txt_emb = txt_emb / (txt_emb.norm(dim=-1, keepdim=True) + 1e-8)

            optimizer.zero_grad(set_to_none=True)
            force_emb = model(force_seq)
            loss = criterion(force_emb, txt_emb)
            if torch.isnan(loss):
                print(f"NaN loss at epoch {epoch} step {step}")
                return
            loss.backward()
            if args.gradient_clipping > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clipping)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / (step + 1)

        model.eval()
        v_total = 0.0
        v_steps = 0
        with torch.no_grad():
            for v_force, v_txt in val_loader:
                v_force = v_force.to(device)
                v_txt = v_txt.to(device).float()
                v_txt = v_txt / (v_txt.norm(dim=-1, keepdim=True) + 1e-8)
                v_force_emb = model(v_force)
                v_loss = criterion(v_force_emb, v_txt)
                v_total += v_loss.item()
                v_steps += 1
        val_loss = v_total / max(1, v_steps)

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model": model.state_dict(),
                    "cnn_base_channels": cnn_width,
                    "cnn_depth": args.cnn_depth,
                    "target_patchtst_params": target_params,
                    "cnn_params": cnn_params,
                },
                best_path,
            )
            print(f"  * New best val_loss {best_val:.4f} -> saved {best_path}")

        log_data = {
            "epoch": epoch + 1,
            "train_loss": avg_loss,
            "valid_loss": val_loss,
            "lr": scheduler.get_last_lr()[0],
            "target_patchtst_params": target_params,
            "cnn_params": cnn_params,
            "cnn_base_channels": cnn_width,
        }
        if isinstance(criterion, SigmoidLoss):
            log_data["t_prime"] = criterion.t_prime.item()
            log_data["bias"] = criterion.b.item()
        elif isinstance(criterion, InfoNCELoss):
            log_data["logit_scale"] = criterion.logit_scale_param.exp().item()

        print(
            f"Epoch {epoch + 1}/{args.epochs} - train: {avg_loss:.4f} "
            f"val: {val_loss:.4f} lr: {log_data['lr']:.6f}"
        )
        wandb.log(log_data)

        if (epoch + 1) % max(10, args.epochs // 10) == 0 or (epoch + 1) == args.epochs:
            ckpt_path = os.path.join(args.save_dir, f"{run_name}_epoch{epoch + 1}.pth")
            torch.save(
                {
                    "model": model.state_dict(),
                    "cnn_base_channels": cnn_width,
                    "cnn_depth": args.cnn_depth,
                    "target_patchtst_params": target_params,
                    "cnn_params": cnn_params,
                },
                ckpt_path,
            )
            print("Saved", ckpt_path)

    wandb.finish()


if __name__ == "__main__":
    main()
