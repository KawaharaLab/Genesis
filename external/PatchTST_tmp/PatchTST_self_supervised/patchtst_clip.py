import os
import math
import argparse
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from src.models.patchTST import PatchTST
from src.callback.patch_mask import create_patch

try:
    import wandb
    _WANDB_AVAILABLE = True
except Exception:
    _WANDB_AVAILABLE = False


################################################################################
# Dataset (unchanged: use precomputed text embeddings as-is)
################################################################################

DEFAULT_FORCE_COLS = [
    "left_fx","left_fy","left_fz","left_tx","left_ty","left_tz",
    "right_fx","right_fy","right_fz","right_tx","right_ty","right_tz"
]


class ForceContrastiveDataset(Dataset):
    """
    Returns (force_seq, clip_embedding)
    force_seq: [seq_len, nvars]
    clip_embedding: tensor [dim]
    """
    def __init__(self, data_dir: str, index_csv: str = 'train_thin_15pct.csv',
                 seq_len: int = 80, use_cols: List[str] = None):
        super().__init__()
        self.data_dir = data_dir
        self.index_csv = os.path.join(data_dir, index_csv)
        self.seq_len = seq_len
        self.use_cols = use_cols or DEFAULT_FORCE_COLS

        if not os.path.exists(self.index_csv):
            raise FileNotFoundError(self.index_csv)
        df = pd.read_csv(self.index_csv)
        required = {"csv_path", "start", "clip_index"}
        miss = required - set(df.columns)
        if miss:
            raise ValueError(f"Missing columns {miss} in {self.index_csv}")

        self.force_segments = []
        self.clip_embeddings = []

        # cache each csv
        csv_cache = {}
        unique_paths = df['csv_path'].unique()
        for p in unique_paths:
            full_p = os.path.join(data_dir, 'csv', p)
            if os.path.exists(full_p):
                csv_cache[p] = pd.read_csv(full_p, usecols=self.use_cols).values.astype('float32')

        for _, row in df.iterrows():
            p = row['csv_path']
            if p not in csv_cache: continue
            start = int(row['start'])
            seg = csv_cache[p][start:start+self.seq_len, :]
            if seg.shape[0] != self.seq_len:  # skip short
                continue
            emb_file = os.path.join(data_dir, 'clip_emb', f"{row['clip_index']}.pt")
            if not os.path.exists(emb_file):
                continue
            try:
                t_emb = torch.load(emb_file, map_location='cpu')
            except Exception:
                continue
            # Force segment shape -> [seq_len, nvars]
            self.force_segments.append(seg)
            # Clip embedding assumed 1D or [1,D]
            if isinstance(t_emb, torch.Tensor) and t_emb.ndim == 2 and t_emb.shape[0] == 1:
                t_emb = t_emb.squeeze(0)
            self.clip_embeddings.append(t_emb.float())

        if len(self.force_segments) == 0:
            raise RuntimeError(f"No usable pairs in {data_dir}")

    def __len__(self):
        return len(self.force_segments)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.force_segments[idx])          # [seq_len, nvars]
        y = self.clip_embeddings[idx]                          # [dim]
        return x, y


################################################################################
# InfoNCE (CLIP-style) Contrastive Loss
################################################################################

class InfoNCELoss(nn.Module):
    """Symmetric InfoNCE (CLIP-style) with learnable temperature and optional bias.

    Loss = 0.5 * (CE(img->txt) + CE(txt->img)) where logits = t * (x @ y^T) + b
    with t = exp(t_prime) (learnable), and b is a learnable scalar bias.
    """
    def __init__(self, initial_t_prime: float = None, initial_b: float = 0.0):
        super().__init__()
        if initial_t_prime is None:
            # log(1/0.07) is a common default for CLIP logit scale
            initial_t_prime = math.log(1/0.07)
        self.t_prime = nn.Parameter(torch.tensor(float(initial_t_prime)))
        self.b = nn.Parameter(torch.tensor(float(initial_b)))
        self.ce = nn.CrossEntropyLoss()

    def forward(self, img_emb: torch.Tensor, txt_emb: torch.Tensor):
        # img_emb, txt_emb: [N, D] (assumed L2-normalized)
        n = img_emb.size(0)
        t = torch.exp(self.t_prime)
        logits = torch.matmul(img_emb, txt_emb.t()) * t + self.b  # [N, N]
        target = torch.arange(n, device=logits.device)
        loss_i2t = self.ce(logits, target)
        loss_t2i = self.ce(logits.t(), target)
        return 0.5 * (loss_i2t + loss_t2i)


################################################################################
# PatchTST Contrastive Wrapper (backbone + projection)
################################################################################

class PatchTSTContrastive(nn.Module):
    def __init__(self, nvars: int, context_points: int, patch_len: int, stride: int,
                 n_layers: int, n_heads: int, d_model: int, d_ff: int, dropout: float,
                 head_dropout: float, out_embed_dim: int, pool: str = 'mean'):
        super().__init__()
        self.context_points = context_points
        self.patch_len = patch_len
        self.stride = stride
        self.pool = pool

        num_patch = (max(context_points, patch_len) - patch_len) // stride + 1
        # Use head_type='pretrain' to reuse backbone weights; we'll add our projection head.
        self.backbone = PatchTST(
            c_in=nvars,
            target_dim=0,             # unused
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
            act='relu',
            head_type='pretrain',
            res_attention=False,
        ).backbone  # only keep encoder part

        proj_layers = [nn.Linear(d_model, out_embed_dim)] if d_model == out_embed_dim else [
            nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(head_dropout), nn.Linear(d_model, out_embed_dim)
        ]
        self.proj = nn.Sequential(*proj_layers)

    def forward(self, x: torch.Tensor):
        """
        x: [B, seq_len, nvars]
        Returns: normalized embedding [B, out_embed_dim]
        """
        xb_patch, _ = create_patch(x, self.patch_len, self.stride)  # [B, num_patch, nvars, patch_len]
        z = self.backbone(xb_patch)  # [B, nvars, d_model, num_patch]
        if self.pool == 'mean':
            z = z.mean(dim=(1, 3))
        elif self.pool == 'last':
            z = z[:, :, :, -1].mean(dim=1)
        else:
            z = z.permute(0,1,3,2).contiguous().view(z.size(0), -1)
        emb = self.proj(z)
        emb = emb / (emb.norm(dim=-1, keepdim=True) + 1e-8)
        return emb


################################################################################
# Scheduler (warmup + cosine)
################################################################################

class WarmupCosineScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-5, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            return [base_lr * (self.last_epoch + 1) / self.warmup_epochs for base_lr in self.base_lrs]
        progress = (self.last_epoch - self.warmup_epochs) / max(1, (self.total_epochs - self.warmup_epochs))
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return [self.min_lr + (base_lr - self.min_lr) * cosine for base_lr in self.base_lrs]


################################################################################
# Training Loop
################################################################################

def parse_args():
    ap = argparse.ArgumentParser()
    # Data
    ap.add_argument('--data_dir', type=str, default='/home/user/Genesis/data/')
    ap.add_argument('--index_csv', type=str, default='train_old_thin_15pct.csv')
    ap.add_argument('--val_index_csv', type=str, default='eval_heavy_thin_15pct.csv')
    ap.add_argument('--context_points', type=int, default=80)
    ap.add_argument('--force_cols', type=str, default=','.join(DEFAULT_FORCE_COLS))
    # Model / backbone
    ap.add_argument('--pretrained_path', type=str, required=True, help='Path to SSL pre-trained PatchTST weights (.pth)')
    ap.add_argument('--patch_len', type=int, default=10)
    ap.add_argument('--stride', type=int, default=10)
    ap.add_argument('--n_layers', type=int, default=3)
    ap.add_argument('--n_heads', type=int, default=16)
    ap.add_argument('--d_model', type=int, default=128)
    ap.add_argument('--d_ff', type=int, default=768)
    ap.add_argument('--dropout', type=float, default=0.2)
    ap.add_argument('--head_dropout', type=float, default=0.2)
    ap.add_argument('--out_embed_dim', type=int, default=768)
    ap.add_argument('--pool', type=str, default='mean', choices=['mean','last','flatten'])
    # Optim
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--warmup_epochs', type=int, default=10)
    ap.add_argument('--peak_lr', type=float, default=5e-4)
    ap.add_argument('--weight_decay', type=float, default=0.0)
    ap.add_argument('--batch_size', type=int, default=256)
    ap.add_argument('--gradient_clipping', type=float, default=1.0)
    ap.add_argument('--min_lr', type=float, default=1e-5)
    ap.add_argument('--freeze_backbone_epochs', type=int, default=0, help='Freeze backbone for initial epochs')
    # Misc
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--num_workers', type=int, default=4)
    ap.add_argument('--project', type=str, default='patchtst_clip')
    ap.add_argument('--wandb_api_key', type=str, default='c85b817c62f441243d232b381088358e72fa2b19')  # user should replace
    ap.add_argument('--save_dir', type=str, default='saved_models/clip/')
    return ap.parse_args()


def set_seed(seed: int):
    if seed is None: return
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_backbone_weights(model: PatchTSTContrastive, pretrained_path: str, device: str):
    """Transfer weights from a pre-trained PatchTST (pretrain head)."""
    state = torch.load(pretrained_path, map_location=device)
    if 'model' in state:  # saved with optim
        state = state['model']
    model_dict = model.backbone.state_dict()
    filtered = {k.replace('backbone.', ''): v for k,v in state.items() if k.startswith('backbone.')}
    missing = []
    for k,v in filtered.items():
        if k in model_dict and model_dict[k].shape == v.shape:
            model_dict[k] = v
        else:
            missing.append(k)
    model.backbone.load_state_dict(model_dict, strict=False)
    if missing:
        print(f"[load_backbone_weights] Unmatched keys (ignored): {missing[:5]} ... total {len(missing)}")
    else:
        print("Backbone weights loaded.")


def main():
    args = parse_args()
    set_seed(args.seed)
    train_data_dir  = os.path.join(args.data_dir, "train_old/")
    eval_data_dir   = os.path.join(args.data_dir, "eval_heavy/")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    os.makedirs(args.save_dir, exist_ok=True)

    cols = args.force_cols.split(',') if args.force_cols else DEFAULT_FORCE_COLS
    dataset = ForceContrastiveDataset(train_data_dir, args.index_csv, args.context_points, cols)
    nvars = dataset[0][0].shape[1]
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)

    # Validation dataset
    try:
        val_dataset = ForceContrastiveDataset(eval_data_dir, args.val_index_csv, args.context_points, cols)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)
        print(f"Validation samples: {len(val_dataset)}")
    except Exception as e:
        val_loader = None
        print(f"[warn] validation dataset not available ({e}); skipping validation.")

    model = PatchTSTContrastive(
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
        pool=args.pool,
    ).to(device)

    load_backbone_weights(model, args.pretrained_path, device)

    criterion = InfoNCELoss().to(device)
    # Combine parameters (optionally freeze backbone initially)
    backbone_params = list(model.backbone.parameters())
    proj_params = list(model.proj.parameters())
    crit_params = list(criterion.parameters())

    optimizer = torch.optim.Adam(
        [{'params': backbone_params, 'lr': args.peak_lr},
         {'params': proj_params, 'lr': args.peak_lr},
         {'params': crit_params, 'lr': args.peak_lr}],
        lr=args.peak_lr, weight_decay=args.weight_decay
    )

    scheduler = WarmupCosineScheduler(optimizer, args.warmup_epochs, args.epochs, min_lr=args.min_lr)

    # Optionally freeze backbone
    if args.freeze_backbone_epochs > 0:
        for p in backbone_params:
            p.requires_grad = False
        print(f"Backbone frozen for first {args.freeze_backbone_epochs} epochs.")

    # W&B init
    run_name = None
    if _WANDB_AVAILABLE:
        try:
            if args.wandb_api_key and args.wandb_api_key != 'API_KEY':
                wandb.login(key=args.wandb_api_key)
            wandb.init(project=args.project, config=vars(args))
            run_name = wandb.run.name
        except Exception as e:
            print('[wandb] init failed:', e)
    if run_name is None:
        import datetime
        run_name = f"clip_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

    best_val = float('inf')
    best_path = os.path.join(args.save_dir, f"{run_name}_best.pth")

    for epoch in range(args.epochs):
        model.train()
        if args.freeze_backbone_epochs and epoch == args.freeze_backbone_epochs:
            for p in backbone_params:
                p.requires_grad = True
            print('Backbone unfrozen.')

        scheduler.step()
        total_loss = 0.0
        for step, (force_seq, txt_emb) in enumerate(loader):
            force_seq = force_seq.to(device)                  # [B, seq_len, nvars]
            txt_emb = txt_emb.to(device).float()
            # Normalize text embeddings
            txt_emb = txt_emb / (txt_emb.norm(dim=-1, keepdim=True) + 1e-8)

            optimizer.zero_grad(set_to_none=True)
            force_emb = model(force_seq)                      # [B, out_embed_dim]
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
        val_loss = None
        if val_loader is not None:
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
                torch.save({'model': model.state_dict()}, best_path)
                print(f"  * New best val_loss {best_val:.4f} -> saved {best_path}")

        log_data = {'epoch': epoch+1, 'train_loss': avg_loss, 'lr': scheduler.get_last_lr()[0],
                    't_prime': criterion.t_prime.item(), 'bias': criterion.b.item()}
        if val_loss is not None:
            log_data['valid_loss'] = val_loss
        msg = f"Epoch {epoch+1}/{args.epochs} - train: {avg_loss:.4f}"
        if val_loss is not None:
            msg += f" val: {val_loss:.4f}"
        msg += f" lr: {log_data['lr']:.6f} t: {math.exp(log_data['t_prime']):.4f} b: {log_data['bias']:.4f}"
        print(msg)
        if _WANDB_AVAILABLE and wandb.run:
            wandb.log(log_data)

        # Save periodic
        if (epoch + 1) % max(10, args.epochs//5) == 0 or (epoch + 1) == args.epochs:
            ckpt_path = os.path.join(args.save_dir, f"{run_name}_epoch{epoch+1}.pth")
            torch.save({'model': model.state_dict()}, ckpt_path)
            print('Saved', ckpt_path)

    if _WANDB_AVAILABLE and wandb.run:
        wandb.finish()


if __name__ == '__main__':
    main()
