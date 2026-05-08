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
from src.learner import transfer_weights, load_model

import wandb



################################################################################
# Dataset
################################################################################

DEFAULT_FORCE_COLS = [
    "left_fx","left_fy","left_fz","left_tx","left_ty","left_tz",
    "right_fx","right_fy","right_fz","right_tx","right_ty","right_tz"
]


class ForceContrastiveDataset(Dataset):
    """
    Returns (force_seq, bert_embedding)
    force_seq: [seq_len, nvars]
    bert_embedding: tensor [dim]
    """
    def __init__(self, data_dir: str, index_csv: str,
                 seq_len: int = 80, use_cols: List[str] = None, mode: str = 'train'):
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

        # cache each csv
        csv_cache = {}
        unique_paths = df['csv_path'].unique()
        for p in unique_paths:
            full_p = os.path.join(data_dir, 'csv', p)
            if os.path.exists(full_p):
                csv_cache[p] = pd.read_csv(full_p, usecols=self.use_cols).values.astype('float32')

        debug_limit = 100
        debug_prints = 0
        for _, row in df.iterrows():
            p = row['csv_path']
            if p not in csv_cache:
                continue
            start = int(row['start'])
            seg = csv_cache[p][start:start+self.seq_len, :]
            if seg.shape[0] != self.seq_len:  # skip short
                continue

            # Check finiteness for force segment
            if not np.isfinite(seg).all():
                nonfinite = ~np.isfinite(seg)
                nf_total = int(nonfinite.sum())
                per_col = nonfinite.sum(axis=0).tolist()
                if debug_prints < debug_limit:
                    print(f"[ForceContrastiveDataset] non-finite force segment csv={p} start={start} total={nf_total} per_col(head)={per_col[:min(12,len(per_col))]}")
                    debug_prints += 1
                continue  # drop this pair
            tmp = "bert_emb"
            emb_file = os.path.join(data_dir, tmp, f"{row[self.index_name]}.pt")
            if not os.path.exists(emb_file):
                continue
            try:
                t_emb = torch.load(emb_file, map_location='cpu')
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
            # squeeze optional leading dim
            if t_emb.ndim == 2 and t_emb.shape[0] == 1:
                t_emb = t_emb.squeeze(0)
            # Check finiteness for text embedding
            if not torch.isfinite(t_emb).all():
                nf_total = int((~torch.isfinite(t_emb)).sum().item())
                if debug_prints < debug_limit:
                    print(f"[ForceContrastiveDataset] non-finite text emb emb_index={row[self.index_name]} file={emb_file} total={nf_total}")
                    debug_prints += 1
                continue  # drop this pair

            # Valid pair -> append
            self.force_segments.append(seg)
            self.bert_embeddings.append(t_emb.float())

        if len(self.force_segments) == 0:
            raise RuntimeError(f"No usable pairs in {data_dir}")

    def __len__(self):
        return len(self.force_segments)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.force_segments[idx])          # [seq_len, nvars]
        y = self.bert_embeddings[idx]                          # [dim]
        return x, y


################################################################################
# Sigmoid (SigLIP-style) Contrastive Loss
################################################################################

class SigmoidLoss(nn.Module):
    """SigLIP-style symmetric sigmoid loss with learnable temperature & bias."""
    def __init__(self, initial_t_prime: float = 0.0, initial_b: float = 0.0):
        super().__init__()
        initial_t_prime = math.log(1/0.07)
        self.t_prime = nn.Parameter(torch.tensor(initial_t_prime))
        self.b = nn.Parameter(torch.tensor(initial_b))

    def forward(self, img_emb: torch.Tensor, txt_emb: torch.Tensor):
        # img_emb, txt_emb: [N, D]
        n = img_emb.size(0)
        device = img_emb.device
        t = torch.exp(self.t_prime)  # temperature > 0
        logits = (img_emb @ txt_emb.T) * t + self.b
        labels = 2 * torch.eye(n, device=device) - torch.ones(n, n, device=device)
        loss = -torch.sum(F.logsigmoid(labels * logits)) / n
        return loss


class InfoNCELoss(nn.Module):
    """CLIP-style symmetric InfoNCE (cross-entropy) loss with learnable logit scale.

    logit_scale = exp(logit_scale_param) where initial scale ~ 1/0.07 (CLIP default temp=0.07).
    """
    def __init__(self, initial_logit_scale: float = math.log(1/0.07), learnable: bool = True, max_scale: float = 100.0):
        super().__init__()
        if learnable:
            self.logit_scale_param = nn.Parameter(torch.tensor(initial_logit_scale))
        else:
            self.register_buffer('logit_scale_param', torch.tensor(initial_logit_scale))
        self.max_scale = max_scale

    def forward(self, img_emb: torch.Tensor, txt_emb: torch.Tensor):
        # img_emb, txt_emb: [N, D]
        n = img_emb.size(0)
        device = img_emb.device
        # Normalize just in case (upstream may already do it)
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
# PatchTST Contrastive Wrapper
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
            act='gelu',
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
        # patching expects [B, seq_len, nvars]; create_patch yields [B, num_patch, nvars, patch_len]
        xb_patch, _ = create_patch(x, self.patch_len, self.stride)
        # backbone expects [B, num_patch, nvars, patch_len]
        z = self.backbone(xb_patch)  # [B, nvars, d_model, num_patch]
        if self.pool == 'mean':
            z = z.mean(dim=(1, 3))          # [B, d_model]
        elif self.pool == 'last':
            z = z[:, :, :, -1].mean(dim=1)  # [B, d_model]
        else:  # flatten
            z = z.permute(0,1,3,2).contiguous().view(z.size(0), -1) # [B, nvars*num_patch*d_model]
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
    ap.add_argument('--data_dir', type=str, default='/home/user/Genesis/data/train_04272026/')
    ap.add_argument('--index_csv', type=str, default='train_04272026_thin_15pct.csv')
    ap.add_argument('--context_points', type=int, default=80)
    ap.add_argument('--force_cols', type=str, default=','.join(DEFAULT_FORCE_COLS))
    # Validation from train split
    ap.add_argument('--val_from_train_pct', type=float, default=0.10, help='Use this percentage of training data as validation during training.')
    # Model / backbone
    ap.add_argument('--pretrained_path', type=str, default='', help='Path to SSL pre-trained PatchTST weights (.pth). Leave empty to train from scratch.')
    ap.add_argument('--from_scratch', action='store_true', help='Ignore pretrained and start from random initialization.')
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
    ap.add_argument('--epochs', type=int, default=5000)
    ap.add_argument('--warmup_epochs', type=int, default=30)
    ap.add_argument('--peak_lr', type=float, default=5e-4)
    ap.add_argument('--weight_decay', type=float, default=0.0)
    ap.add_argument('--batch_size', type=int, default=512)
    ap.add_argument('--gradient_clipping', type=float, default=1.0)
    ap.add_argument('--min_lr', type=float, default=1e-5)
    ap.add_argument('--freeze_backbone_epochs', type=int, default=0, help='Freeze backbone for initial epochs')
    # Loss
    ap.add_argument('--loss_type', type=str, default='sigmoid', choices=['sigmoid','infonce'], help='Contrastive loss type')
    ap.add_argument('--learnable_temp', action='store_true', help='Make temperature (logit scale) learnable for InfoNCE')
    # Misc
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--num_workers', type=int, default=4)
    ap.add_argument('--wandb_api_key', type=str, default='c85b817c62f441243d232b381088358e72fa2b19')  # user will replace
    ap.add_argument('--save_dir', type=str, default='saved_models/bert/')
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


def load_backbone_weights(model: PatchTSTContrastive, pretrained_path: str, device: str):
    """Transfer weights from a pre-trained PatchTST (pretrain head)."""
    state = torch.load(pretrained_path, map_location=device)
    if 'model' in state:  # saved with optim
        state = state['model']
    # Filter keys belonging to backbone.*
    model_dict = model.backbone.state_dict()
    filtered = {k.replace('backbone.', ''): v for k,v in state.items() if k.startswith('backbone.')}
    # In some saved files keys might already start directly with encoder layers
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    # W&B init
    if args.from_scratch:
        project_name = 'patchtst_scratch_bert'
    else:
        project_name = 'patchtst_bert'
    wandb.login(key=args.wandb_api_key)
    wandb.init(project=project_name, config=vars(args))
    run_name = wandb.run.name

    os.makedirs(args.save_dir, exist_ok=True)

    cols = args.force_cols.split(',') if args.force_cols else DEFAULT_FORCE_COLS
    dataset_full = ForceContrastiveDataset(args.data_dir, args.index_csv, args.context_points, cols, mode='train')
    nvars = dataset_full[0][0].shape[1]

    # Build train/val loaders
    if args.val_from_train_pct and args.val_from_train_pct > 0:
        total_len = len(dataset_full)
        val_len = max(1, int(round(total_len * args.val_from_train_pct)))
        train_len = max(1, total_len - val_len)
        if train_len + val_len > total_len:
            val_len = total_len - train_len
        g = torch.Generator()
        g.manual_seed(args.seed if args.seed is not None else 42)
        train_subset, val_subset = torch.utils.data.random_split(dataset_full, [train_len, val_len], generator=g)
        loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
        val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)
        print(f"Train/Val split from train: train={train_len} val={val_len} (pct={args.val_from_train_pct:.2f})")
    else:
        raise ValueError("validation set must be derived from training set")
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

    if (not args.from_scratch) and args.pretrained_path:
        load_backbone_weights(model, args.pretrained_path, device)
    else:
        print("[info] Training from scratch (no pretrained backbone loaded).")

    if args.loss_type == 'sigmoid':
        criterion = SigmoidLoss().to(device)
        crit_params = list(criterion.parameters())
    else:  # InfoNCE
        criterion = InfoNCELoss(learnable=args.learnable_temp).to(device)
        # If not learnable, criterion has no grad params
        crit_params = [p for p in criterion.parameters() if p.requires_grad]
    # Combine parameters (optionally freeze backbone initially)
    backbone_params = list(model.backbone.parameters())
    proj_params = list(model.proj.parameters())
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

        log_data = {'epoch': epoch+1, 'train_loss': avg_loss, 'lr': scheduler.get_last_lr()[0]}
        # Add loss-specific parameters
        if isinstance(criterion, SigmoidLoss):
            log_data['t_prime'] = criterion.t_prime.item()
            log_data['bias'] = criterion.b.item()
        elif isinstance(criterion, InfoNCELoss):
            log_data['logit_scale'] = criterion.logit_scale_param.exp().item()
        if val_loss is not None:
            log_data['valid_loss'] = val_loss
        msg = f"Epoch {epoch+1}/{args.epochs} - train: {avg_loss:.4f}"
        if val_loss is not None:
            msg += f" val: {val_loss:.4f}"
        msg += f" lr: {log_data['lr']:.6f}"
        print(msg)
        wandb.log(log_data)

        # Save periodic
        if (epoch + 1) % max(10, args.epochs//10) == 0 or (epoch + 1) == args.epochs:
            ckpt_path = os.path.join(args.save_dir, f"{run_name}_epoch{epoch+1}.pth")
            torch.save({'model': model.state_dict()}, ckpt_path)
            print('Saved', ckpt_path)

    wandb.finish()


if __name__ == '__main__':
    main()
