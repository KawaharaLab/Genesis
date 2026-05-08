import os
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datautils import get_dls
from src.callback.patch_mask import create_patch, random_masking
from src.models.layers.revin import RevIN
from src.models.patchTST import PatchTST

import wandb


class ResidualPatchBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        pad = dilation * (kernel_size - 1) // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=pad, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=pad, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(channels)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = self.drop(self.act(self.bn1(self.conv1(x))))
        h = self.bn2(self.conv2(h))
        return self.act(x + self.drop(h))


class CNNPatchAutoEncoder(nn.Module):
    """Patch-wise CNN autoencoder. Input/Output: [B, num_patch, nvars, patch_len]."""

    def __init__(self, patch_len: int, base_channels: int, depth: int, kernel_size: int, dropout: float):
        super().__init__()
        # Keep stem/encoder names for downstream weight loading in patchtst_cnn.py.
        self.stem = nn.Sequential(
            nn.Conv1d(patch_len, base_channels, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(base_channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        blocks = []
        for i in range(depth):
            blocks.append(
                ResidualPatchBlock(
                    channels=base_channels,
                    kernel_size=kernel_size,
                    dilation=2 ** (i % 4),
                    dropout=dropout,
                )
            )
        self.encoder = nn.Sequential(*blocks)
        self.decoder = nn.Sequential(
            nn.Conv1d(base_channels, base_channels, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(base_channels, patch_len, kernel_size=1),
        )

    def forward(self, xb_patch_masked: torch.Tensor):
        # xb_patch_masked: [B, L, V, P]
        b, l, v, p = xb_patch_masked.shape
        x = xb_patch_masked.permute(0, 2, 3, 1).contiguous().view(b * v, p, l)  # [B*V, P, L]
        z = self.stem(x)
        z = self.encoder(z)
        out = self.decoder(z)  # [B*V, P, L]
        out = out.view(b, v, p, l).permute(0, 3, 1, 2).contiguous()  # [B, L, V, P]
        return out


def masked_patch_mse(preds: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
    # Same formula as PatchMaskCB._loss in patchtst_pretrain.py
    loss = (preds - target) ** 2
    loss = loss.mean(dim=-1)
    loss = (loss * mask).sum() / mask.sum().clamp_min(1.0)
    return loss


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_patchtst_ref_params(args, nvars: int) -> int:
    num_patch = (max(args.context_points, args.patch_len) - args.patch_len) // args.stride + 1
    ref = PatchTST(
        c_in=nvars,
        target_dim=args.target_points,
        patch_len=args.patch_len,
        stride=args.stride,
        num_patch=num_patch,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_model=args.d_model,
        shared_embedding=True,
        d_ff=args.d_ff,
        dropout=args.dropout,
        head_dropout=args.head_dropout,
        act="gelu",
        head_type="pretrain",
        res_attention=False,
    )
    return count_trainable_params(ref)


def choose_cnn_width(target_params: int, patch_len: int, depth: int, kernel_size: int, dropout: float):
    candidates = [32, 48, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 384]
    best_w, best_p, best_gap = candidates[0], -1, float("inf")
    for w in candidates:
        model = CNNPatchAutoEncoder(
            patch_len=patch_len,
            base_channels=w,
            depth=depth,
            kernel_size=kernel_size,
            dropout=dropout,
        )
        p = count_trainable_params(model)
        gap = abs(p - target_params)
        if gap < best_gap:
            best_w, best_p, best_gap = w, p, gap
    return best_w, best_p


def parse_args():
    parser = argparse.ArgumentParser()
    # Match patchtst_pretrain defaults/args as much as possible.
    parser.add_argument("--dset_pretrain", type=str, default="force")
    parser.add_argument("--context_points", type=int, default=80)
    parser.add_argument("--target_points", type=int, default=96)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--scaler", type=str, default="standard")
    parser.add_argument("--features", type=str, default="M")

    parser.add_argument("--patch_len", type=int, default=10)
    parser.add_argument("--stride", type=int, default=10)

    parser.add_argument("--revin", type=int, default=1)

    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--n_heads", type=int, default=16)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--d_ff", type=int, default=768)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--head_dropout", type=float, default=0.2)

    parser.add_argument("--mask_ratio", type=float, default=0.4)

    parser.add_argument("--n_epochs_pretrain", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val_from_train_pct", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)

    # CNN-specific architecture knobs
    parser.add_argument("--cnn_base_channels", type=int, default=0, help="<=0: auto-match PatchTST param scale")
    parser.add_argument("--cnn_depth", type=int, default=6)
    parser.add_argument("--cnn_kernel_size", type=int, default=5)

    parser.add_argument("--wandb_api_key", type=str, default="c85b817c62f441243d232b381088358e72fa2b19")
    parser.add_argument("--save_dir", type=str, default="saved_models/force/masked_cnn/based_model/")
    parser.add_argument("--model_id", type=str, default="1")
    return parser.parse_args()


def set_seed(seed: int):
    if seed is None:
        return
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_train_val_loaders(args):
    # Reuse exactly the same base data pipeline as patchtst_pretrain.py
    args.dset = args.dset_pretrain
    dls = get_dls(args)
    base_ds = dls.train.dataset

    total_len = len(base_ds)
    val_len = max(1, int(round(total_len * args.val_from_train_pct)))
    train_len = max(1, total_len - val_len)
    if train_len + val_len > total_len:
        val_len = total_len - train_len

    g = torch.Generator()
    g.manual_seed(args.seed if args.seed is not None else 42)
    train_subset, val_subset = torch.utils.data.random_split(base_ds, [train_len, val_len], generator=g)

    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    x0, _ = train_subset[0]
    nvars = x0.shape[1]
    print(f"[split] Train/Val from train: train={train_len} val={val_len} (pct={args.val_from_train_pct:.2f})")
    return train_loader, val_loader, nvars


def main():
    args = parse_args()
    set_seed(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    wandb.login(key=args.wandb_api_key)
    wandb.init(project="icra_ssl_cnn", config=vars(args))
    run_name = wandb.run.name

    train_loader, val_loader, nvars = build_train_val_loaders(args)

    target_params = build_patchtst_ref_params(args, nvars)
    if args.cnn_base_channels > 0:
        cnn_w = args.cnn_base_channels
        cnn_p = count_trainable_params(
            CNNPatchAutoEncoder(
                patch_len=args.patch_len,
                base_channels=cnn_w,
                depth=args.cnn_depth,
                kernel_size=args.cnn_kernel_size,
                dropout=args.dropout,
            )
        )
    else:
        cnn_w, cnn_p = choose_cnn_width(
            target_params=target_params,
            patch_len=args.patch_len,
            depth=args.cnn_depth,
            kernel_size=args.cnn_kernel_size,
            dropout=args.dropout,
        )
    print(f"Reference PatchTST params: {target_params:,}")
    print(f"CNN pretrain params: {cnn_p:,} (base_channels={cnn_w}, depth={args.cnn_depth})")

    model = CNNPatchAutoEncoder(
        patch_len=args.patch_len,
        base_channels=cnn_w,
        depth=args.cnn_depth,
        kernel_size=args.cnn_kernel_size,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    revin = RevIN(nvars, eps=1e-5, affine=False).to(device) if args.revin else None

    best_val = float("inf")
    best_path = os.path.join(args.save_dir, f"cnn_pretrained_model_id{args.model_id}_{run_name}.pth")

    for epoch in range(args.n_epochs_pretrain):
        model.train()
        total = 0.0
        t_steps = 0
        for xb, _ in train_loader:
            xb = xb.to(device).float()  # [B,T,V]
            if revin is not None:
                xb = revin(xb, "norm")

            xb_patch, _ = create_patch(xb, args.patch_len, args.stride)  # [B,L,V,P]
            xb_mask, _, mask, _ = random_masking(xb_patch, args.mask_ratio)
            mask = mask.bool().float()

            optimizer.zero_grad(set_to_none=True)
            pred = model(xb_mask)
            loss = masked_patch_mse(pred, xb_patch, mask)
            loss.backward()
            optimizer.step()

            total += loss.item()
            t_steps += 1

        train_loss = total / max(1, t_steps)

        model.eval()
        v_total = 0.0
        v_steps = 0
        with torch.no_grad():
            for xb, _ in val_loader:
                xb = xb.to(device).float()
                if revin is not None:
                    xb = revin(xb, "norm")
                xb_patch, _ = create_patch(xb, args.patch_len, args.stride)
                xb_mask, _, mask, _ = random_masking(xb_patch, args.mask_ratio)
                mask = mask.bool().float()
                pred = model(xb_mask)
                v_loss = masked_patch_mse(pred, xb_patch, mask)
                v_total += v_loss.item()
                v_steps += 1
        val_loss = v_total / max(1, v_steps)

        print(f"Epoch {epoch+1}/{args.n_epochs_pretrain} - train: {train_loss:.6f} val: {val_loss:.6f}")
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "valid_loss": val_loss,
            "target_patchtst_params": target_params,
            "cnn_params": cnn_p,
            "cnn_base_channels": cnn_w,
        })

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model": model.state_dict(),
                    "cnn_base_channels": cnn_w,
                    "cnn_depth": args.cnn_depth,
                    "target_patchtst_params": target_params,
                    "cnn_params": cnn_p,
                    "patch_len": args.patch_len,
                    "stride": args.stride,
                },
                best_path,
            )
            print(f"  * New best val_loss {best_val:.6f} -> saved {best_path}")

    wandb.finish()


if __name__ == "__main__":
    main()
