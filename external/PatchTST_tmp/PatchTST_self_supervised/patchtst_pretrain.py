import argparse
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, random_split

from datautils import get_dls as get_generic_dls
from src.basics import set_device
from src.callback.patch_mask import PatchMaskCB
from src.callback.tracking import SaveModelCB, ValidateFiniteCB, WandbLoggingCB
from src.callback.transforms import RevInCB
from src.learner import Learner
from src.models.patchTST import PatchTST

FORCE_DATA_DIR = "/home/user/Genesis/data/train_04272026"
FORCE_INDEX_CSV = "train_04272026_thin_15pct.csv"
PURE_FORCE_COLS = [
    "left_fx", "left_fy", "left_fz", "left_tx", "left_ty", "left_tz",
    "right_fx", "right_fy", "right_fz", "right_tx", "right_ty", "right_tz",
]


class ForcePretrainDataset(Dataset):
    """Fast in-memory force dataset for pretraining."""

    def __init__(self, context_points: int, base_dir: str = FORCE_DATA_DIR, use_cols=None):
        super().__init__()
        self.context_points = context_points
        self.base_dir = base_dir
        self.use_cols = use_cols or PURE_FORCE_COLS

        index_csv = os.path.join(base_dir, FORCE_INDEX_CSV)
        if not os.path.exists(index_csv):
            raise FileNotFoundError(index_csv)

        df_index = pd.read_csv(index_csv)
        required_cols = {"csv_path", "start", "emb_index"}
        missing = required_cols - set(df_index.columns)
        if missing:
            raise ValueError(f"Missing columns {missing} in {index_csv}")

        csv_cache = {}
        for rel_path in df_index["csv_path"].unique():
            full_path = self._resolve_csv_path(rel_path)
            if full_path is None:
                continue
            csv_cache[rel_path] = pd.read_csv(full_path, usecols=self.use_cols).values.astype("float32")

        self.force_segments = []
        self.text_embs = []
        for _, row in df_index.iterrows():
            csv_rel = row["csv_path"]
            arr = csv_cache.get(csv_rel)
            if arr is None:
                continue

            start = int(row["start"])
            seg = arr[start : start + self.context_points, :]
            if seg.shape[0] != self.context_points:
                continue
            if not np.isfinite(seg).all():
                continue

            emb_file = os.path.join(base_dir, "bert_emb", f"{row['emb_index']}.pt")
            if not os.path.exists(emb_file):
                continue
            try:
                emb = torch.load(emb_file, map_location="cpu")
            except Exception:
                continue

            if not torch.is_tensor(emb):
                try:
                    emb = torch.tensor(emb)
                except Exception:
                    continue

            if emb.ndim == 2 and emb.shape[0] == 1:
                emb = emb.squeeze(0)
            emb = emb.detach().float().cpu()
            if not torch.isfinite(emb).all():
                continue

            self.force_segments.append(seg)
            self.text_embs.append(emb)

        if len(self.force_segments) == 0:
            raise RuntimeError("No usable force samples were loaded")

    def _resolve_csv_path(self, path_value: str):
        candidates = [
            path_value,
            os.path.join(self.base_dir, path_value),
            os.path.join(self.base_dir, "csv", path_value),
        ]
        for cand in candidates:
            if isinstance(cand, str) and os.path.exists(cand):
                return cand
        return None

    def __len__(self):
        return len(self.force_segments)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.force_segments[idx])
        y = self.text_embs[idx]
        return x, y


parser = argparse.ArgumentParser()
# Dataset and dataloader
parser.add_argument("--dset_pretrain", type=str, default="force", help="dataset name")
parser.add_argument("--context_points", type=int, default=80, help="sequence length")
parser.add_argument("--target_points", type=int, default=96, help="forecast horizon")
parser.add_argument("--batch_size", type=int, default=256, help="batch size")
parser.add_argument("--num_workers", type=int, default=16, help="number of workers for DataLoader")
parser.add_argument("--scaler", type=str, default="standard", help="scale the input data")
parser.add_argument("--features", type=str, default="M", help="for multivariate model or univariate model")
parser.add_argument("--pin_memory", type=int, default=1, help="pin CPU memory for faster H2D copy")
parser.add_argument("--persistent_workers", type=int, default=1, help="keep workers alive across epochs")
parser.add_argument("--prefetch_factor", type=int, default=4, help="prefetch batches per worker")
parser.add_argument("--drop_last", type=int, default=1, help="drop last partial train batch")
parser.add_argument("--use_fast_force_dataloader", type=int, default=1, help="use in-file force loader")
# Patch
parser.add_argument("--patch_len", type=int, default=10, help="patch length")
parser.add_argument("--stride", type=int, default=10, help="stride between patch")
# RevIN
parser.add_argument("--revin", type=int, default=1, help="reversible instance normalization")
# Model args
parser.add_argument("--n_layers", type=int, default=3, help="number of Transformer layers")
parser.add_argument("--n_heads", type=int, default=16, help="number of Transformer heads")
parser.add_argument("--d_model", type=int, default=128, help="Transformer d_model")
parser.add_argument("--d_ff", type=int, default=768, help="Tranformer MLP dimension")
parser.add_argument("--dropout", type=float, default=0.2, help="Transformer dropout")
parser.add_argument("--head_dropout", type=float, default=0.2, help="head dropout")
# Pretrain mask
parser.add_argument("--mask_ratio", type=float, default=0.4, help="masking ratio for the input")
# Optimization args
parser.add_argument("--n_epochs_pretrain", type=int, default=20, help="number of pre-training epochs")
parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
# Train/val split from training data
parser.add_argument("--val_from_train_pct", type=float, default=0.10, help="validation split ratio from train")
parser.add_argument("--seed", type=int, default=42, help="random seed for splitting and training")
# model id to keep track of the number of models saved
parser.add_argument("--pretrained_model_id", type=str, default="04052026", help="id of the saved pretrained model")
parser.add_argument("--model_type", type=str, default="based_model", help="for multivariate model or univariate model")
parser.add_argument("--device", type=str, default="cuda")

args = parser.parse_args()
print("args:", args)
args.save_path = f"saved_models/{args.dset_pretrain}/masked_patchtst/{args.model_type}/"
os.makedirs(args.save_path, exist_ok=True)


def _seed_everything(seed: int):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _loader_kwargs(args):
    kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": bool(args.pin_memory),
        "drop_last": bool(args.drop_last),
    }
    if args.num_workers > 0:
        kwargs["persistent_workers"] = bool(args.persistent_workers)
        kwargs["prefetch_factor"] = args.prefetch_factor
    return kwargs


# device setup
def init_device():
    torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    set_device()
    d = torch.device("cuda")
    _ = torch.randn(1, device=d) * 0.0
    print(f"Using CUDA device: {torch.cuda.current_device()} ({torch.cuda.get_device_name(torch.cuda.current_device())})")
    return d


DEVICE = init_device()


def get_model(c_in, args):
    num_patch = (max(args.context_points, args.patch_len) - args.patch_len) // args.stride + 1
    print("number of patches:", num_patch)

    model = PatchTST(
        c_in=c_in,
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
    model.to(DEVICE)
    print("number of model params", sum(p.numel() for p in model.parameters() if p.requires_grad))
    return model


def build_dls(args):
    """Build dataloaders once and reuse for both lr_finder and pretraining."""
    if args.dset_pretrain == "force" and bool(args.use_fast_force_dataloader):
        base_ds = ForcePretrainDataset(context_points=args.context_points)
        total_len = len(base_ds)

        if args.val_from_train_pct and args.val_from_train_pct > 0:
            val_len = max(1, int(round(total_len * args.val_from_train_pct)))
            train_len = max(1, total_len - val_len)
            if train_len + val_len > total_len:
                val_len = total_len - train_len
        else:
            train_len, val_len = total_len, 0

        g = torch.Generator()
        g.manual_seed(args.seed if args.seed is not None else 42)

        if val_len > 0:
            train_subset, val_subset = random_split(base_ds, [train_len, val_len], generator=g)
        else:
            train_subset, val_subset = base_ds, None

        train_loader = DataLoader(
            train_subset,
            batch_size=args.batch_size,
            shuffle=True,
            **_loader_kwargs(args),
        )

        valid_loader = None
        if val_subset is not None:
            val_kwargs = _loader_kwargs(args)
            val_kwargs["drop_last"] = False
            valid_loader = DataLoader(
                val_subset,
                batch_size=args.batch_size,
                shuffle=False,
                **val_kwargs,
            )

        x0, y0 = train_subset[0]
        dls = SimpleNamespace()
        dls.train = train_loader
        dls.valid = valid_loader
        dls.test = None
        dls.vars = x0.shape[1]
        dls.len = args.context_points
        dls.c = y0.shape[0]

        print(
            f"[split-fast] Train/Val from force train: train={len(train_subset)} "
            f"val={0 if val_subset is None else len(val_subset)} (pct={args.val_from_train_pct:.2f})"
        )
        return dls

    # fallback path keeps compatibility for non-force datasets
    args.dset = args.dset_pretrain
    dls = get_generic_dls(args)

    if args.val_from_train_pct and args.val_from_train_pct > 0 and dls.train is not None:
        base_ds = dls.train.dataset
        total_len = len(base_ds)
        val_len = max(1, int(round(total_len * args.val_from_train_pct)))
        train_len = max(1, total_len - val_len)
        if train_len + val_len > total_len:
            val_len = total_len - train_len

        g = torch.Generator()
        g.manual_seed(args.seed if args.seed is not None else 42)
        train_subset, val_subset = random_split(base_ds, [train_len, val_len], generator=g)

        train_loader = DataLoader(
            train_subset,
            batch_size=args.batch_size,
            shuffle=True,
            **_loader_kwargs(args),
        )

        val_kwargs = _loader_kwargs(args)
        val_kwargs["drop_last"] = False
        valid_loader = DataLoader(
            val_subset,
            batch_size=args.batch_size,
            shuffle=False,
            **val_kwargs,
        )

        dls.train = train_loader
        dls.valid = valid_loader

        x0, y0 = train_subset[0]
        dls.vars, dls.len = x0.shape[1], args.context_points
        dls.c = y0.shape[0]
        print(f"[split] Train/Val from train: train={train_len} val={val_len} (pct={args.val_from_train_pct:.2f})")

    return dls


def find_lr(dls):
    model = get_model(dls.vars, args)
    loss_func = torch.nn.MSELoss(reduction="mean")

    cbs = [RevInCB(dls.vars, denorm=False)] if args.revin else []
    cbs += [
        PatchMaskCB(patch_len=args.patch_len, stride=args.stride, mask_ratio=args.mask_ratio),
        ValidateFiniteCB(check_pred=False),
        WandbLoggingCB(project="icra_ssl", config=vars(args)),
    ]

    learn = Learner(dls, model, loss_func, lr=args.lr, cbs=cbs)
    learn.device = DEVICE

    suggested_lr = learn.lr_finder()
    print("suggested_lr", suggested_lr)
    return suggested_lr


def pretrain_func(dls, lr=args.lr):
    model = get_model(dls.vars, args)
    loss_func = torch.nn.MSELoss(reduction="mean")

    cbs = [RevInCB(dls.vars, denorm=False)] if args.revin else []
    cbs += [
        PatchMaskCB(patch_len=args.patch_len, stride=args.stride, mask_ratio=args.mask_ratio),
        ValidateFiniteCB(check_pred=False),
        SaveModelCB(
            monitor="valid_loss",
            path=args.save_path,
            fname="pretrained_model_id" + str(args.pretrained_model_id),
        ),
        WandbLoggingCB(project="icra_ssl", config=vars(args)),
    ]

    learn = Learner(dls, model, loss_func, lr=lr, cbs=cbs)
    learn.device = DEVICE
    learn.fit_one_cycle(n_epochs=args.n_epochs_pretrain, lr_max=lr)


if __name__ == "__main__":
    _seed_everything(args.seed)
    dls = build_dls(args)
    suggested_lr = find_lr(dls)
    pretrain_func(dls, suggested_lr)
    print("pretraining completed")
