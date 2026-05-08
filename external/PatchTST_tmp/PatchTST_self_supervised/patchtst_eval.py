import os
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import umap
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

from src.models.patchTST import PatchTST
from src.callback.patch_mask import create_patch   # patch creation function
from src.learner import load_model   # for loading state_dict

# CNN baseline model (same structure as patchtst_cnn.py)
class ResidualTemporalBlock(torch.nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.conv1 = torch.nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.bn1 = torch.nn.BatchNorm1d(channels)
        self.conv2 = torch.nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.bn2 = torch.nn.BatchNorm1d(channels)
        self.act = torch.nn.GELU()
        self.drop = torch.nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        h = self.conv1(x)
        h = self.bn1(h)
        h = self.act(h)
        h = self.drop(h)
        h = self.conv2(h)
        h = self.bn2(h)
        return self.act(x + self.drop(h))


class CNNContrastive(torch.nn.Module):
    def __init__(self, nvars: int, base_channels: int, depth: int, kernel_size: int,
                 dropout: float, out_embed_dim: int, head_dropout: float):
        super().__init__()
        self.stem = torch.nn.Sequential(
            torch.nn.Conv1d(nvars, base_channels, kernel_size=kernel_size, padding=kernel_size // 2),
            torch.nn.BatchNorm1d(base_channels),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
        )
        blocks = []
        for i in range(depth):
            blocks.append(ResidualTemporalBlock(base_channels, kernel_size, 2 ** (i % 4), dropout))
        self.encoder = torch.nn.Sequential(*blocks)
        self.pool = torch.nn.AdaptiveAvgPool1d(1)
        if base_channels == out_embed_dim:
            self.proj = torch.nn.Linear(base_channels, out_embed_dim)
        else:
            self.proj = torch.nn.Sequential(
                torch.nn.Linear(base_channels, base_channels),
                torch.nn.GELU(),
                torch.nn.Dropout(head_dropout),
                torch.nn.Linear(base_channels, out_embed_dim),
            )

    def forward(self, x: torch.Tensor):
        x = x.transpose(1, 2)
        z = self.stem(x)
        z = self.encoder(z)
        z = self.pool(z).squeeze(-1)
        emb = self.proj(z)
        emb = emb / (emb.norm(dim=-1, keepdim=True) + 1e-8)
        return emb


class ResidualMLPBlock(torch.nn.Module):
    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim * 4),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim * 4, hidden_dim),
            torch.nn.Dropout(dropout),
        )
        self.norm = torch.nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor):
        return self.norm(x + self.net(x))


class MLPContrastive(torch.nn.Module):
    def __init__(self, nvars: int, seq_len: int, hidden_dim: int, depth: int,
                 dropout: float, out_embed_dim: int, head_dropout: float):
        super().__init__()
        self.nvars = nvars
        self.seq_len = seq_len
        self.stem = torch.nn.Sequential(
            torch.nn.Linear(seq_len, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
        )
        self.encoder = torch.nn.Sequential(*[ResidualMLPBlock(hidden_dim, dropout) for _ in range(depth)])
        if hidden_dim == out_embed_dim:
            self.proj = torch.nn.Linear(hidden_dim, out_embed_dim)
        else:
            self.proj = torch.nn.Sequential(
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.GELU(),
                torch.nn.Dropout(head_dropout),
                torch.nn.Linear(hidden_dim, out_embed_dim),
            )

    def forward(self, x: torch.Tensor):
        # x: [B, T, V]
        b, t, v = x.shape
        if t != self.seq_len:
            raise ValueError(f"Expected seq_len={self.seq_len}, got {t}")
        if v != self.nvars:
            raise ValueError(f"Expected nvars={self.nvars}, got {v}")
        x = x.transpose(1, 2).contiguous().view(b * v, t)
        z = self.stem(x)
        z = self.encoder(z)
        z = z.view(b, v, -1).mean(dim=1)
        emb = self.proj(z)
        emb = emb / (emb.norm(dim=-1, keepdim=True) + 1e-8)
        return emb

# Load force evaluation dataset (validation directory)
def load_force_segments(eval_dir, index_csv="eval_thin_15pct.csv",
                        cols=None, seq_len=80, target_col="label", drop_nonfinite=True, debug=False):
    if cols is None:
        cols = [
            "left_fx","left_fy","left_fz","left_tx","left_ty","left_tz",
            "right_fx","right_fy","right_fz","right_tx","right_ty","right_tz"
        ]
    csv_path = os.path.join(eval_dir, index_csv)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)
    df = pd.read_csv(csv_path)
    required = {"csv_path","start",f"{target_col}_emb_index"}
    miss = required - set(df.columns)
    if miss:
        raise ValueError(f"missing columns: {miss}")
    # Drop rows with NaN in target column (for action/weight/interaction etc.)
    if target_col in df.columns:
        before = len(df)
        df = df.dropna(subset=[target_col])
        dropped = before - len(df)
        if dropped > 0 and debug:
            print(f"[debug] dropped {dropped} rows with NaN in target '{target_col}'")

    segments = []
    labels = []
    meta_csv_paths = []
    meta_starts = []
    skipped_nonfinite = 0
    skipped_short = 0
    for _, row in df.iterrows():
        fcsv = os.path.join(eval_dir, "csv", row["csv_path"])
        if not os.path.exists(fcsv):
            continue
        raw = pd.read_csv(fcsv)
        if any(c not in raw.columns for c in cols):
            continue
        start = int(row["start"])
        arr = raw[cols].values.astype("float32")[start:start+seq_len, :]
        if arr.shape[0] != seq_len:
            skipped_short += 1
            continue
        if drop_nonfinite and not np.isfinite(arr).all():
            skipped_nonfinite += 1
            continue
        segments.append(arr)  # [seq_len, nvars]
        meta_csv_paths.append(row["csv_path"])
        meta_starts.append(start)
        # Prefer explicit target column if present; fallback to target_col_emb_index
        if target_col in row and not pd.isna(row[target_col]):
            labels.append(str(row[target_col]))
        else:
            labels.append(str(row.get(target_col, row.get(f"{target_col}_emb_index"))))
    if debug:
        print(f"[debug] load_force_segments: total_rows={len(df)} usable={len(segments)} short_skipped={skipped_short} nonfinite_skipped={skipped_nonfinite}")
    if len(segments)==0:
        raise RuntimeError("No usable segments after filtering.")
    return np.stack(segments), labels, meta_csv_paths, meta_starts   # (N, seq_len, nvars), labels, metadata

def build_patchtst_model(nvars, args):
    num_patch = (max(args.context_points, args.patch_len)-args.patch_len)//args.stride + 1
    model = PatchTST(
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
        dropout=0.0,
        head_dropout=0.0,
        act="gelu",
        head_type="pretrain",
        res_attention=False
    )
    return model


def infer_cnn_config_from_state(state, args):
    # Prefer checkpoint metadata if available; fallback to state_dict shape inference.
    base_channels = state.get("cnn_base_channels", None)
    depth = state.get("cnn_depth", None)
    if base_channels is None and "stem.0.weight" in state:
        base_channels = int(state["stem.0.weight"].shape[0])
    if depth is None:
        enc_ids = []
        for k in state.keys():
            if k.startswith("encoder.") and ".conv1.weight" in k:
                try:
                    enc_ids.append(int(k.split(".")[1]))
                except Exception:
                    pass
        depth = (max(enc_ids) + 1) if len(enc_ids) > 0 else args.cnn_depth

    out_embed_dim = args.out_embed_dim
    if "proj.weight" in state:
        out_embed_dim = int(state["proj.weight"].shape[0])
    elif "proj.3.weight" in state:
        out_embed_dim = int(state["proj.3.weight"].shape[0])
    elif "proj.0.weight" in state and "proj.3.weight" not in state:
        out_embed_dim = int(state["proj.0.weight"].shape[0])

    if base_channels is None:
        base_channels = args.cnn_base_channels
    return int(base_channels), int(depth), int(out_embed_dim)


def build_cnn_model(nvars, args, state=None):
    if state is not None:
        base_channels, depth, out_embed_dim = infer_cnn_config_from_state(state, args)
    else:
        base_channels, depth, out_embed_dim = args.cnn_base_channels, args.cnn_depth, args.out_embed_dim
    model = CNNContrastive(
        nvars=nvars,
        base_channels=base_channels,
        depth=depth,
        kernel_size=args.cnn_kernel_size,
        dropout=args.cnn_dropout,
        out_embed_dim=out_embed_dim,
        head_dropout=args.head_dropout,
    )
    return model, base_channels, depth, out_embed_dim


def infer_mlp_config_from_state(state, args):
    hidden_dim = state.get("mlp_hidden_dim", None)
    depth = state.get("mlp_depth", None)
    if hidden_dim is None and "stem.0.weight" in state:
        hidden_dim = int(state["stem.0.weight"].shape[0])
    if depth is None:
        enc_ids = []
        for k in state.keys():
            if k.startswith("encoder.") and ".net.0.weight" in k:
                try:
                    enc_ids.append(int(k.split(".")[1]))
                except Exception:
                    pass
        depth = (max(enc_ids) + 1) if len(enc_ids) > 0 else args.mlp_depth

    out_embed_dim = args.out_embed_dim
    if "proj.weight" in state:
        out_embed_dim = int(state["proj.weight"].shape[0])
    elif "proj.3.weight" in state:
        out_embed_dim = int(state["proj.3.weight"].shape[0])
    elif "proj.0.weight" in state and "proj.3.weight" not in state:
        out_embed_dim = int(state["proj.0.weight"].shape[0])

    if hidden_dim is None:
        hidden_dim = args.mlp_hidden_dim
    return int(hidden_dim), int(depth), int(out_embed_dim)


def build_mlp_model(nvars, args, state=None):
    if state is not None:
        hidden_dim, depth, out_embed_dim = infer_mlp_config_from_state(state, args)
    else:
        hidden_dim, depth, out_embed_dim = args.mlp_hidden_dim, args.mlp_depth, args.out_embed_dim
    model = MLPContrastive(
        nvars=nvars,
        seq_len=args.context_points,
        hidden_dim=hidden_dim,
        depth=depth,
        dropout=args.mlp_dropout,
        out_embed_dim=out_embed_dim,
        head_dropout=args.head_dropout,
    )
    return model, hidden_dim, depth, out_embed_dim


def extract_embeddings_patchtst(model, force_array, patch_len, stride, pool="mean", proj=None):
    """
    force_array: (N, seq_len, nvars)
    output: (N, d_model)
    """
    device = next(model.parameters()).device
    model.eval()
    embs = []
    with torch.no_grad():
        for batch in torch.split(torch.from_numpy(force_array), 64):  # batch size 64
            # batch: [B, seq_len, nvars] -> [B, num_patch, nvars, patch_len]
            xb_patch, _ = create_patch(batch.to(device), patch_len, stride)
            # backbone 出力: [B, nvars, d_model, num_patch]
            z = model.backbone(xb_patch)
            if pool == "mean":
                # nvars & num_patch 平均 -> [B, d_model]
                z_pool = z.mean(dim=(1,3))
            elif pool == "patchcat_mean":
                # パッチ方向に連結 -> [B, nvars, num_patch*d_model]、その後チャンネル平均 -> [B, num_patch*d_model]
                z_perm = z.permute(0, 1, 3, 2).contiguous()  # [B, nvars, num_patch, d_model]
                z_pool = z_perm.view(z_perm.size(0), z_perm.size(1), -1).mean(dim=1)
            elif pool == "flatten":
                z_pool = z.permute(0,1,3,2).contiguous().view(z.size(0), -1)
            else:
                # 末尾パッチ平均のみ
                z_pool = z[:,:,:, -1].mean(dim=1)  # [B, d_model]
            if proj is not None:
                z_pool = proj(z_pool)
            # L2 正規化（contrastive 空間に揃える）
            z_pool = z_pool / (z_pool.norm(dim=-1, keepdim=True) + 1e-8)
            embs.append(z_pool.cpu())
    return torch.cat(embs, dim=0).numpy()


def extract_embeddings_cnn(model, force_array):
    device = next(model.parameters()).device
    model.eval()
    embs = []
    with torch.no_grad():
        for batch in torch.split(torch.from_numpy(force_array), 64):
            z = model(batch.to(device).float())
            z = z / (z.norm(dim=-1, keepdim=True) + 1e-8)
            embs.append(z.cpu())
    return torch.cat(embs, dim=0).numpy()


def extract_embeddings_mlp(model, force_array):
    device = next(model.parameters()).device
    model.eval()
    embs = []
    with torch.no_grad():
        for batch in torch.split(torch.from_numpy(force_array), 64):
            z = model(batch.to(device).float())
            z = z / (z.norm(dim=-1, keepdim=True) + 1e-8)
            embs.append(z.cpu())
    return torch.cat(embs, dim=0).numpy()

def plot_umap_dual(force_emb, force_labels, out_path, text_emb=None, text_labels=None, ckpt_type="pretrained", debug=False):
    # Merge labels for unified mapping
    all_labels = list(force_labels)
    if text_labels is not None:
        all_labels += list(text_labels)
    unique = sorted(set(all_labels))
    mapping = {l:i for i,l in enumerate(unique)}
    t2d = None
    reducer = umap.UMAP(random_state=42)

    """Fit UMAP on force embeddings only, and then map text embeddings over it: looks good"""
    f2d = reducer.fit_transform(force_emb)  # (N, 2)
    if text_emb is not None:
        t2d = reducer.transform(text_emb)    # (N, 2)
    """textbase: doesn't work except for annotations"""
    # t2d = reducer.fit_transform(text_emb)
    # f2d = reducer.transform(force_emb)  # (N, 2)
    """create a joint embedding space by fitting on both force and text: looks awful"""
    # if ckpt_type == "contrastive" and text_emb is not None:
    #     combined = np.concatenate([force_emb, text_emb], axis=0)
    #     combined_2d = reducer.fit_transform(combined)
    #     f2d = combined_2d[:len(force_emb)]
    #     t2d = combined_2d[len(force_emb):]
    # else:
    #     f2d = reducer.fit_transform(force_emb)
    #     t2d = None
    #     if ckpt_type == "contrastive" and text_emb is not None:
    #         # Fallback transform (should rarely happen now) – but keeps previous behavior if condition mismatched
    #         t2d = reducer.transform(text_emb)

    plt.figure(figsize=(5,5))
    # Force points
    f_colors = [mapping[l] for l in force_labels]
    sc = plt.scatter(f2d[:,0], f2d[:,1], c=f_colors, cmap="tab10", s=5, alpha=0.8, label="force")
    # Text points (X marker)
    if t2d is not None:
        t_colors = [mapping[l] for l in text_labels]
    # Outer black outline (slightly larger)
        plt.scatter(t2d[:,0], t2d[:,1], c='black', s=100, marker='X', linewidths=5.0, alpha=0.95, zorder=3)
    # Inner colored marker (overlay slightly smaller)
        plt.scatter(t2d[:,0], t2d[:,1], c=t_colors, cmap="tab10", s=80, marker='X', linewidths=1.2, alpha=0.95, zorder=4, label="text")
    import matplotlib.patches as mpatches
    handles = [mpatches.Patch(color=sc.cmap(sc.norm(mapping[lbl])), label=lbl) for lbl in unique]
    plt.legend(handles=handles, bbox_to_anchor=(1.04,1), loc="upper left", fontsize=12)
    plt.xticks([]); plt.yticks([])
    for spine in ['top','right','left','bottom']:
        plt.gca().spines[spine].set_visible(False)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    if debug:
        print(f"[debug] UMAP(force fit on {'force+text' if (ckpt_type=='contrastive' and text_emb is not None) else 'force only'}) force_emb shape={force_emb.shape} text_emb shape={None if text_emb is None else text_emb.shape}")
    print("Saved:", out_path)

def load_label_text_embeddings(eval_dir, emb_dir="text_emb", debug=False, target_col="label"):
    csv_name=f"{target_col}_index_map.csv"
    emb_dir=f"{emb_dir}/{target_col}"
    path = os.path.join(eval_dir, csv_name)
    if not os.path.exists(path):
        return None, None
    df = pd.read_csv(path)
    if not {target_col,"emb_index"}.issubset(df.columns):
        return None, None
    labels = []
    vecs = []
    skipped_nonfinite = 0
    for _, row in df.iterrows():
        p = os.path.join(eval_dir, emb_dir, str(row["emb_index"]) + ".pt")
        try:
            t = torch.load(p, map_location='cpu')
        except Exception:
            continue
        if isinstance(t, torch.Tensor):
            if t.ndim == 2 and t.shape[0] == 1:
                t = t.squeeze(0)
            t = t.float()
            if not torch.isfinite(t).all():
                skipped_nonfinite += 1
                continue
            vecs.append(t.numpy())
            labels.append(str(row[target_col]))
    if skipped_nonfinite > 0 and debug:
        print(f"[debug] skipped {skipped_nonfinite} non-finite text embeddings in {emb_dir}")
    return np.stack(vecs), labels

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, required=True,
                    help="pretrained PatchTST .pth (pretrain head)")
    ap.add_argument("--ckpt_type", type=str, required=True, choices=["pretrained","contrastive"])
    ap.add_argument("--model_arch", type=str, default="patchtst", choices=["patchtst", "cnn", "mlp"],
                    help="model architecture to evaluate")

    ap.add_argument("--eval_dir", type=str, default="/home/user/Genesis/data/eval_04272026/")
    ap.add_argument("--eval_csv", type=str, default="eval_04272026_thin_15pct.csv")
    ap.add_argument("--context_points", type=int, default=80)
    ap.add_argument("--target_points", type=int, default=96)
    ap.add_argument("--patch_len", type=int, default=10)
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--n_layers", type=int, default=3)
    ap.add_argument("--n_heads", type=int, default=16)
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--d_ff", type=int, default=768)
    ap.add_argument("--pool", type=str, default="mean", choices=["mean","flatten","last","patchcat_mean"])
    ap.add_argument("--head_dropout", type=float, default=0.2)
    # CNN args (default to patchtst_cnn.py defaults)
    ap.add_argument("--cnn_base_channels", type=int, default=128)
    ap.add_argument("--cnn_depth", type=int, default=6)
    ap.add_argument("--cnn_kernel_size", type=int, default=5)
    ap.add_argument("--cnn_dropout", type=float, default=0.2)
    # MLP args (default to patchtst_mlp.py defaults)
    ap.add_argument("--mlp_hidden_dim", type=int, default=256)
    ap.add_argument("--mlp_depth", type=int, default=6)
    ap.add_argument("--mlp_dropout", type=float, default=0.2)
    ap.add_argument("--out_embed_dim", type=int, default=768)
    ap.add_argument("--out", type=str, default="patchtst_umap.png")
    ap.add_argument("--label_index_map", type=str, default="label_index_map.csv")
    ap.add_argument("--text_type", type=str, default="bert", choices=["siglip","bert","clip"], help="text embedding type")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()
    targets = ["label", "action", "weight", "interaction"]
    # targets = ["annotation"]
    model_name = os.path.basename(args.model_path).replace('.pth','')
    model_id = model_name.split('_')[0]
    os.makedirs(f"data/{model_id}/{model_name}/umap", exist_ok=True)
    if args.ckpt_type == 'contrastive':
        os.makedirs(f"data/{model_id}/{model_name}/heatmap", exist_ok=True)
        os.makedirs(f"data/{model_id}/{model_name}/pred", exist_ok=True)
    # ターゲット別メトリクス格納
    metrics_by_target = {}

    for target in targets:
        umap_out = f"data/{model_id}/{model_name}/umap/{target}_umap.png"
        heatmap_out = f"data/{model_id}/{model_name}/heatmap/{target}_heatmap.png"
        pred_out = f"data/{model_id}/{model_name}/pred/{target}_pred.csv"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load force segments
        force_segments, labels, csv_paths, starts = load_force_segments(
            args.eval_dir, args.eval_csv,
            seq_len=args.context_points,
            target_col=target,
            drop_nonfinite=True,
            debug=args.debug
        )
        nvars = force_segments.shape[2]

        # Build model & load checkpoint / restore proj for contrastive
        proj = None
        if args.model_arch == "patchtst":
            model = build_patchtst_model(nvars, args).to(device)
        else:
            model = None

        if args.ckpt_type == "pretrained":
            if args.model_arch == "patchtst":
                load_model(args.model_path, model, device=device, with_opt=False)
                print("[info] loaded pretrained checkpoint fully")
            else:
                print("[warn] ckpt_type=pretrained with model_arch in {cnn,mlp} is not supported for embedding eval; use contrastive ckpt.")
                continue
        else:  # contrastive
            ckpt_all = torch.load(args.model_path, map_location=device)
            state = ckpt_all.get('model', ckpt_all)
            if args.model_arch == "patchtst":
                # Infer projection structure from saved weights (robust to pooling changes)
                if any(k.startswith('proj.') for k in state.keys()):
                    if 'proj.3.weight' in state and 'proj.0.weight' in state:
                        in_dim = state['proj.0.weight'].shape[1]
                        hidden = state['proj.0.weight'].shape[0]
                        out_dim = state['proj.3.weight'].shape[0]
                        proj = torch.nn.Sequential(
                            torch.nn.Linear(in_dim, hidden),
                            torch.nn.GELU(),
                            torch.nn.Dropout(0.0),
                            torch.nn.Linear(hidden, out_dim)
                        ).to(device)
                        proj.eval()
                    elif 'proj.0.weight' in state:
                        in_dim = state['proj.0.weight'].shape[1]
                        out_dim = state['proj.0.weight'].shape[0]
                        proj = torch.nn.Sequential(
                            torch.nn.Linear(in_dim, out_dim)
                        ).to(device)
                # Load backbone only
                cur = model.state_dict()
                loaded_b = 0
                for k,v in state.items():
                    if k.startswith('backbone.') and k in cur and cur[k].shape == v.shape:
                        cur[k] = v
                        loaded_b += 1
                model.load_state_dict(cur, strict=False)
                # Load projection
                if proj is not None:
                    p_state = proj.state_dict()
                    loaded_p = 0
                    for k,v in state.items():
                        if k.startswith('proj.'):
                            k2 = k.replace('proj.','')
                            if k2 in p_state and p_state[k2].shape == v.shape:
                                p_state[k2] = v
                                loaded_p += 1
                    proj.load_state_dict(p_state, strict=False)
                    print(f"[info] loaded contrastive backbone:{loaded_b} proj:{loaded_p} (out_dim={out_dim})")
                else:
                    print(f"[warn] contrastive proj not found; using backbone only (loaded {loaded_b} tensors)")
            elif args.model_arch == "cnn":
                model, bw, dp, od = build_cnn_model(nvars, args, state=state)
                model = model.to(device)
                msg = model.load_state_dict(state, strict=False)
                print(
                    f"[info] loaded cnn contrastive ckpt "
                    f"(base_channels={bw}, depth={dp}, out_embed_dim={od}) "
                    f"missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)}"
                )
            else:
                model, hw, dp, od = build_mlp_model(nvars, args, state=state)
                model = model.to(device)
                msg = model.load_state_dict(state, strict=False)
                print(
                    f"[info] loaded mlp contrastive ckpt "
                    f"(hidden_dim={hw}, depth={dp}, out_embed_dim={od}) "
                    f"missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)}"
                )
        model.eval()

        # Extract embeddings
        if args.model_arch == "patchtst":
            embeddings = extract_embeddings_patchtst(model, force_segments, args.patch_len,
                                                     args.stride, pool=args.pool, proj=proj)
        elif args.model_arch == "cnn":
            embeddings = extract_embeddings_cnn(model, force_segments)
        else:
            embeddings = extract_embeddings_mlp(model, force_segments)

        # Load text embeddings with labels when contrastive
        text_emb = text_labels = None
        base_dir_map = {"bert": "bert_emb", "clip": "clip_emb", "siglip": "text_emb"}
        emb_base = base_dir_map[args.text_type]
        if args.ckpt_type == 'contrastive':
            print("contrastive ckpt; loading text embeddings")
            text_emb, text_labels = load_label_text_embeddings(args.eval_dir, emb_dir=emb_base, target_col=target, debug=args.debug)
            if text_emb is not None and args.debug:
                print(f"[debug] loaded text_emb shape={text_emb.shape} labels={len(text_labels)} from {emb_base}")

        # UMAP plot 1: all (text embeddings with X)
        plot_umap_dual(embeddings, labels, umap_out, text_emb, text_labels, ckpt_type=args.ckpt_type, debug=args.debug)
        if args.ckpt_type == 'pretrained':
            continue
        # Minimal confusion matrix (contrastive + prototypes available)
        if args.ckpt_type == 'contrastive' and text_emb is not None and len(text_labels) > 0:

            # embeddings already L2-normalized in extract_embeddings
            emb_t = torch.from_numpy(embeddings).float()
            prot = torch.from_numpy(text_emb).float()
            prot = prot / (prot.norm(dim=-1, keepdim=True) + 1e-8)
            sims = emb_t @ prot.t()
            pred_idx = sims.argmax(dim=1).cpu().numpy()
            preds = [text_labels[i] for i in pred_idx]
            y_true = labels
            # Order: sorted unique labels from data (deterministic)
            sel_order = sorted(set(y_true))
            cm = confusion_matrix(y_true, preds, labels=sel_order)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=sel_order, yticklabels=sel_order, vmax=max(cm.max(), 1))
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            plt.tight_layout()
            plt.savefig(heatmap_out)
            plt.close()
            acc = (cm.trace() / cm.sum()) if cm.sum() > 0 else 0.0
            # 追加メトリクス
            micro_precision = precision_score(y_true, preds, average='micro', zero_division=0)
            micro_recall    = recall_score(y_true, preds, average='micro', zero_division=0)
            micro_f1        = f1_score(y_true, preds, average='micro', zero_division=0)
            macro_precision = precision_score(y_true, preds, average='macro', zero_division=0)
            macro_recall    = recall_score(y_true, preds, average='macro', zero_division=0)
            macro_f1        = f1_score(y_true, preds, average='macro', zero_division=0)
            metrics_by_target[target] = {
                "Accuracy": acc,
                "Micro Precision": micro_precision,
                "Micro Recall": micro_recall,
                "Micro F1": micro_f1,
                "Macro Precision": macro_precision,
                "Macro Recall": macro_recall,
                "Macro F1": macro_f1
            }
            print(f"[info] UMAP confusion matrix saved to {heatmap_out} (acc={acc:.4f}, samples={len(y_true)})")
            # Export per-sample predictions with cosine similarities
            export_len = min(len(y_true), len(csv_paths))
            sims_np = sims.cpu().numpy()
            # Prepare label order file (order corresponds to columns of sims/prototypes)
            order_txt = os.path.join(os.path.dirname(pred_out), f"{target}_pred_label_order.txt")
            with open(order_txt, 'w', encoding='utf-8') as f:
                for j, lbl in enumerate(text_labels):
                    f.write(f"{j}\t{lbl}\n")
            # Build rows
            rows = []
            for i in range(export_len):
                row = {
                    "csv_path": csv_paths[i],
                    "start": int(starts[i]),
                    "true": y_true[i],
                    "pred": preds[i]
                }
                # add cosine similarities per label as sim_0, sim_1, ... aligned with text_labels order
                for j in range(len(text_labels)):
                    row[f"sim_{j}"] = float(sims_np[i, j])
                rows.append(row)
            pd.DataFrame(rows).to_csv(pred_out, index=False)
            print(f"[info] Saved per-sample predictions to {pred_out} ({export_len} rows)")
            print(f"[info] Saved label order mapping to {order_txt}")
        else:
            print("[info] Confusion matrix skipped (need contrastive + text embeddings).")

    
    if args.ckpt_type == "contrastive" and len(metrics_by_target) > 0:
        all_metrics = [
            "Accuracy",
            "Micro Precision","Micro Recall","Micro F1",
            "Macro Precision","Macro Recall","Macro F1"
        ]
        header = ["Metric"] + targets
        lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
        for m in all_metrics:
            row = [m]
            for t in targets:
                if t in metrics_by_target:
                    row.append(f"{metrics_by_target[t][m]:.4f}")
                else:
                    row.append("-")
            lines.append("| " + " | ".join(row) + " |")
        out_md = f"data/{model_id}/{model_name}/umap_metrics_summary.md"
        with open(out_md, 'w', encoding='utf-8') as f:
            f.write("# UMAP Prototype Classification Metrics Summary\n\n")
            f.write("\n".join(lines) + "\n")
        print(f"Saved aggregated UMAP metrics markdown -> {out_md}")

if __name__ == "__main__":
    main()
