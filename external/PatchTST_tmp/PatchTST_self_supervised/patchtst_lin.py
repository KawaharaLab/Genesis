import argparse
import os
import json
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import wandb

from src.models.patchTST import PatchTST
from src.callback.patch_mask import create_patch

DEFAULT_FORCE_COLS = [
	"left_fx","left_fy","left_fz","left_tx","left_ty","left_tz",
	"right_fx","right_fy","right_fz","right_tx","right_ty","right_tz"
]

class ForceLabelDataset(Dataset):
	"""Linear classifier 用データセット (force_seq, label_idx) + メタ情報保持"""
	def __init__(self, data_dir, index_csv, seq_len=80, use_cols=None, label_map=None, allowed_labels=None):
		self.data_dir = data_dir
		self.index_csv = os.path.join(data_dir, index_csv)
		self.seq_len = seq_len
		self.use_cols = use_cols or DEFAULT_FORCE_COLS
		if not os.path.exists(self.index_csv):
			raise FileNotFoundError(self.index_csv)
		df = pd.read_csv(self.index_csv)
		if allowed_labels is not None:
			allowed_set = set([str(x) for x in allowed_labels])
			df = df[df['label'].astype(str).isin(allowed_set)]
		required = {"csv_path", "start", "label"}
		miss = required - set(df.columns)
		if miss:
			raise ValueError(f"Missing columns {miss} in {self.index_csv}")
		self.force_segments, self.labels = [], []
		self.sample_csv_paths, self.sample_starts = [], []
		cache = {}
		for p in df['csv_path'].unique():
			full_p = os.path.join(data_dir, 'csv', p)
			if os.path.exists(full_p):
				cache[p] = pd.read_csv(full_p, usecols=self.use_cols).values.astype('float32')
		for _, row in df.iterrows():
			p = row['csv_path']
			if p not in cache:
				continue
			start = int(row['start'])
			seg = cache[p][start:start+self.seq_len, :]
			if seg.shape[0] != self.seq_len:
				continue
			label = str(row['label'])
			self.force_segments.append(seg)
			self.labels.append(label)
			self.sample_csv_paths.append(p)
			self.sample_starts.append(start)
		if len(self.force_segments) == 0:
			raise RuntimeError(f"No usable pairs in {data_dir}")
		if label_map is None:
			if allowed_labels is not None:
				present = [l for l in allowed_labels if l in set(self.labels)]
				self.label_map = {l: i for i, l in enumerate(present)}
			else:
				self.label_map = {l: i for i, l in enumerate(sorted(set(self.labels)))}
		else:
			self.label_map = label_map
		self.label_indices = [self.label_map[l] for l in self.labels]

	def __len__(self):
		return len(self.force_segments)

	def __getitem__(self, idx):
		return torch.from_numpy(self.force_segments[idx]), self.label_indices[idx]

def build_patchtst_proj(nvars, args, device, state):
	"""PatchTST + 2-layer projection head を構築 (pretrained / contrastive ckpt 想定)。"""
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
		dropout=args.dropout,
		head_dropout=args.head_dropout,
		act="gelu",
		head_type="pretrain",
		res_attention=False
	).to(device)
	if 'proj.3.weight' not in state:
		raise KeyError("Expected 'proj.3.weight' in checkpoint for projection head")
	out_dim = state['proj.3.weight'].shape[0]
	proj = nn.Sequential(
		nn.Linear(args.d_model, args.d_model),
		nn.GELU(),
		nn.Dropout(args.head_dropout),
		nn.Linear(args.d_model, out_dim)
	).to(device)
	cur = model.state_dict()
	for k, v in state.items():
		if k.startswith('backbone.') and k in cur and cur[k].shape == v.shape:
			cur[k] = v
	model.load_state_dict(cur, strict=False)
	p_state = proj.state_dict()
	for k, v in state.items():
		if k.startswith('proj.'):
			k2 = k.replace('proj.', '')
			if k2 in p_state and p_state[k2].shape == v.shape:
				p_state[k2] = v
	proj.load_state_dict(p_state, strict=False)
	return model, proj

def main():
	parser = argparse.ArgumentParser(description="Linear classifier fine-tuning on frozen PatchTST embeddings")
	parser.add_argument("--model_path", type=str, required=True)
	parser.add_argument('--selected_labels', type=str, default="stable,slip", help='Comma-separated label subset (order preserved)')
	parser.add_argument("--train_dir", type=str, default="/home/user/Genesis/data/train_old/")
	parser.add_argument("--eval_dir", type=str, default="/home/user/Genesis/data/eval_heavy/")
	parser.add_argument("--train_csv", type=str, default="train_old_vision_wide.csv")
	parser.add_argument("--eval_csv", type=str, default="eval_heavy_vision_wide.csv")
	parser.add_argument("--context_points", type=int, default=80)
	parser.add_argument("--target_points", type=int, default=0)
	parser.add_argument("--patch_len", type=int, default=10)
	parser.add_argument("--stride", type=int, default=10)
	parser.add_argument("--n_layers", type=int, default=3)
	parser.add_argument("--n_heads", type=int, default=16)
	parser.add_argument("--d_model", type=int, default=128)
	parser.add_argument("--d_ff", type=int, default=768)
	parser.add_argument("--dropout", type=float, default=0.0)
	parser.add_argument("--head_dropout", type=float, default=0.0)
	parser.add_argument("--pool", type=str, default="mean", choices=["mean", "flatten", "last"])
	parser.add_argument("--batch_size", type=int, default=256)
	parser.add_argument("--epochs", type=int, default=5)
	parser.add_argument("--lr", type=float, default=1e-3)
	parser.add_argument("--weight_decay", type=float, default=0.0)
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--wandb_project", type=str, default="patchtst_lin")
	parser.add_argument("--no_wandb", action="store_true", help="Disable wandb logging")
	args = parser.parse_args()
	model_name = os.path.basename(args.model_path).replace(".pth", "").replace(".pt", "")
	model_id = model_name.split('_')[0]
	out_dir = f"data/{model_id}/{model_name}/lin"
	os.makedirs(out_dir, exist_ok=True)
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	if not args.no_wandb:
		try:
			wandb.login(key="c85b817c62f441243d232b381088358e72fa2b19")
			wandb.init(project=args.wandb_project, config=vars(args))
		except Exception as e:
			print(f"[warn] wandb init failed: {e}")

	ckpt = torch.load(args.model_path, map_location=device)
	state = ckpt.get('model', ckpt)

	sel = None
	if args.selected_labels:
		sel = [s.strip() for s in args.selected_labels.split(',') if s.strip()]
	train_ds = ForceLabelDataset(args.train_dir, args.train_csv, args.context_points, DEFAULT_FORCE_COLS, label_map=None, allowed_labels=sel)
	label_map = train_ds.label_map
	idx_to_label = {v: k for k, v in label_map.items()}
	eval_ds = ForceLabelDataset(args.eval_dir, args.eval_csv, args.context_points, DEFAULT_FORCE_COLS, label_map=label_map, allowed_labels=sel)
	nvars = train_ds[0][0].shape[1]
	n_classes = len(label_map)
	train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
	eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

	patchtst, proj = build_patchtst_proj(nvars, args, device, state)
	patchtst.eval()
	for p in patchtst.parameters():
		p.requires_grad = False
	proj.eval()
	for p in proj.parameters():
		p.requires_grad = False

	embed_dim = proj[-1].out_features if proj is not None else args.d_model
	classifier = nn.Linear(embed_dim, n_classes).to(device)
	optimizer = torch.optim.Adam(classifier.parameters(), lr=args.lr, weight_decay=args.weight_decay)
	criterion = nn.CrossEntropyLoss()

	def forward_embed(xb):
		with torch.no_grad():
			xb_patch, _ = create_patch(xb, args.patch_len, args.stride)
			z = patchtst.backbone(xb_patch)
			if args.pool == "mean":
				z_pool = z.mean(dim=(1, 3))
			elif args.pool == "flatten":
				z_pool = z.permute(0, 1, 3, 2).contiguous().view(z.size(0), -1)
			else:
				z_pool = z[:, :, :, -1].mean(dim=1)
			if proj is not None:
				z_pool = proj(z_pool)
			z_pool = z_pool / (z_pool.norm(dim=-1, keepdim=True) + 1e-8)
		return z_pool

	# 評価関数
	def evaluate():
		classifier.eval()
		all_true, all_pred = [], []
		with torch.no_grad():
			for xb, y in eval_loader:
				xb = xb.to(device)
				y = y.to(device)
				emb = forward_embed(xb)
				logits = classifier(emb)
				preds = logits.argmax(dim=1)
				all_true.append(y.cpu())
				all_pred.append(preds.cpu())
		if len(all_true) == 0:
			return 0.0, torch.tensor([]), torch.tensor([])
		y_true_t = torch.cat(all_true)
		y_pred_t = torch.cat(all_pred)
		acc_local = (y_true_t == y_pred_t).float().mean().item()
		return acc_local, y_true_t, y_pred_t

	best_acc = -1.0
	best_epoch = -1
	best_state = None
	best_y_true = None
	best_y_pred = None

	for epoch in range(args.epochs):
		classifier.train()
		total_loss = 0.0
		n_total = 0
		n_correct = 0
		for xb, y in train_loader:
			xb = xb.to(device)
			y = y.to(device)
			emb = forward_embed(xb)
			logits = classifier(emb)
			loss = criterion(logits, y)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			total_loss += loss.item() * xb.size(0)
			preds = logits.argmax(dim=1)
			n_correct += (preds == y).sum().item()
			n_total += xb.size(0)
		train_acc = n_correct / n_total
		avg_loss = total_loss / n_total
		if not args.no_wandb:
			wandb.log({"epoch": epoch+1, "train_loss": avg_loss, "train_acc": train_acc})

			eval_acc, y_true_t, y_pred_t = evaluate()
			improved = eval_acc > best_acc
			if improved:
				best_acc = eval_acc
				best_epoch = epoch + 1
				best_state = classifier.state_dict()
				best_y_true = y_true_t.clone()
				best_y_pred = y_pred_t.clone()
				torch.save({
					'epoch': best_epoch,
					'eval_acc': best_acc,
					'state_dict': best_state,
					'label_map': label_map
				}, f"{out_dir}/best_classifier.pth")
				with open(f"{out_dir}/metrics_best.json", 'w', encoding='utf-8') as f:
					json.dump({'best_epoch': best_epoch, 'best_eval_acc': best_acc}, f, ensure_ascii=False, indent=2)
				print(f"[best] epoch {best_epoch} new best eval_acc={best_acc:.4f} -> saved")
			print(f"Epoch {epoch+1}/{args.epochs} - loss: {avg_loss:.4f} train_acc: {train_acc:.4f} eval_acc: {eval_acc:.4f}{' *' if improved else ''}")
			if not args.no_wandb:
				wandb.log({"epoch": epoch+1, "train_loss": avg_loss, "train_acc": train_acc, "eval_acc": eval_acc, "best_eval_acc": best_acc})

	# ベストモデルで成果物生成
	if best_state is None:
		best_acc, _, best_y_true, best_y_pred = evaluate()
	else:
		classifier.load_state_dict(best_state)
	print(f"Final best eval accuracy: {best_acc:.4f} (epoch {best_epoch})")
	if not args.no_wandb:
		wandb.log({"final_best_eval_acc": best_acc, "final_best_epoch": best_epoch})

	y_true_t = best_y_true
	y_pred_t = best_y_pred
	y_true_labels = [idx_to_label[i.item()] for i in y_true_t]
	y_pred_labels = [idx_to_label[i.item()] for i in y_pred_t]
	if len(y_true_labels) != len(eval_ds.sample_csv_paths):
		print(f"[warn] Length mismatch: predictions {len(y_true_labels)} vs metadata {len(eval_ds.sample_csv_paths)}")
	export_len = min(len(y_true_labels), len(eval_ds.sample_csv_paths))
	recs = []
	for i in range(export_len):
		recs.append({
			"csv_path": eval_ds.sample_csv_paths[i],
			"start": int(eval_ds.sample_starts[i]),
			"true": y_true_labels[i],
			"pred": y_pred_labels[i]
		})
	pd.DataFrame(recs).to_csv(f"{out_dir}/predictions_linear.csv", index=False)
	print(f"Saved predictions_linear.csv ({export_len} rows)")

	if args.selected_labels:
		order = [s for s in sel if s in label_map]
	else:
		order = [idx_to_label[i] for i in range(len(idx_to_label))]
	cm = confusion_matrix(y_true_labels, y_pred_labels, labels=order)
	plt.figure(figsize=(10, 8))
	sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=order, yticklabels=order, vmax=max(cm.max(), 1))
	plt.ylabel('True label')
	plt.xlabel('Predicted label')
	plt.tight_layout()
	plt.savefig(f'{out_dir}/confusion_matrix_heatmap_linear.png')
	print("Saved confusion_matrix_heatmap_linear.png")

if __name__ == "__main__":
	main()
