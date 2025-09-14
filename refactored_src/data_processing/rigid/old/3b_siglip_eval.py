import pandas as pd
import torch
from transformers import AutoProcessor, AutoModel
from tqdm import tqdm
import os
import numpy as np

"""
本スクリプトは行ごとではなくユニークな label ごとに 1 個だけテキスト埋め込み(.pt)を生成し、
各行には label -> emb_index のマッピングを付与します。

出力:
  data/eval/text_emb/{emb_index}.pt  (ユニークラベル数分)
  data/eval/eval.csv (emb_index列を追加/更新)
"""

save_dir = "data/eval/text_emb"
os.makedirs(save_dir, exist_ok=True)

model_name = "google/siglip-base-patch16-224"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float32).to(device)
processor = AutoProcessor.from_pretrained(model_name)
model.eval()

csv_path = "data/eval/eval.csv"
train_df = pd.read_csv(csv_path)

unique_labels = sorted(train_df["label"].unique())
print(f"Unique labels: {len(unique_labels)} / Rows: {len(train_df)}")

# 既存の emb_index があってユニークラベル数と .pt が一致していれば再計算をスキップするロジックを入れても良いが、
# とりあえず常に再生成（必要なら条件付きスキップを追加）。

batch_size = 64
label_to_index: dict[str, int] = {}
embeddings: list[torch.Tensor] = []

for start in tqdm(range(0, len(unique_labels), batch_size), desc="embed unique labels"):
    batch_labels = unique_labels[start:start + batch_size]
    inputs = processor(text=batch_labels, padding=True, return_tensors="pt").to(device)
    with torch.no_grad():
        feats = model.get_text_features(**inputs)
    feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
    feats_cpu = feats.cpu()
    for label, vec in zip(batch_labels, feats_cpu):
        emb_index = len(embeddings)
        embeddings.append(vec)
        label_to_index[label] = emb_index

# 保存 (index順に {i}.pt)
for i, vec in enumerate(embeddings):
    torch.save(vec, os.path.join(save_dir, f"{i}.pt"))

# DataFrame に emb_index 列を付与
train_df["emb_index"] = train_df["label"].map(label_to_index).astype(int)
train_df.to_csv(csv_path, index=False)
print(f"Saved {len(embeddings)} embeddings and updated {csv_path}")

# ラベル->index の対応表を別ファイルにも書き出し（解析用）
mapping_path = os.path.join("data/eval", "label_index_map.csv")
pd.DataFrame({"label": unique_labels, "emb_index": [label_to_index[l] for l in unique_labels]}).to_csv(mapping_path, index=False)
print(f"Label-index map saved to {mapping_path}")