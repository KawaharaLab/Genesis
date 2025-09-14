import pandas as pd
import torch
from transformers import AutoProcessor, AutoModel
from tqdm import tqdm
import os

"""
label だけでなく action / weight / interaction も
  1) NaN / 空文字は除外
  2) ユニーク文字列ごとに1個だけテキスト埋め込みを生成
  3) 元 DataFrame に <col>_emb_index を付与 (欠損は NaN のまま)
埋め込みは save_dir/<col>/<index>.pt に保存
対応表は save_dir/<col>_index_map.csv
"""
data_dir = "data/eval_heavy"
save_dir = "data/eval_heavy/text_emb"
os.makedirs(save_dir, exist_ok=True)

model_name = "google/siglip-base-patch16-224"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float32).to(device)
processor = AutoProcessor.from_pretrained(model_name)
model.eval()

csv_path = "data/eval_heavy/eval_heavy_thin_15pct.csv"
df = pd.read_csv(csv_path)

target_cols = ["label", "action", "weight", "interaction"]
batch_size = 64

def build_unique_list(series: pd.Series) -> list[str]:
    # NaN 除去 -> 文字列化 -> 前後空白除去 -> 空文字除去 -> ユニーク
    s = series.dropna().astype(str).str.strip()
    s = s[s != ""]
    return sorted(s.unique())

def embed_text_list(text_list: list[str]) -> list[torch.Tensor]:
    embeddings: list[torch.Tensor] = []
    for start in tqdm(range(0, len(text_list), batch_size), desc="embed", leave=False):
        batch = text_list[start:start + batch_size]
        inputs = processor(text=batch, padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            feats = model.get_text_features(**inputs)
        feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
        feats_cpu = feats.cpu()
        embeddings.extend(list(feats_cpu))
    return embeddings

for col in target_cols:
    if col not in df.columns:
        print(f"[WARN] column '{col}' not found. skip.")
        continue

    uniq_texts = build_unique_list(df[col])
    print(f"[INFO] {col}: unique valid texts = {len(uniq_texts)}")

    if len(uniq_texts) == 0:
        df[f"{col}_emb_index"] = pd.NA
        continue

    col_dir = os.path.join(save_dir, col)
    os.makedirs(col_dir, exist_ok=True)

    # 埋め込み生成
    emb_list = embed_text_list(uniq_texts)

    # 保存 & マッピング
    text_to_index = {}
    for idx, vec in enumerate(emb_list):
        torch.save(vec, os.path.join(col_dir, f"{idx}.pt"))
        text_to_index[uniq_texts[idx]] = idx

    # DataFrame へ index 付与（欠損/空は NaN のまま）
    def map_func(v):
        if pd.isna(v):
            return pd.NA
        v_str = str(v).strip()
        if v_str == "":
            return pd.NA
        return text_to_index.get(v_str, pd.NA)

    df[f"{col}_emb_index"] = df[col].map(map_func)

    # 対応表保存
    mapping_path = os.path.join(data_dir, f"{col}_index_map.csv")
    pd.DataFrame({col: uniq_texts, "emb_index": [text_to_index[t] for t in uniq_texts]}).to_csv(mapping_path, index=False)
    print(f"[INFO] {col}: saved {len(emb_list)} embeddings -> {col_dir}, mapping -> {mapping_path}")

# CSV 更新
df.to_csv(csv_path, index=False)
print(f"[DONE] updated {csv_path} with *_emb_index columns")