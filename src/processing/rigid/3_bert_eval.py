import os
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

data_dir = "data/eval_04072026"
save_dir = "data/eval_04072026/bert_emb"
os.makedirs(save_dir, exist_ok=True)

model_name = "sentence-transformers/all-mpnet-base-v2"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SentenceTransformer(model_name, device=device)
model.max_seq_length = 512

csv_path = "data/eval_04072026/eval_04072026_thin_20pct.csv"
df = pd.read_csv(csv_path)

target_cols = ["label", "action", "weight", "interaction", "annotation"]
batch_size = 64

def collect_unique(series: pd.Series) -> list[str]:
    s = series.dropna().astype(str).str.strip()
    s = s[s != ""]
    return sorted(s.unique())

def encode_texts(texts: list[str]) -> list[torch.Tensor]:
    out: list[torch.Tensor] = []
    for start in tqdm(range(0, len(texts), batch_size), desc="encode", leave=False):
        batch = texts[start:start + batch_size]
        feats = model.encode(
            batch,
            batch_size=len(batch),
            convert_to_tensor=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        out.extend(list(feats.detach().cpu()))
    return out

for col in target_cols:
    if col not in df.columns:
        print(f"[WARN] column '{col}' not found. skip.")
        continue

    uniq = collect_unique(df[col])
    print(f"[INFO] {col}: unique valid entries = {len(uniq)}")
    if len(uniq) == 0:
        df[f"{col}_emb_index"] = pd.NA
        continue

    col_dir = os.path.join(save_dir, col)
    os.makedirs(col_dir, exist_ok=True)

    embeddings = encode_texts(uniq)
    if len(embeddings) != len(uniq):
        raise RuntimeError(f"embedding count mismatch for {col}")

    text_to_index = {}
    for idx, vec in enumerate(embeddings):
        torch.save(vec, os.path.join(col_dir, f"{idx}.pt"))
        text_to_index[uniq[idx]] = idx

    def map_index(v):
        if pd.isna(v):
            return pd.NA
        v_str = str(v).strip()
        if v_str == "":
            return pd.NA
        return text_to_index.get(v_str, pd.NA)

    df[f"{col}_emb_index"] = df[col].map(map_index)

    mapping_path = os.path.join(data_dir, f"{col}_index_map.csv")
    pd.DataFrame({col: uniq, "emb_index": [text_to_index[t] for t in uniq]}).to_csv(mapping_path, index=False)
    print(f"[INFO] {col}: saved {len(embeddings)} embeddings -> {col_dir}")

df.to_csv(csv_path, index=False)
print(f"[DONE] updated {csv_path} with *_emb_index columns")