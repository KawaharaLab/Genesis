# ...existing code...
import os
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

save_dir = "data/train_04072026/bert_emb"
os.makedirs(save_dir, exist_ok=True)

model_name = "sentence-transformers/all-mpnet-base-v2"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SentenceTransformer(model_name, device=device)
model.max_seq_length = 512

csv_path = "data/train_04072026/train_04072026_thin_20pct.csv"
df = pd.read_csv(csv_path)

# Collect unique annotations (ignore NaN / empty)
series = df.get("annotation", pd.Series([], dtype=str)).fillna("").astype(str).str.strip()
series = series[series != ""]
unique_ann = sorted(series.unique())
print(f"unique annotations: {len(unique_ann)} / rows: {len(df)}")

batch_size = 64
embeddings_list: list[torch.Tensor] = []

def encode_texts(texts: list[str]) -> list[torch.Tensor]:
    out: list[torch.Tensor] = []
    for start in tqdm(range(0, len(texts), batch_size), desc="encode(unique)", leave=False):
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

if len(unique_ann) == 0:
    print("No annotations found. Skipping embedding generation.")
    df["emb_index"] = pd.NA
else:
    embeddings_list = encode_texts(unique_ann)
    if len(embeddings_list) != len(unique_ann):
        raise RuntimeError("embedding count mismatch")

    # Save each unique embedding once; file name = emb_index.pt (0..U-1)
    for idx, vec in enumerate(embeddings_list):
        torch.save(vec, os.path.join(save_dir, f"{idx}.pt"))

    # Map annotation -> index
    ann_to_index = {ann: i for i, ann in enumerate(unique_ann)}

    def map_index(v):
        if pd.isna(v):
            return pd.NA
        v_str = str(v).strip()
        if v_str == "":
            return pd.NA
        return ann_to_index.get(v_str, pd.NA)

    df["emb_index"] = df.get("annotation", pd.Series([], dtype=str)).map(map_index)

    # Write mapping file (annotation -> emb_index)
    mapping_path = os.path.join(os.path.dirname(csv_path), "annotation_index_map.csv")
    pd.DataFrame({"annotation": unique_ann, "emb_index": [ann_to_index[a] for a in unique_ann]}).to_csv(mapping_path, index=False)
    print(f"wrote mapping -> {mapping_path}")

df.to_csv(csv_path, index=False)
print(f"[DONE] updated {csv_path} with emb_index (unique annotation embeddings in {save_dir})")
# ...existing