import pandas as pd
import torch
from transformers import AutoProcessor, AutoModel
from tqdm import tqdm
import os
import numpy as np

save_dir = "data/train/text_emb"
os.makedirs(save_dir, exist_ok=True)

model_name = "google/siglip-base-patch16-224"
model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float32).cuda()
processor = AutoProcessor.from_pretrained(model_name)
model.eval()

train_df = pd.read_csv("data/train/train.csv")
annotations = train_df["label"].tolist()

batch_size = 64
indices = []

for i in tqdm(range(0, len(annotations), batch_size)):
    batch_annotations = annotations[i:i + batch_size]
    
    inputs = processor(
        text=batch_annotations,
        padding=True,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
    
    text_features /= text_features.norm(p=2, dim=-1, keepdim=True)
    
    text_features_cpu = text_features.cpu()
    for j, feature_vec in enumerate(text_features_cpu):
        current_index = i + j
        
        save_path = os.path.join(save_dir, f"{current_index}.pt")
        
        torch.save(feature_vec, save_path)
        
        indices.append(current_index)

if len(indices) == len(train_df):
    train_df["emb_index"] = indices
    train_df.to_csv("data/train/train.csv", index=False)
else:
    print("Error: Mismatch in number of indices and dataframe rows.")