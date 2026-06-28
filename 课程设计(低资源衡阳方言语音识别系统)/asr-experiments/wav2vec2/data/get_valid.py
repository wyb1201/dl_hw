import pandas as pd
from pathlib import Path

train_path = Path(r"wav2vec2\data\all.csv")
df = pd.read_csv(train_path)

# 打乱
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# 9:1 切分
split = int(len(df) * 0.9)
train_df = df.iloc[:split]
valid_df = df.iloc[split:]

train_df.to_csv(r"wav2vec2\data\train.csv", index=False)
valid_df.to_csv(r"wav2vec2\data\valid.csv", index=False)

print("done")
print("train:", len(train_df))
print("valid:", len(valid_df))
