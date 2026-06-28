import os
import re
import pandas as pd

# 配置区：根据当前批次修改
audio_dir = r"D:\asr-experiments\wav2vec2\static\分工\speaker_01\part\per\C147"  # 音频文件夹路径
text_file = r"D:\asr-experiments\wav2vec2\static\分工\speaker_01\C147.txt"
train_csv_path = r"D:\asr-experiments\wav2vec2\data\all.csv"
speaker = "speaker_03"

# 读取文本：剥掉行首编号（如“12.”、“12:”等）
def drop_prefix_number(line: str) -> str:
    # 匹配行首数字及紧随的标点/空白：12.  12:  12、 12 - 等
    return re.sub(r"^\s*\d+\s*[.、:：-]?\s*", "", line)


def load_sentences(path: str):
    with open(path, "r", encoding="utf-8") as f:
        raw_text = f.read().strip()

    if not raw_text:
        return []

    lines = [ln.strip() for ln in raw_text.splitlines() if ln.strip()]

    # 兼容 B.txt 这类“单行 + 顿号分隔”格式
    if len(lines) == 1 and "、" in lines[0]:
        parts = [p.strip() for p in lines[0].split("、") if p.strip()]
        return [drop_prefix_number(p) for p in parts]

    return [drop_prefix_number(ln) for ln in lines]


def wav_sort_key(filename: str):
    """
    按文件名中的数字做自然排序，避免字符串排序导致 1,10,100,2 的错序。
    例如：-01.wav, -02.wav, ..., -10.wav, ..., -147.wav
    """
    stem = os.path.splitext(filename)[0]
    numbers = re.findall(r"\d+", stem)
    if numbers:
        return (0, tuple(int(n) for n in numbers), stem)
    return (1, stem)


sentences = load_sentences(text_file)

# 获取音频并排序
audio_files = sorted(
    [f for f in os.listdir(audio_dir) if f.lower().endswith(".wav")],
    key=wav_sort_key,
)

assert len(sentences) == len(audio_files), f"数量不一致：文本{len(sentences)}条，音频{len(audio_files)}个"

data = []
for wav, text in zip(audio_files, sentences):
    full_path = os.path.join(audio_dir, wav)
    data.append({"path": full_path, "sentence": text, "speaker": speaker})

df = pd.DataFrame(data)
if os.path.exists(train_csv_path):
    # 追加到已有 all.csv 末尾，不重复写表头
    df.to_csv(train_csv_path, mode="a", header=False, index=False)
else:
    # 如果还不存在则创建并写入表头
    df.to_csv(train_csv_path, index=False)

print("all.csv 生成完成")
