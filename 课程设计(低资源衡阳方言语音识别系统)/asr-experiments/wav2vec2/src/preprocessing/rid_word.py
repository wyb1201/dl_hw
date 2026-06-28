import os
import re

change_file = r"D:\asr-experiments\wav2vec2\static\分工\speaker_03\E54.txt"

# 去掉每行里冒号前的内容（含冒号），保留冒号后的部分
def strip_before_colon(line: str) -> str:
    # 支持半角: 和全角：，若无冒号则原样去掉首尾空白
    if ":" not in line and "：" not in line:
        return line.strip()
    parts = re.split(r"[:：]", line)
    return parts[-1].strip()


if __name__ == "__main__":
    if not os.path.exists(change_file):
        raise FileNotFoundError(change_file)

    with open(change_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    cleaned = [strip_before_colon(line) for line in lines if line.strip()]

    with open(change_file, "w", encoding="utf-8") as f:
        for line in cleaned:
            f.write(line + "\n")

    print(f"处理完成，已覆盖写回: {change_file}")
