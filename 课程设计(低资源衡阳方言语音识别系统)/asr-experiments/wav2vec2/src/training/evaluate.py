import re
from functools import lru_cache

import torch
import torchaudio
from datasets import load_dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

model_id = "./wav2vec2-large-xlsr-hengyang"
model_path = "./wav2vec2-large-xlsr-hengyang/checkpoint-xxxx"  # 替换成实际的 checkpoint 路径
test_csv = "data/valid.csv"  # 或 data/test.csv

chars_to_ignore_regex = '[\,\?\.\!\-\;\:"\“\%\‘\”\�\．\⋯\！\－\：\–\。\》\,\）\,\？\；\～\~\…\︰\，\（\」\‧\《\﹔\、\—\／\,\「\﹖\·\']'


def _levenshtein_distance(ref: str, hyp: str) -> int:
    """轻量级编辑距离实现，用于计算 CER。"""
    m, n = len(ref), len(hyp)
    if m == 0:
        return n
    if n == 0:
        return m
    prev_row = list(range(n + 1))
    for i in range(1, m + 1):
        cur_row = [i] + [0] * n
        ref_char = ref[i - 1]
        for j in range(1, n + 1):
            cost = 0 if ref_char == hyp[j - 1] else 1
            cur_row[j] = min(
                prev_row[j] + 1,       # deletion
                cur_row[j - 1] + 1,    # insertion
                prev_row[j - 1] + cost # substitution
            )
        prev_row = cur_row
    return prev_row[n]


def char_error_rate(predictions, references) -> float:
    total_dist, total_chars = 0, 0
    for hyp, ref in zip(predictions, references):
        total_dist += _levenshtein_distance(ref, hyp)
        total_chars += len(ref)
    return total_dist / total_chars if total_chars > 0 else 0.0


@lru_cache(maxsize=None)
def get_resampler(orig_sr: int):
    return torchaudio.transforms.Resample(orig_sr, 16000)


def preprocess(batch):
    batch["sentence"] = re.sub(chars_to_ignore_regex, "", batch["sentence"]).strip() + " "
    speech, sr = torchaudio.load(batch["path"])
    sr = int(sr)
    if sr != 16000:
        speech = get_resampler(sr)(speech)
    batch["speech"] = speech.squeeze().numpy()
    return batch


def infer(batch, processor, model):
    inputs = processor(batch["speech"], sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(
            inputs.input_values.to("cuda"),
            attention_mask=inputs.attention_mask.to("cuda")
        ).logits
    pred_ids = torch.argmax(logits, dim=-1)
    batch["pred_strings"] = processor.batch_decode(pred_ids)
    return batch


def main():
    ds = load_dataset("csv", data_files={"test": test_csv})["test"]
    ds = ds.map(preprocess)

    processor = Wav2Vec2Processor.from_pretrained(model_id)
    model = Wav2Vec2ForCTC.from_pretrained(model_path).to("cuda").eval()

    result = ds.map(lambda b: infer(b, processor, model), batched=True, batch_size=16)
    print("CER:", char_error_rate(result["pred_strings"], result["sentence"]))


if __name__ == "__main__":
    main()
