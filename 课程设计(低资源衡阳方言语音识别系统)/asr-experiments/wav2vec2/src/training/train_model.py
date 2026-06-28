import json
import random
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import jiwer


import numpy as np
import pandas as pd
import torch
import torchaudio
import transformers
from datasets import ClassLabel, load_dataset
from transformers import (Trainer, TrainingArguments, Wav2Vec2CTCTokenizer,
                          Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC,
                          Wav2Vec2Processor, EarlyStoppingCallback)

import argparse
parser = argparse.ArgumentParser() 
parser.add_argument('--model', type=str, default="facebook/wav2vec2-large-xlsr-53")
parser.add_argument('--unfreeze', action='store_true')
parser.add_argument('--lr', type=float, default=5e-5)
parser.add_argument('--warmup', type=float, default=500)
parser.add_argument('--train_csv', type=str, default="asr-experiments/wav2vec2/data/train.csv")
parser.add_argument('--eval_csv', type=str, default="asr-experiments/wav2vec2/data/valid.csv")

args = parser.parse_args()


print(f"args: {args}")

dataset = load_dataset(
    "csv",
    data_files={"train": args.train_csv, "test": args.eval_csv},
)
common_voice_train = dataset["train"]
common_voice_test = dataset["test"]


chars_to_ignore_regex = '[\丶\,\?\.\!\-\;\:"\“\%\‘\”\�\．\⋯\！\－\：\–\。\》\,\）\,\？\；\～\~\…\︰\，\（\」\‧\《\﹔\、\—\／\,\「\﹖\·\']'

import string
def remove_special_characters(batch):
    batch["sentence"] = re.sub(chars_to_ignore_regex, "", batch["sentence"]).strip() + " "
    return batch


common_voice_train = common_voice_train.map(remove_special_characters)
common_voice_test = common_voice_test.map(remove_special_characters)

def extract_all_chars(batch):
    all_text = " ".join(batch["sentence"])
    vocab = list(set(all_text))
    return {"vocab": [vocab], "all_text": [all_text]}

vocab_train = common_voice_train.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=common_voice_train.column_names,)
vocab_test = common_voice_test.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=common_voice_test.column_names,)

def is_good_char(ch: str) -> bool:
    if ch.isspace():
        return True
    # 过滤 ASCII
    if ch.isascii():
        return False
    # 过滤不可见/控制字符
    if ord(ch) < 32:
        return False
    # 过滤一些奇怪空白
    if ch in ["\u200b", "\ufeff", "\u3000"]:  # 零宽空格/BOM/全角空格
        return False
    return True

vocab_list = sorted(
    [ch for ch in set(vocab_train["vocab"][0]) | set(vocab_test["vocab"][0]) if is_good_char(ch)]
)

# 保证空格存在（CTC delimiter）
if " " not in vocab_list:
    vocab_list.append(" ")

vocab_dict = {v: k for k, v in enumerate(vocab_list)}
vocab_dict["|"] = vocab_dict[" "]
del vocab_dict[" "]
vocab_dict["[UNK]"] = len(vocab_dict)
vocab_dict["[PAD]"] = len(vocab_dict)

with open("vocab.json", "w") as vocab_file:
    json.dump(vocab_dict, vocab_file)

tokenizer = Wav2Vec2CTCTokenizer("./vocab.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")

feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True,)

processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
processor.save_pretrained("./wav2vec2-large-xlsr-hengyang")


# resamplers = {
#     48000: torchaudio.transforms.Resample(48000, 16000),
#     44100: torchaudio.transforms.Resample(44100, 16000),
# }

# def load_and_resample(batch):
#     speech_array, sampling_rate = torchaudio.load(batch["path"])
#     batch["speech"] = resamplers[sampling_rate](speech_array).squeeze().numpy()
#     batch["sampling_rate"] = 16_000
#     batch["target_text"] = batch["sentence"]
#     return batch

from functools import lru_cache

@lru_cache(maxsize=None)
def get_resampler(orig_sr: int):
    return torchaudio.transforms.Resample(orig_sr, 16000)

def load_and_resample(batch):
    speech_array, sampling_rate = torchaudio.load(batch["path"])
    sr = int(sampling_rate)

    if sr != 16000:
        speech_array = get_resampler(sr)(speech_array)

    batch["speech"] = speech_array.squeeze().numpy()
    batch["sampling_rate"] = 16000
    batch["target_text"] = batch["sentence"]
    return batch


common_voice_train = common_voice_train.map(load_and_resample, remove_columns=common_voice_train.column_names,)
common_voice_test = common_voice_test.map(load_and_resample, remove_columns=common_voice_test.column_names,)

def prepare_dataset(batch):
    batch["input_values"] = processor(batch["speech"], sampling_rate=16000).input_values
    with processor.as_target_processor():
        batch["labels"] = processor(batch["target_text"]).input_ids
    return batch

common_voice_train = common_voice_train.map(prepare_dataset, remove_columns=common_voice_train.column_names, batch_size=-1, num_proc=10, batched=True,)
common_voice_test = common_voice_test.map(prepare_dataset, remove_columns=common_voice_test.column_names, batch_size=-1, num_proc=10, batched=True,)


@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [
            {"input_values": feature["input_values"]} for feature in features
        ]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        batch["labels"] = labels

        return batch


data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)


def _levenshtein_distance(ref: str, hyp: str) -> int:
    """Lightweight Levenshtein distance to avoid external metric files."""
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
                prev_row[j] + 1,      # deletion
                cur_row[j - 1] + 1,   # insertion
                prev_row[j - 1] + cost  # substitution
            )
        prev_row = cur_row
    return prev_row[n]


def char_error_rate(predictions, references) -> float:
    total_dist, total_chars = 0, 0
    for hyp, ref in zip(predictions, references):
        total_dist += _levenshtein_distance(ref, hyp)
        total_chars += len(ref)
    return total_dist / total_chars if total_chars > 0 else 0.0

# def compute_metrics(pred):
#     pred_logits = pred.predictions
#     pred_ids = np.argmax(pred_logits, axis=-1)

#     pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

#     pred_str = processor.batch_decode(pred_ids)
#     # we do not want to group tokens when computing the metrics
#     label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

#     cer = char_error_rate(pred_str, label_str)
#     #         ---------------------------------------------------------------
# # 调试
#     PRINT_N = 10
#     printed = getattr(compute_metrics, "_printed", 0)

#     pred_ids = np.argmax(pred.predictions, axis=-1)
    
#     pred_str = processor.batch_decode(pred_ids)  # 预测解码
#     label_ids = pred.label_ids.copy()
#     label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
#     label_str = processor.batch_decode(label_ids, group_tokens=False)

#     if printed == 0:
#         for i in range(min(PRINT_N, len(pred_str))):
#             print("GT:", label_str[i])
#             print("PR:", pred_str[i])
#             print("-"*60)
#         compute_metrics._printed = 1
# #         ---------------------------------------------------------------

#     return {"cer": cer}

import json, numpy as np

INV_VOCAB = None
def _load_inv_vocab():
    global INV_VOCAB
    if INV_VOCAB is None:
        vocab = json.load(open(f"{training_args.output_dir}/vocab.json","r",encoding="utf-8"))
        INV_VOCAB = {v:k for k,v in vocab.items()}
    return INV_VOCAB

def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    # ====== 诊断：看第0条预测是否塌缩 ======
    pid = pred_ids[0]
    vals, cnts = np.unique(pid, return_counts=True)
    top = sorted(zip(vals, cnts), key=lambda x: -x[1])[:5]
    inv = _load_inv_vocab()
    print("TOP_IDS:", [(int(i), int(c), inv.get(int(i), "?")) for i,c in top])
    print("pad_id:", processor.tokenizer.pad_token_id, "pad_tok:", inv.get(processor.tokenizer.pad_token_id, "?"))

    # ====== CER ======
    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str = processor.batch_decode(pred_ids)
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
    cer = char_error_rate(pred_str, label_str)
    return {"cer": cer}



model = Wav2Vec2ForCTC.from_pretrained(
    args.model,
    attention_dropout=0.1,
    hidden_dropout=0.1,
    feat_proj_dropout=0.0,
    mask_time_prob=0.05,
    layerdrop=0.1,
    gradient_checkpointing=True,
    ctc_loss_reduction="mean",
    pad_token_id=processor.tokenizer.pad_token_id,
    vocab_size=len(processor.tokenizer),
)

if not args.unfreeze:
    model.freeze_feature_extractor()

# 调试
print("pad:", processor.tokenizer.pad_token, processor.tokenizer.pad_token_id)
print("unk:", processor.tokenizer.unk_token, processor.tokenizer.unk_token_id)
print("wdt:", getattr(processor.tokenizer, "word_delimiter_token", None),
            getattr(processor.tokenizer, "word_delimiter_token_id", None))
from transformers import TrainerCallback
import torch

class PrintPredCallback(TrainerCallback):
    def __init__(self, processor, eval_dataset, n=5):
        self.processor = processor
        self.eval_dataset = eval_dataset
        self.n = n

    def on_evaluate(self, args, state, control, **kwargs):
        model = kwargs["model"]
        model.eval()

        n = min(self.n, len(self.eval_dataset))
        for i in range(n):
            ex = self.eval_dataset[i]
            input_values = torch.tensor(ex["input_values"]).unsqueeze(0).to(model.device)

            with torch.no_grad():
                logits = model(input_values).logits
            pred_ids = torch.argmax(logits, dim=-1)
            pred_str = self.processor.batch_decode(pred_ids.cpu().numpy(), group_tokens=True)[0]

            # label
            label_ids = ex["labels"]
            label_str = self.processor.batch_decode([label_ids], group_tokens=False)[0]

            print(f"GT: {label_str}")
            print(f"PR: {pred_str}")
            print("-"*60)

        model.train()
        return control
# ---------------------------------------------------------------------
    
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./wav2vec2-large-xlsr-hengyang",

    group_by_length=True,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,

    evaluation_strategy="epoch",
    num_train_epochs=15,

    fp16=True,
    fp16_backend="auto",

    logging_strategy="steps",
    logging_steps=50,

    # ✅ TensorBoard
    report_to=["tensorboard"],
    logging_dir="./wav2vec2-large-xlsr-hengyang/tb",
    run_name=f"xlsr53_hengyang_lr{args.lr}",  # 可选：区分不同实验

    learning_rate=args.lr,
    warmup_steps=100,

    save_strategy="epoch",
    save_total_limit=1,

    load_best_model_at_end=False,
    
)



trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=common_voice_train,
    eval_dataset=common_voice_test,
    tokenizer=processor.feature_extractor,
    callbacks=[
        PrintPredCallback(processor, common_voice_test, n=5),
    ],
)
trainer.train()


 
