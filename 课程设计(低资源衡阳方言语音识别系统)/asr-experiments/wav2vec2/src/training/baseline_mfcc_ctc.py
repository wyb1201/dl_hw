# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import time
from pathlib import Path

import pandas as pd
import torch
import torchaudio
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


DEFAULT_OUTPUT_DIR = "asr-experiments/wav2vec2/baseline_results"
PUNCTUATION_REGEX = r"[，。！？、；：“”‘’（）《》〈〉【】\[\]{}<>…,.?!;:\"'()\-_%~`·]"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MFCC + BiLSTM + CTC baseline for Hengyang dialect ASR.")
    parser.add_argument("--train-csv", default="asr-experiments/wav2vec2/data/train.csv")
    parser.add_argument("--valid-csv", default="asr-experiments/wav2vec2/data/valid.csv")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--quick", action="store_true", help="Run a small smoke test.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--n-mfcc", type=int, default=40)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-valid-samples", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def clean_text(text: str) -> str:
    text = "" if text is None else str(text)
    text = re.sub(PUNCTUATION_REGEX, "", text)
    text = re.sub(r"\s+", "", text)
    return text.strip()


def levenshtein(reference: str, hypothesis: str) -> int:
    if not reference:
        return len(hypothesis)
    if not hypothesis:
        return len(reference)
    previous = list(range(len(hypothesis) + 1))
    for i, ref_char in enumerate(reference, 1):
        current = [i] + [0] * len(hypothesis)
        for j, hyp_char in enumerate(hypothesis, 1):
            substitution = 0 if ref_char == hyp_char else 1
            current[j] = min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + substitution,
            )
        previous = current
    return previous[-1]


def cer(reference: str, hypothesis: str) -> float:
    return levenshtein(reference, hypothesis) / max(1, len(reference))


def resolve_path(path_value: str | Path, anchors: list[Path]) -> Path:
    path = Path(path_value)
    if path.is_absolute() and path.exists():
        return path
    if path.is_absolute():
        return path

    candidates: list[Path] = []
    for anchor in anchors:
        candidates.append(anchor / path)
        if path.parts[:2] == ("asr-experiments", "wav2vec2"):
            candidates.append(anchor / "课程设计(低资源衡阳方言语音识别系统)" / path)
        if path.parts[:1] == ("wav2vec2",):
            candidates.append(anchor / "asr-experiments" / path)

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    for candidate in candidates:
        if candidate.parent.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def default_anchors() -> list[Path]:
    script_path = Path(__file__).resolve()
    anchors = [Path.cwd().resolve()]
    anchors.extend(parent for parent in script_path.parents)
    return anchors


def load_rows(csv_path: Path, anchors: list[Path], limit: int, seed: int) -> list[dict[str, str]]:
    data = pd.read_csv(csv_path)
    rows: list[dict[str, str]] = []
    for _, row in data.iterrows():
        sentence = clean_text(row.get("sentence", ""))
        if not sentence:
            continue
        wav_path = resolve_path(str(row.get("path", "")), anchors)
        if not wav_path.exists():
            continue
        rows.append({"path": str(wav_path), "sentence": sentence})

    if limit and limit < len(rows):
        rng = random.Random(seed)
        rows = rng.sample(rows, limit)
    return rows


def build_vocab(rows: list[dict[str, str]]) -> tuple[dict[str, int], dict[int, str]]:
    chars = sorted({char for row in rows for char in row["sentence"]})
    char_to_id = {char: index + 1 for index, char in enumerate(chars)}
    id_to_char = {index: char for char, index in char_to_id.items()}
    id_to_char[0] = ""
    return char_to_id, id_to_char


class SpeechDataset(Dataset):
    def __init__(self, rows: list[dict[str, str]], char_to_id: dict[str, int], n_mfcc: int) -> None:
        self.rows = rows
        self.char_to_id = char_to_id
        self.mfcc = torchaudio.transforms.MFCC(
            sample_rate=16000,
            n_mfcc=n_mfcc,
            melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 80},
        )
        self.resamplers: dict[int, torchaudio.transforms.Resample] = {}

    def __len__(self) -> int:
        return len(self.rows)

    def _resample(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        if sample_rate == 16000:
            return waveform
        if sample_rate not in self.resamplers:
            self.resamplers[sample_rate] = torchaudio.transforms.Resample(sample_rate, 16000)
        return self.resamplers[sample_rate](waveform)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        row = self.rows[index]
        waveform, sample_rate = torchaudio.load(row["path"])
        waveform = waveform.mean(dim=0, keepdim=True)
        waveform = self._resample(waveform, int(sample_rate))
        features = self.mfcc(waveform).squeeze(0).transpose(0, 1).contiguous()
        features = (features - features.mean(dim=0, keepdim=True)) / (features.std(dim=0, keepdim=True) + 1e-5)
        labels = torch.tensor([self.char_to_id[char] for char in row["sentence"]], dtype=torch.long)
        return {"features": features, "labels": labels, "text": row["sentence"]}


def collate_batch(batch: list[dict[str, torch.Tensor | str]]) -> dict[str, torch.Tensor | list[str]]:
    features = [item["features"] for item in batch]
    labels = [item["labels"] for item in batch]
    texts = [str(item["text"]) for item in batch]
    input_lengths = torch.tensor([feature.shape[0] for feature in features], dtype=torch.long)
    target_lengths = torch.tensor([label.shape[0] for label in labels], dtype=torch.long)
    return {
        "features": pad_sequence(features, batch_first=True),
        "input_lengths": input_lengths,
        "targets": torch.cat(labels),
        "target_lengths": target_lengths,
        "texts": texts,
    }


class MfcCtcBaseline(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, num_layers: int, vocab_size: int) -> None:
        super().__init__()
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.1 if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Linear(hidden_size * 2, vocab_size)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        encoded, _ = self.encoder(features)
        return self.classifier(encoded)


def greedy_decode(logits: torch.Tensor, input_lengths: torch.Tensor, id_to_char: dict[int, str]) -> list[str]:
    token_ids = logits.argmax(dim=-1).cpu()
    predictions: list[str] = []
    for sequence, length in zip(token_ids, input_lengths.cpu()):
        previous = -1
        chars: list[str] = []
        for token_id in sequence[: int(length)].tolist():
            if token_id != 0 and token_id != previous:
                chars.append(id_to_char.get(token_id, ""))
            previous = token_id
        predictions.append("".join(chars))
    return predictions


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.CTCLoss,
    device: torch.device,
    id_to_char: dict[int, str],
) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    sample_cers: list[float] = []
    total_distance = 0
    total_chars = 0

    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device)
            input_lengths = batch["input_lengths"].to(device)
            targets = batch["targets"].to(device)
            target_lengths = batch["target_lengths"].to(device)
            logits = model(features)
            log_probs = logits.log_softmax(dim=-1).transpose(0, 1)
            loss = criterion(log_probs, targets, input_lengths, target_lengths)
            losses.append(float(loss.item()))

            predictions = greedy_decode(logits, batch["input_lengths"], id_to_char)
            for reference, hypothesis in zip(batch["texts"], predictions):
                distance = levenshtein(reference, hypothesis)
                sample_cers.append(distance / max(1, len(reference)))
                total_distance += distance
                total_chars += len(reference)

    return {
        "loss": sum(losses) / max(1, len(losses)),
        "mean_cer": sum(sample_cers) / max(1, len(sample_cers)),
        "weighted_cer": total_distance / max(1, total_chars),
    }


def write_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    id_to_char: dict[int, str],
    output_path: Path,
) -> None:
    model.eval()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["CER", "LEN", "GT", "PR"])
        with torch.no_grad():
            for batch in loader:
                features = batch["features"].to(device)
                logits = model(features)
                predictions = greedy_decode(logits, batch["input_lengths"], id_to_char)
                for reference, hypothesis in zip(batch["texts"], predictions):
                    writer.writerow([f"{cer(reference, hypothesis):.4f}", len(reference), reference, hypothesis])


def train(args: argparse.Namespace) -> None:
    if args.quick:
        args.epochs = 1
        args.hidden_size = min(args.hidden_size, 64)
        args.num_layers = 1
        args.max_train_samples = args.max_train_samples or 64
        args.max_valid_samples = args.max_valid_samples or 32
        if args.output_dir == DEFAULT_OUTPUT_DIR:
            args.output_dir = DEFAULT_OUTPUT_DIR + "_quick"

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    anchors = default_anchors()
    train_csv = resolve_path(args.train_csv, anchors)
    valid_csv = resolve_path(args.valid_csv, anchors)
    output_dir = resolve_path(args.output_dir, anchors)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_rows = load_rows(train_csv, anchors, args.max_train_samples, args.seed)
    valid_rows = load_rows(valid_csv, anchors, args.max_valid_samples, args.seed + 1)
    if not train_rows:
        raise RuntimeError(f"No training rows found from {train_csv}")
    if not valid_rows:
        raise RuntimeError(f"No validation rows found from {valid_csv}")

    char_to_id, id_to_char = build_vocab(train_rows + valid_rows)
    vocab_path = output_dir / "baseline_vocab.json"
    vocab_path.write_text(json.dumps(char_to_id, ensure_ascii=False, indent=2), encoding="utf-8")

    train_dataset = SpeechDataset(train_rows, char_to_id, args.n_mfcc)
    valid_dataset = SpeechDataset(valid_rows, char_to_id, args.n_mfcc)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
    )

    device = torch.device(args.device)
    model = MfcCtcBaseline(args.n_mfcc, args.hidden_size, args.num_layers, len(char_to_id) + 1).to(device)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    checkpoint_path = output_dir / "baseline_mfcc_ctc.pt"
    if args.checkpoint:
        state = torch.load(resolve_path(args.checkpoint, anchors), map_location=device)
        model.load_state_dict(state["model"])

    history: list[dict[str, float]] = []
    best_weighted_cer = float("inf")
    start_time = time.time()

    if not args.eval_only:
        for epoch in range(1, args.epochs + 1):
            model.train()
            train_losses: list[float] = []
            for batch in train_loader:
                features = batch["features"].to(device)
                input_lengths = batch["input_lengths"].to(device)
                targets = batch["targets"].to(device)
                target_lengths = batch["target_lengths"].to(device)

                optimizer.zero_grad(set_to_none=True)
                logits = model(features)
                log_probs = logits.log_softmax(dim=-1).transpose(0, 1)
                loss = criterion(log_probs, targets, input_lengths, target_lengths)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                train_losses.append(float(loss.item()))

            metrics = evaluate(model, valid_loader, criterion, device, id_to_char)
            metrics["epoch"] = float(epoch)
            metrics["train_loss"] = sum(train_losses) / max(1, len(train_losses))
            history.append(metrics)
            print(
                f"epoch={epoch} train_loss={metrics['train_loss']:.4f} "
                f"valid_loss={metrics['loss']:.4f} mean_cer={metrics['mean_cer']:.4f} "
                f"weighted_cer={metrics['weighted_cer']:.4f}",
                flush=True,
            )
            if metrics["weighted_cer"] < best_weighted_cer:
                best_weighted_cer = metrics["weighted_cer"]
                torch.save(
                    {
                        "model": model.state_dict(),
                        "char_to_id": char_to_id,
                        "args": vars(args),
                    },
                    checkpoint_path,
                )

        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state["model"])

    final_metrics = evaluate(model, valid_loader, criterion, device, id_to_char)
    summary = {
        "model": "MFCC + BiLSTM + CTC",
        "quick": args.quick,
        "train_samples": len(train_rows),
        "valid_samples": len(valid_rows),
        "epochs": args.epochs,
        "n_mfcc": args.n_mfcc,
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "final": final_metrics,
        "history": history,
        "total_sec": time.time() - start_time,
    }
    (output_dir / "baseline_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_predictions(model, valid_loader, device, id_to_char, output_dir / "baseline_predictions.csv")
    print(json.dumps(summary["final"], ensure_ascii=False), flush=True)
    print(f"summary: {output_dir / 'baseline_summary.json'}", flush=True)
    print(f"predictions: {output_dir / 'baseline_predictions.csv'}", flush=True)


if __name__ == "__main__":
    train(parse_args())
