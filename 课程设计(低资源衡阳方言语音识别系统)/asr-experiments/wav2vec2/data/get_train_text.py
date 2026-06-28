from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable


def parse_args() -> argparse.Namespace:
    data_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="抽取 train.csv 句子，每行一句")
    parser.add_argument(
        "--csv",
        type=Path,
        default=data_dir / "train.csv",
        help="输入 CSV 路径",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=data_dir / "train_text.txt",
        help="输出文本路径（每行一句）",
    )
    parser.add_argument(
        "--text-column",
        default="sentence",
        help="句子列名，默认 sentence",
    )
    parser.add_argument(
        "--encoding",
        default=None,
        help="CSV 编码（不传则自动尝试 utf-8-sig/utf-8/gbk）",
    )
    return parser.parse_args()


def read_csv_rows(csv_path: Path, encoding: str | None) -> tuple[list[dict[str, str]], str]:
    encodings: Iterable[str] = [encoding] if encoding else ("utf-8-sig", "utf-8", "gbk")
    last_error: Exception | None = None

    for enc in encodings:
        try:
            with csv_path.open("r", encoding=enc, newline="") as f:
                return list(csv.DictReader(f)), enc
        except UnicodeDecodeError as exc:
            last_error = exc

    raise RuntimeError(f"无法读取 CSV 编码: {csv_path} ({last_error})")


def normalize_line(text: str) -> str:
    return text.replace("\r", " ").replace("\n", " ").strip()


def main() -> None:
    args = parse_args()
    csv_path = args.csv.resolve()
    output_path = args.output.resolve()

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV 不存在: {csv_path}")

    rows, used_encoding = read_csv_rows(csv_path, args.encoding)
    if not rows:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("", encoding="utf-8")
        print(f"CSV 为空，已创建空文件: {output_path}")
        return

    if args.text_column not in rows[0]:
        all_columns = ", ".join(rows[0].keys())
        raise KeyError(f"找不到列 `{args.text_column}`，当前列: {all_columns}")

    sentences: list[str] = []
    empty_count = 0
    for row in rows:
        text = normalize_line(row.get(args.text_column, ""))
        if text:
            sentences.append(text)
        else:
            empty_count += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as f:
        for sentence in sentences:
            f.write(sentence + "\n")

    print(f"CSV: {csv_path}")
    print(f"编码: {used_encoding}")
    print(f"总行数: {len(rows)}")
    print(f"输出句子数: {len(sentences)}")
    print(f"空句子数: {empty_count}")
    print(f"输出文件: {output_path}")


if __name__ == "__main__":
    main()
