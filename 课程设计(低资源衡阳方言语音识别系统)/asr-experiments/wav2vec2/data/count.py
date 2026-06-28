# CSV: D:\asr-experiments\wav2vec2\data\all.csv
# 编码: utf-8-sig
# 总行数: 4035
# 成功统计: 4035
# 缺失文件: 0
# 无法读取: 0
# 空路径: 0
# 总时长(秒): 11097.719
# 总时长(HH:MM:SS.sss): 03:04:57.719
# 总时长(小时): 3.0827

from __future__ import annotations

import argparse
import csv
import wave
from pathlib import Path
from typing import Iterable


def parse_args() -> argparse.Namespace:
    default_csv = Path(__file__).with_name("all.csv")
    parser = argparse.ArgumentParser(description="统计 CSV 中音频文件总时长")
    parser.add_argument(
        "--csv",
        type=Path,
        default=default_csv,
        help=f"CSV 文件路径，默认: {default_csv}",
    )
    parser.add_argument(
        "--path-column",
        default="path",
        help="CSV 中存放音频路径的列名，默认: path",
    )
    parser.add_argument(
        "--encoding",
        default=None,
        help="CSV 编码（不传则自动尝试 utf-8-sig/utf-8/gbk）",
    )
    parser.add_argument(
        "--show-missing",
        type=int,
        default=10,
        help="最多展示多少条缺失文件路径，默认: 10",
    )
    return parser.parse_args()


def load_rows(csv_path: Path, encoding: str | None) -> tuple[list[dict[str, str]], str]:
    encodings: Iterable[str] = [encoding] if encoding else ("utf-8-sig", "utf-8", "gbk")
    last_error: Exception | None = None

    for enc in encodings:
        try:
            with csv_path.open("r", encoding=enc, newline="") as f:
                rows = list(csv.DictReader(f))
            return rows, enc
        except UnicodeDecodeError as exc:
            last_error = exc

    raise RuntimeError(f"无法读取 CSV 编码: {csv_path} ({last_error})")


def resolve_audio_path(raw_path: str, csv_dir: Path, project_root: Path) -> Path:
    p = Path(raw_path.replace("\\", "/"))
    if p.is_absolute():
        return p

    candidates: list[Path] = [
        csv_dir / p,
        project_root / p,
        project_root.parent / p,
    ]

    if p.parts and p.parts[0] == project_root.name:
        trimmed = Path(*p.parts[1:])
        candidates.extend(
            [
                project_root / trimmed,
                csv_dir / trimmed,
            ]
        )

    for c in candidates:
        if c.exists():
            return c

    return candidates[0]


def wav_duration_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        if rate <= 0:
            raise ValueError(f"无效采样率: {rate}")
        return frames / rate


def format_hms(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds - h * 3600 - m * 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def main() -> None:
    args = parse_args()
    csv_path = args.csv.resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV 不存在: {csv_path}")

    rows, used_encoding = load_rows(csv_path, args.encoding)
    if not rows:
        print("CSV 为空，没有可统计的数据。")
        return

    fieldnames = set(rows[0].keys())
    if args.path_column not in fieldnames:
        raise KeyError(
            f"找不到路径列 `{args.path_column}`，当前列: {', '.join(sorted(fieldnames))}"
        )

    project_root = Path(__file__).resolve().parents[2]
    total_seconds = 0.0
    processed = 0
    missing: list[str] = []
    unreadable: list[str] = []
    empty_path = 0

    for row in rows:
        raw_path = (row.get(args.path_column) or "").strip()
        if not raw_path:
            empty_path += 1
            continue

        audio_path = resolve_audio_path(raw_path, csv_path.parent, project_root)
        if not audio_path.exists():
            missing.append(raw_path)
            continue

        try:
            total_seconds += wav_duration_seconds(audio_path)
            processed += 1
        except (wave.Error, OSError, EOFError, ValueError):
            unreadable.append(raw_path)

    print(f"CSV: {csv_path}")
    print(f"编码: {used_encoding}")
    print(f"总行数: {len(rows)}")
    print(f"成功统计: {processed}")
    print(f"缺失文件: {len(missing)}")
    print(f"无法读取: {len(unreadable)}")
    print(f"空路径: {empty_path}")
    print(f"总时长(秒): {total_seconds:.3f}")
    print(f"总时长(HH:MM:SS.sss): {format_hms(total_seconds)}")
    print(f"总时长(小时): {total_seconds / 3600:.4f}")

    if missing and args.show_missing > 0:
        print("\n缺失文件示例:")
        for p in missing[: args.show_missing]:
            print(f"- {p}")

    if unreadable and args.show_missing > 0:
        print("\n无法读取示例:")
        for p in unreadable[: args.show_missing]:
            print(f"- {p}")


if __name__ == "__main__":
    main()
