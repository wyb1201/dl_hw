# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import re
import uuid
import zipfile
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DATASET_CSV = BASE_DIR / "data" / "all.csv"
SPEAKER_ROOT = BASE_DIR / "static" / "分工"
TEXT_EXTENSIONS = {".py", ".md", ".txt", ".html", ".json", ".yaml", ".yml"}


def discover_speakers() -> list[str]:
    with DATASET_CSV.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames or "speaker" not in reader.fieldnames:
            raise ValueError(f"{DATASET_CSV} 缺少 speaker 字段")
        speakers = sorted({row["speaker"].strip() for row in reader if row["speaker"].strip()})
    return [speaker for speaker in speakers if not re.fullmatch(r"speaker_\d{2}", speaker)]


def build_mapping(speakers: list[str]) -> dict[str, str]:
    existing_ids = {
        path.name for path in SPEAKER_ROOT.iterdir() if path.is_dir() and re.fullmatch(r"speaker_\d{2}", path.name)
    }
    available_ids = (
        candidate
        for index in range(1, 100)
        if (candidate := f"speaker_{index:02d}") not in existing_ids
    )
    return {speaker: next(available_ids) for speaker in speakers}


def replace_identifiers(value: str, mapping: dict[str, str]) -> str:
    result = value
    for original, anonymous in mapping.items():
        token_pattern = rf"(?<![A-Za-z0-9_]){re.escape(original)}(?![A-Za-z0-9_])"
        result = re.sub(token_pattern, anonymous, result)
    return result


def update_csv(path: Path, mapping: dict[str, str], apply: bool) -> bool:
    try:
        with path.open(encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            if not reader.fieldnames:
                return False
            rows = list(reader)
    except (UnicodeDecodeError, csv.Error):
        return False

    changed = False
    for row in rows:
        for field, value in row.items():
            if value is None:
                continue
            replacement = mapping.get(value, value) if field == "speaker" else replace_identifiers(value, mapping)
            if replacement != value:
                row[field] = replacement
                changed = True

    if changed and apply:
        with path.open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=reader.fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    return changed


def update_text_file(path: Path, mapping: dict[str, str], apply: bool) -> bool:
    try:
        original_text = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        return False
    updated_text = replace_identifiers(original_text, mapping)
    if updated_text == original_text:
        return False
    if apply:
        path.write_text(updated_text, encoding="utf-8")
    return True


def rename_descendants(mapping: dict[str, str], apply: bool) -> list[tuple[Path, Path]]:
    planned = []
    for original in mapping:
        root = SPEAKER_ROOT / original
        if not root.exists():
            continue
        for path in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
            new_name = replace_identifiers(path.name, mapping)
            if new_name != path.name:
                target = path.with_name(new_name)
                planned.append((path, target))
                if apply:
                    if target.exists():
                        raise FileExistsError(f"重命名目标已存在: {target}")
                    path.rename(target)
    return planned


def rename_speaker_roots(mapping: dict[str, str], apply: bool) -> list[tuple[Path, Path]]:
    planned = []
    temporary_paths = {}
    for original, anonymous in mapping.items():
        source = SPEAKER_ROOT / original
        target = SPEAKER_ROOT / anonymous
        if not source.exists():
            continue
        if target.exists():
            raise FileExistsError(f"匿名目录已存在: {target}")
        planned.append((source, target))
        if apply:
            temporary = SPEAKER_ROOT / f".__anonymizing_{uuid.uuid4().hex}"
            source.rename(temporary)
            temporary_paths[temporary] = target
    if apply:
        for temporary, target in temporary_paths.items():
            temporary.rename(target)
    return planned


def rebuild_static_archive() -> None:
    archive_path = BASE_DIR / "static.zip"
    temporary_path = BASE_DIR / "static.anonymized.tmp.zip"
    if temporary_path.exists():
        temporary_path.unlink()
    with zipfile.ZipFile(temporary_path, "w", compression=zipfile.ZIP_STORED) as archive:
        for path in sorted((BASE_DIR / "static").rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(BASE_DIR).as_posix())
    temporary_path.replace(archive_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="匿名化说话人字段、路径和目录名称")
    parser.add_argument("--apply", action="store_true", help="实际写入；不指定时仅预览")
    parser.add_argument(
        "--rebuild-archive",
        action="store_true",
        help="从匿名化后的 static 目录重建 static.zip",
    )
    args = parser.parse_args()

    speakers = discover_speakers()
    if not speakers:
        print("未发现需要匿名化的说话人标识。")
        if args.rebuild_archive:
            rebuild_static_archive()
            print("已重建 static.zip。")
        return
    mapping = build_mapping(speakers)

    changed_files = []
    for path in BASE_DIR.rglob("*"):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        if path.suffix.lower() == ".csv":
            changed = update_csv(path, mapping, args.apply)
        elif path.suffix.lower() in TEXT_EXTENSIONS:
            changed = update_text_file(path, mapping, args.apply)
        else:
            changed = False
        if changed:
            changed_files.append(path)

    descendant_renames = rename_descendants(mapping, args.apply)
    root_renames = rename_speaker_roots(mapping, args.apply)

    mode = "已执行" if args.apply else "预览"
    print(f"{mode}: {len(mapping)} 个说话人、{len(changed_files)} 个文本文件")
    print(f"{mode}: {len(descendant_renames) + len(root_renames)} 个文件或目录重命名")
    if not args.apply:
        print("确认后使用 --apply 参数执行。")
    if args.rebuild_archive:
        rebuild_static_archive()
        print("已重建 static.zip。")


if __name__ == "__main__":
    main()
