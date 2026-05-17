"""Merge LeRobotDataset v3.0 directories produced by ``utils/data_collection``.

The recorder in ``utils/data_collection`` writes the v3.0 layout:

    <root>/
      meta/info.json
      meta/stats.json
      meta/tasks.parquet
      meta/episodes/chunk-XXX/file-YYY.parquet
      data/chunk-XXX/file-YYY.parquet
      videos/<video_key>/chunk-XXX/file-YYY.mp4

Each frame parquet file packs the rows of one or more episodes; each
video mp4 packs the encoded frames of those same episodes back-to-back.
The episode parquet ties them together: per episode it records the
data/video chunk + file it lives in, the byte/row range
(`dataset_from_index`, `dataset_to_index`), and the time window inside
the video file.

To merge N such datasets we:

1. Validate that the shared schema agrees (features, fps, codec, robot
   type, codebase_version, chunks_size).
2. Build a unified ``tasks.parquet`` (deduped by task string).
3. Stream the input data/video/episode files into the output, remapping:
   - ``index``               (global frame index) -- shift by frame_offset
   - ``episode_index``       -- shift by episode_offset
   - ``task_index``          -- remap via the unified task table
   - ``data/file_index``     -- shift by data_file_offset (per chunk)
   - ``videos/<key>/file_index`` -- shift by video_file_offset
   - ``dataset_from_index`` / ``dataset_to_index`` -- shift by frame_offset
4. Recompute ``info.json`` totals and ``stats.json`` (count-weighted
   pool of the per-episode stats already stored in the episode parquet).

The video files themselves are copied byte-for-byte. We never re-encode.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

logger = logging.getLogger("merge_datasets")

DATA_PATH_TMPL = "data/chunk-{chunk:03d}/file-{file:03d}.parquet"
VIDEO_PATH_TMPL = "videos/{key}/chunk-{chunk:03d}/file-{file:03d}.mp4"
META_EP_PATH_TMPL = "meta/episodes/chunk-{chunk:03d}/file-{file:03d}.parquet"


@dataclass
class FileAddr:
    chunk: int
    file: int


def _chunked_addr(seq_index: int, chunks_size: int) -> FileAddr:
    """Turn a flat sequence index into (chunk, file) honoring chunks_size."""
    return FileAddr(chunk=seq_index // chunks_size, file=seq_index % chunks_size)


def _read_info(root: Path) -> dict:
    with (root / "meta" / "info.json").open() as f:
        return json.load(f)


def _list_files(root: Path, template: str, **fmt_extra) -> list[tuple[FileAddr, Path]]:
    """Enumerate existing files matching the template, sorted by (chunk, file)."""
    base = root
    out: list[tuple[FileAddr, Path]] = []
    chunk_glob = template.split("/chunk-")[0]
    chunk_dir = base / chunk_glob.format(**fmt_extra) if "{" in chunk_glob else base / chunk_glob
    if not chunk_dir.exists():
        return out
    for chunk_path in sorted(chunk_dir.iterdir()):
        if not chunk_path.is_dir() or not chunk_path.name.startswith("chunk-"):
            continue
        try:
            chunk_idx = int(chunk_path.name.removeprefix("chunk-"))
        except ValueError:
            continue
        for file_path in sorted(chunk_path.iterdir()):
            if not file_path.is_file():
                continue
            stem = file_path.stem
            if not stem.startswith("file-"):
                continue
            try:
                file_idx = int(stem.removeprefix("file-"))
            except ValueError:
                continue
            out.append((FileAddr(chunk_idx, file_idx), file_path))
    out.sort(key=lambda t: (t[0].chunk, t[0].file))
    return out


def _list_data_files(root: Path) -> list[tuple[FileAddr, Path]]:
    return _list_files(root, DATA_PATH_TMPL.split("/chunk-")[0])


def _list_video_files(root: Path, key: str) -> list[tuple[FileAddr, Path]]:
    base = root / "videos" / key
    out: list[tuple[FileAddr, Path]] = []
    if not base.exists():
        return out
    for chunk_path in sorted(base.iterdir()):
        if not chunk_path.is_dir() or not chunk_path.name.startswith("chunk-"):
            continue
        chunk_idx = int(chunk_path.name.removeprefix("chunk-"))
        for file_path in sorted(chunk_path.iterdir()):
            if not file_path.is_file() or file_path.suffix != ".mp4":
                continue
            stem = file_path.stem
            if not stem.startswith("file-"):
                continue
            file_idx = int(stem.removeprefix("file-"))
            out.append((FileAddr(chunk_idx, file_idx), file_path))
    out.sort(key=lambda t: (t[0].chunk, t[0].file))
    return out


def _list_meta_episode_files(root: Path) -> list[tuple[FileAddr, Path]]:
    base = root / "meta" / "episodes"
    out: list[tuple[FileAddr, Path]] = []
    if not base.exists():
        return out
    for chunk_path in sorted(base.iterdir()):
        if not chunk_path.is_dir() or not chunk_path.name.startswith("chunk-"):
            continue
        chunk_idx = int(chunk_path.name.removeprefix("chunk-"))
        for file_path in sorted(chunk_path.iterdir()):
            if not file_path.is_file() or file_path.suffix != ".parquet":
                continue
            stem = file_path.stem
            if not stem.startswith("file-"):
                continue
            file_idx = int(stem.removeprefix("file-"))
            out.append((FileAddr(chunk_idx, file_idx), file_path))
    out.sort(key=lambda t: (t[0].chunk, t[0].file))
    return out


def _video_keys(info: dict) -> list[str]:
    return [k for k, v in info.get("features", {}).items() if v.get("dtype") == "video"]


def _validate_compatible(infos: list[dict]) -> None:
    """Raise if the inputs disagree on anything we can't reconcile."""
    if not infos:
        raise ValueError("no input datasets")
    ref = infos[0]
    ref_video_keys = sorted(_video_keys(ref))
    for i, other in enumerate(infos[1:], start=1):
        for key in ("codebase_version", "fps", "chunks_size", "robot_type"):
            if other.get(key) != ref.get(key):
                raise ValueError(
                    f"dataset[{i}] {key}={other.get(key)!r} disagrees "
                    f"with dataset[0] {key}={ref.get(key)!r}"
                )
        ref_feats = ref["features"]
        other_feats = other["features"]
        if set(ref_feats) != set(other_feats):
            raise ValueError(
                f"dataset[{i}] features {sorted(other_feats)} disagree "
                f"with dataset[0] {sorted(ref_feats)}"
            )
        for fname, fspec in ref_feats.items():
            if other_feats[fname].get("dtype") != fspec.get("dtype"):
                raise ValueError(f"feature {fname!r} dtype mismatch")
            if other_feats[fname].get("shape") != fspec.get("shape"):
                raise ValueError(f"feature {fname!r} shape mismatch")
        if sorted(_video_keys(other)) != ref_video_keys:
            raise ValueError(f"dataset[{i}] video keys disagree")


def _build_task_table(roots: list[Path]) -> tuple[pd.DataFrame, list[dict[int, int]]]:
    """Return (merged_tasks_df, per_input_task_remap).

    `merged_tasks_df` matches the on-disk shape: index = task string,
    column = ``task_index``. `per_input_task_remap[i]` is a dict mapping
    each input's original ``task_index`` -> the new task_index.
    """
    seen: dict[str, int] = {}
    remaps: list[dict[int, int]] = []
    for root in roots:
        df = pd.read_parquet(root / "meta" / "tasks.parquet")
        df = df.reset_index()  # `task` becomes a column, plus `task_index`
        remap: dict[int, int] = {}
        for _, row in df.iterrows():
            task = row["task"]
            old_idx = int(row["task_index"])
            if task not in seen:
                seen[task] = len(seen)
            remap[old_idx] = seen[task]
        remaps.append(remap)
    merged = pd.DataFrame(
        {"task_index": list(seen.values())},
        index=pd.Index(list(seen.keys()), name="task"),
    )
    return merged, remaps


@dataclass
class _DatasetOffsets:
    episode_offset: int
    frame_offset: int
    data_file_seq_offset: int
    video_file_seq_offset: dict[str, int]
    meta_ep_file_seq_offset: int


def _seq_index(addr: FileAddr, chunks_size: int) -> int:
    return addr.chunk * chunks_size + addr.file


def _addr_from_seq(seq: int, chunks_size: int) -> FileAddr:
    return FileAddr(seq // chunks_size, seq % chunks_size)


def _pool_stats(per_episode: list[dict]) -> dict:
    """Aggregate per-episode stats dicts into one dataset-level stats dict.

    Each per-episode entry maps feature name -> {min, max, mean, std,
    count, q01, q10, q50, q90, q99} where every value is a list of
    floats (one per channel). We compute:
        min  -> elementwise min
        max  -> elementwise max
        count -> sum
        mean -> count-weighted mean
        std  -> pooled std (within-episode variance + between-episode
                shift, both count-weighted)
        qNN  -> count-weighted average (exact recompute would need the
                raw data; this matches what lerobot itself uses for
                resumed datasets)
    """
    if not per_episode:
        return {}
    feature_names = list(per_episode[0].keys())
    out: dict[str, dict[str, list[float]]] = {}
    for fname in feature_names:
        mins = [np.asarray(ep[fname]["min"], dtype=np.float64) for ep in per_episode]
        maxs = [np.asarray(ep[fname]["max"], dtype=np.float64) for ep in per_episode]
        means = [np.asarray(ep[fname]["mean"], dtype=np.float64) for ep in per_episode]
        stds = [np.asarray(ep[fname]["std"], dtype=np.float64) for ep in per_episode]
        counts = np.asarray([float(ep[fname]["count"][0]) for ep in per_episode], dtype=np.float64)
        total = float(counts.sum())
        if total == 0:
            continue
        stacked_mean = np.stack(means, axis=0)
        weights = (counts / total).reshape(-1, *([1] * (stacked_mean.ndim - 1)))
        pooled_mean = (stacked_mean * weights).sum(axis=0)
        stacked_var = np.stack([s**2 for s in stds], axis=0)
        within = (stacked_var * weights).sum(axis=0)
        between = (((stacked_mean - pooled_mean) ** 2) * weights).sum(axis=0)
        pooled_std = np.sqrt(np.maximum(within + between, 0.0))
        feat = {
            "min": np.min(np.stack(mins, axis=0), axis=0).tolist(),
            "max": np.max(np.stack(maxs, axis=0), axis=0).tolist(),
            "mean": pooled_mean.tolist(),
            "std": pooled_std.tolist(),
            "count": [int(total)],
        }
        for q in ("q01", "q10", "q50", "q90", "q99"):
            qs = np.stack(
                [np.asarray(ep[fname][q], dtype=np.float64) for ep in per_episode],
                axis=0,
            )
            feat[q] = (qs * weights).sum(axis=0).tolist()
        out[fname] = feat
    return out


def _to_pure_nested(v) -> list:
    """Convert ndarrays (including object-dtype arrays of ndarrays, used by
    the per-channel image stats) into plain nested Python lists of floats."""
    if isinstance(v, np.ndarray):
        if v.dtype == object:
            return [_to_pure_nested(x) for x in v.tolist()]
        return v.tolist()
    if isinstance(v, (list, tuple)):
        return [_to_pure_nested(x) for x in v]
    if isinstance(v, (np.floating, np.integer)):
        return float(v)
    return v


def _ep_stats_from_row(row: pd.Series, feature_names: Iterable[str]) -> dict:
    """Pull the `stats/<feat>/<metric>` columns out of an episode row."""
    out: dict[str, dict[str, list[float]]] = {}
    for fname in feature_names:
        feat = {}
        prefix = f"stats/{fname}/"
        for metric in ("min", "max", "mean", "std", "count", "q01", "q10", "q50", "q90", "q99"):
            col = prefix + metric
            if col not in row:
                feat = None
                break
            converted = _to_pure_nested(row[col])
            if not isinstance(converted, list):
                converted = [converted]
            feat[metric] = converted
        if feat is not None:
            out[fname] = feat
    return out


def merge_datasets(
    inputs: list[Path],
    output: Path,
    *,
    repo_id: str | None = None,
    overwrite: bool = False,
) -> dict:
    """Merge ``inputs`` into a fresh dataset at ``output``.

    Returns the merged ``info.json`` dict.
    """
    inputs = [Path(p) for p in inputs]
    output = Path(output)
    if len(inputs) < 1:
        raise ValueError("need at least one input dataset")
    for p in inputs:
        if not (p / "meta" / "info.json").exists():
            raise FileNotFoundError(f"missing meta/info.json in {p}")
    if output.exists():
        if not overwrite:
            raise FileExistsError(
                f"{output} already exists (pass overwrite=True / --overwrite to wipe)"
            )
        shutil.rmtree(output)
    output.mkdir(parents=True)

    infos = [_read_info(p) for p in inputs]
    _validate_compatible(infos)
    ref = infos[0]
    chunks_size = int(ref["chunks_size"])
    video_keys = sorted(_video_keys(ref))
    feature_names = list(ref["features"].keys())

    merged_tasks, task_remaps = _build_task_table(inputs)

    cumulative_episodes = 0
    cumulative_frames = 0
    cumulative_data_files = 0
    cumulative_video_files: dict[str, int] = {k: 0 for k in video_keys}
    cumulative_meta_ep_files = 0
    all_episode_rows: list[pd.DataFrame] = []

    for ds_idx, (root, info) in enumerate(zip(inputs, infos)):
        offsets = _DatasetOffsets(
            episode_offset=cumulative_episodes,
            frame_offset=cumulative_frames,
            data_file_seq_offset=cumulative_data_files,
            video_file_seq_offset=dict(cumulative_video_files),
            meta_ep_file_seq_offset=cumulative_meta_ep_files,
        )
        logger.info(
            "merging dataset[%d] from %s (eps=%d, frames=%d) at offsets %s",
            ds_idx, root, info["total_episodes"], info["total_frames"], offsets,
        )

        # --- data parquets -------------------------------------------------
        data_files = _list_data_files(root)
        data_seq_remap: dict[tuple[int, int], FileAddr] = {}
        for local_seq, (addr, src) in enumerate(data_files):
            new_addr = _addr_from_seq(local_seq + offsets.data_file_seq_offset, chunks_size)
            data_seq_remap[(addr.chunk, addr.file)] = new_addr
            df = pd.read_parquet(src)
            df["index"] = df["index"].astype(np.int64) + offsets.frame_offset
            df["episode_index"] = (
                df["episode_index"].astype(np.int64) + offsets.episode_offset
            )
            df["task_index"] = df["task_index"].map(task_remaps[ds_idx]).astype(np.int64)
            dst = output / DATA_PATH_TMPL.format(chunk=new_addr.chunk, file=new_addr.file)
            dst.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(dst, index=False)

        # --- videos --------------------------------------------------------
        video_seq_remap: dict[str, dict[tuple[int, int], FileAddr]] = {}
        for key in video_keys:
            video_files = _list_video_files(root, key)
            remap: dict[tuple[int, int], FileAddr] = {}
            for local_seq, (addr, src) in enumerate(video_files):
                new_addr = _addr_from_seq(
                    local_seq + offsets.video_file_seq_offset[key], chunks_size
                )
                remap[(addr.chunk, addr.file)] = new_addr
                dst = output / VIDEO_PATH_TMPL.format(
                    key=key, chunk=new_addr.chunk, file=new_addr.file
                )
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
            video_seq_remap[key] = remap

        # --- meta/episodes -------------------------------------------------
        meta_ep_files = _list_meta_episode_files(root)
        for local_seq, (addr, src) in enumerate(meta_ep_files):
            new_addr = _addr_from_seq(
                local_seq + offsets.meta_ep_file_seq_offset, chunks_size
            )
            df = pd.read_parquet(src)
            df["episode_index"] = df["episode_index"].astype(np.int64) + offsets.episode_offset
            df["dataset_from_index"] = (
                df["dataset_from_index"].astype(np.int64) + offsets.frame_offset
            )
            df["dataset_to_index"] = (
                df["dataset_to_index"].astype(np.int64) + offsets.frame_offset
            )
            # `data/chunk_index` + `data/file_index` -> remapped
            new_data_chunks = []
            new_data_files = []
            for ch, fi in zip(df["data/chunk_index"], df["data/file_index"]):
                addr2 = data_seq_remap[(int(ch), int(fi))]
                new_data_chunks.append(addr2.chunk)
                new_data_files.append(addr2.file)
            df["data/chunk_index"] = pd.Series(new_data_chunks, dtype=np.int64)
            df["data/file_index"] = pd.Series(new_data_files, dtype=np.int64)
            for key in video_keys:
                ck_col = f"videos/{key}/chunk_index"
                fi_col = f"videos/{key}/file_index"
                new_ck = []
                new_fi = []
                for ch, fi in zip(df[ck_col], df[fi_col]):
                    addr2 = video_seq_remap[key][(int(ch), int(fi))]
                    new_ck.append(addr2.chunk)
                    new_fi.append(addr2.file)
                df[ck_col] = pd.Series(new_ck, dtype=np.int64)
                df[fi_col] = pd.Series(new_fi, dtype=np.int64)
            # meta/episodes own file_index pointer
            if "meta/episodes/chunk_index" in df.columns:
                df["meta/episodes/chunk_index"] = np.int64(new_addr.chunk)
                df["meta/episodes/file_index"] = np.int64(new_addr.file)
            dst = output / META_EP_PATH_TMPL.format(chunk=new_addr.chunk, file=new_addr.file)
            dst.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(dst, index=False)
            all_episode_rows.append(df)

        cumulative_episodes += int(info["total_episodes"])
        cumulative_frames += int(info["total_frames"])
        cumulative_data_files += len(data_files)
        for key in video_keys:
            cumulative_video_files[key] += len(_list_video_files(root, key))
        cumulative_meta_ep_files += len(meta_ep_files)

    # --- meta/tasks.parquet ----------------------------------------------
    merged_tasks.to_parquet(output / "meta" / "tasks.parquet")

    # --- meta/info.json ---------------------------------------------------
    merged_info = dict(ref)
    merged_info["total_episodes"] = cumulative_episodes
    merged_info["total_frames"] = cumulative_frames
    merged_info["total_tasks"] = int(len(merged_tasks))
    merged_info["splits"] = {"train": f"0:{cumulative_episodes}"}
    if repo_id is not None:
        merged_info["repo_id"] = repo_id
    with (output / "meta" / "info.json").open("w") as f:
        json.dump(merged_info, f, indent=4)

    # --- meta/stats.json --------------------------------------------------
    merged_eps = pd.concat(all_episode_rows, ignore_index=True) if all_episode_rows else pd.DataFrame()
    per_episode_stats = [
        _ep_stats_from_row(row, feature_names) for _, row in merged_eps.iterrows()
    ]
    stats = _pool_stats(per_episode_stats)
    with (output / "meta" / "stats.json").open("w") as f:
        json.dump(stats, f, indent=4)

    logger.info(
        "merged %d datasets -> %s: %d episodes, %d frames, %d tasks",
        len(inputs), output, cumulative_episodes, cumulative_frames, len(merged_tasks),
    )
    return merged_info


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Merge LeRobotDataset v3.0 directories produced by utils/data_collection.",
    )
    ap.add_argument("inputs", nargs="+", type=Path, help="paths to input dataset roots")
    ap.add_argument("-o", "--output", type=Path, required=True, help="path to write the merged dataset")
    ap.add_argument("--repo-id", type=str, default=None, help="set repo_id field in merged info.json")
    ap.add_argument("--overwrite", action="store_true", help="wipe output if it exists")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        merge_datasets(args.inputs, args.output, repo_id=args.repo_id, overwrite=args.overwrite)
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
