# merge_datasets

Merge two or more `LeRobotDataset` v3.0 directories produced by
`utils/data_collection` into a single dataset.

The teleoperator's recorder writes one dataset per session under
`data/<repo_id>/`. When you collect across multiple machines or
multiple sessions and want a single training corpus, this tool
concatenates them without re-encoding the videos.

## What it does

For each input dataset it streams:

- `data/chunk-XXX/file-YYY.parquet`  — frame rows; `index`,
  `episode_index`, `task_index` are remapped to the merged global space.
- `videos/<key>/chunk-XXX/file-YYY.mp4` — copied byte-for-byte;
  filenames are renumbered to continue from the previous dataset.
- `meta/episodes/chunk-XXX/file-YYY.parquet` — episode metadata;
  `episode_index`, `dataset_from_index`, `dataset_to_index`, and the
  data/video chunk+file pointers are all remapped.

It then rewrites:

- `meta/tasks.parquet` — union of input task lists (deduped by task
  string; reassigns `task_index`).
- `meta/info.json` — `total_episodes`, `total_frames`, `total_tasks`,
  and the `train` split are recomputed.
- `meta/stats.json` — count-weighted pool of the per-episode stats
  stored in the input episode parquets.

## Compatibility

Inputs must agree on: `codebase_version`, `fps`, `chunks_size`,
`robot_type`, feature dtypes/shapes, and the set of video keys.
Mismatch is a hard error — a teleop run with a different state schema
cannot be merged.

Stats for the camera features are an approximation (count-weighted
average of the per-episode percentiles); we never decode the videos.
For min/max/mean/count this is exact.

## Usage

```bash
# from utils/merge_datasets/
uv run python merge_datasets.py \
    /path/to/dataset_a \
    /path/to/dataset_b \
    /path/to/dataset_c \
    --output /path/to/merged \
    --repo-id you/your-merged-repo
```

Pass `--overwrite` to clobber an existing output directory. Pass `-v`
for debug logging.

Python entry point:

```python
from merge_datasets import merge_datasets

merge_datasets(
    inputs=["/path/to/a", "/path/to/b"],
    output="/path/to/merged",
    repo_id="you/your-merged-repo",
    overwrite=True,
)
```
