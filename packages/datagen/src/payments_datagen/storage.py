"""Reading and writing a generated universe as a directory of Parquet files."""

from pathlib import Path

import polars as pl

from payments_datagen.generate import DATASET_PATHS, World
from payments_datagen.manifest import MANIFEST_NAME, build_manifest, write_manifest


def write_world(world: World, out_dir: Path) -> dict:
    """Write every dataset (one zstd Parquet file each) and the manifest; return the manifest."""
    for name, relative in DATASET_PATHS.items():
        path = out_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        world.frames[name].write_parquet(path, compression="zstd")
    manifest = build_manifest(world)
    write_manifest(manifest, out_dir / MANIFEST_NAME)
    return manifest


def read_datasets(out_dir: Path) -> dict[str, pl.DataFrame]:
    """Read every dataset; a missing file raises ``FileNotFoundError`` naming it."""
    frames = {}
    for name, relative in DATASET_PATHS.items():
        path = out_dir / relative
        if not path.is_file():
            raise FileNotFoundError(f"dataset {name!r} is missing: {path}")
        frames[name] = pl.read_parquet(path)
    return frames
